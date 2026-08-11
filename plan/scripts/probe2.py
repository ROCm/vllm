"""Confirm the fa2-prefill divergence without relying on the SDPA reference.

Controls:
  1. fa2 vs aiter prefill, same inputs -> compare to each other directly.
  2. Vary page_size (1, 16, 32, 64) and exact-multiple vs partial last page.
  3. Single-sequence batch (removes batching/indptr from suspicion).
  4. ragged prefill wrapper + single_prefill_with_kv_cache (non-paged paths)
     to localize whether the bug is in paging or in the fa2 prefill kernel.
"""

import flashinfer
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda"
DT = torch.bfloat16
HS = 128
NQ, NKV = 32, 8


def build(seq_lens, block_size):
    nb_per = [(s + block_size - 1) // block_size for s in seq_lens]
    total = sum(nb_per)
    logical = (total, NKV, block_size, 2 * HS)
    so = (0, 2, 1, 3)
    phys = tuple(logical[i] for i in so)
    inv = [so.index(i) for i in range(4)]
    kv = torch.zeros(phys, dtype=DT, device=DEV).permute(*inv)
    indptr, indices, lpl = [0], [], []
    blk = 0
    ks, vs = [], []
    for s, nb in zip(seq_lens, nb_per):
        indices.extend(range(blk, blk + nb))
        blk += nb
        indptr.append(len(indices))
        lpl.append(s - (nb - 1) * block_size)
        k = torch.randn(s, NKV, HS, dtype=DT, device=DEV)
        v = torch.randn(s, NKV, HS, dtype=DT, device=DEV)
        ks.append(k)
        vs.append(v)
        base = indptr[-2]
        for t in range(s):
            b = indices[base + t // block_size]
            kv[b, :, t % block_size, :HS] = k[t]
            kv[b, :, t % block_size, HS:] = v[t]
    t32 = lambda x: torch.tensor(x, dtype=torch.int32, device=DEV)  # noqa: E731
    return kv, t32(indptr), t32(indices), t32(lpl), ks, vs


def sdpa(q, k, v):
    nq, H, D = q.shape
    rep = H // k.shape[1]
    kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    qi = torch.arange(nq, device=DEV)
    ki = torch.arange(k.shape[0], device=DEV)
    m = (ki[None, :] <= qi[:, None])[None, :, :]
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0).float(),
        kk.transpose(0, 1).unsqueeze(0).float(),
        vv.transpose(0, 1).unsqueeze(0).float(),
        attn_mask=m,
    )
    return o.squeeze(0).transpose(0, 1).to(q.dtype)


def paged_prefill(backend, seq_lens, block_size, q):
    kv, indptr, indices, lpl, ks, vs = build(seq_lens, block_size)
    qo = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int32, device=DEV
    )
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=DEV)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
    w.plan(
        qo,
        indptr,
        indices,
        lpl,
        NQ,
        NKV,
        HS,
        block_size,
        causal=True,
        q_data_type=DT,
        kv_data_type=DT,
    )
    return w.run(q, kv.transpose(1, 2).split(HS, dim=-1)), ks, vs, qo


def d(a, b):
    return (a.float() - b.float()).abs().max().item()


print(f"flashinfer {flashinfer.__version__}  dtype={DT}")

print("\n--- (1) fa2 vs aiter vs sdpa, single seq, exact page multiple ---")
for block_size in (1, 16, 32, 64):
    for seq in (128, 130):
        torch.manual_seed(1)
        q = torch.randn(seq, NQ, HS, dtype=DT, device=DEV)
        outs = {}
        for be in ("fa2", "aiter"):
            torch.manual_seed(1)  # identical kv
            try:
                o, ks, vs, qo = paged_prefill(be, [seq], block_size, q)
                outs[be] = o
            except Exception as e:
                outs[be] = f"ERR {type(e).__name__}: {str(e)[:60]}"
        torch.manual_seed(1)
        _, ks, vs, _ = paged_prefill("aiter", [seq], block_size, q)
        ref = sdpa(q, ks[0], vs[0])
        exact = "exact" if seq % block_size == 0 else "partial"
        line = f"  page={block_size:<3} seq={seq:<4} ({exact:>7}) "
        for be in ("fa2", "aiter"):
            o = outs[be]
            line += (
                f"{be}_vs_sdpa={d(o, ref):>9.3e}  "
                if torch.is_tensor(o)
                else f"{be}={o}  "
            )
        if all(torch.is_tensor(outs[b]) for b in ("fa2", "aiter")):
            line += f"fa2_vs_aiter={d(outs['fa2'], outs['aiter']):.3e}"
        print(line)

print("\n--- (2) non-paged paths (localize: paging vs prefill kernel) ---")
torch.manual_seed(2)
seq = 128
q = torch.randn(seq, NQ, HS, dtype=DT, device=DEV)
k = torch.randn(seq, NKV, HS, dtype=DT, device=DEV)
v = torch.randn(seq, NKV, HS, dtype=DT, device=DEV)
ref = sdpa(q, k, v)

for be in ("fa2", "aiter"):
    try:
        o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend=be)
        print(f"  single_prefill      backend={be:<6} max_abs_vs_sdpa={d(o, ref):.3e}")
    except Exception as e:
        print(
            f"  single_prefill      backend={be:<6} "
            f"ERR {type(e).__name__}: {str(e)[:80]}"
        )

for be in ("fa2", "aiter"):
    try:
        ws = torch.empty(256 << 20, dtype=torch.uint8, device=DEV)
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend=be)
        ind = torch.tensor([0, seq], dtype=torch.int32, device=DEV)
        w.plan(ind, ind, NQ, NKV, HS, causal=True, q_data_type=DT, kv_data_type=DT)
        o = w.run(q, k, v)
        print(f"  ragged_prefill      backend={be:<6} max_abs_vs_sdpa={d(o, ref):.3e}")
    except Exception as e:
        print(
            f"  ragged_prefill      backend={be:<6} "
            f"ERR {type(e).__name__}: {str(e)[:80]}"
        )

print("\n--- (3) decode control (expected: all good) ---")
torch.manual_seed(3)
seq_lens = [128]
qd = torch.randn(1, NQ, HS, dtype=DT, device=DEV)
for be in (None, "fa2"):
    torch.manual_seed(3)
    kv, indptr, indices, lpl, ks, vs = build(seq_lens, 16)
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=DEV)
    kw = {"backend": be} if be else {}
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", **kw)
    w.plan(indptr, indices, lpl, NQ, NKV, HS, 16, q_data_type=DT, kv_data_type=DT)
    o = w.run(qd, kv.transpose(1, 2).split(HS, dim=-1))
    rep = NQ // NKV
    r = (
        F.scaled_dot_product_attention(
            qd.transpose(0, 1).unsqueeze(0).float(),
            ks[0].repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0).float(),
            vs[0].repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0).float(),
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    print(f"  decode backend={str(be):<6} max_abs_vs_sdpa={d(o, r):.3e}")
