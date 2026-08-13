"""Phase 0 gate: does amd-flashinfer produce correct paged prefill/decode numerics?

Mirrors the layout vLLM's FlashInfer backend uses:
  logical kv cache (num_blocks, num_kv_heads, block_size, 2*head_size)
  -> NHD physical  (num_blocks, block_size, num_kv_heads, 2*head_size)
  -> split into (k_cache, v_cache) tuple for wrapper.run()
Reference is torch SDPA over the gathered (unpaged) KV.
"""

import inspect

import flashinfer
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda"


def make_paged_kv(seq_lens, num_kv_heads, head_size, block_size, dtype):
    """Build a paged KV cache + index structures for a batch."""
    num_blocks_per_seq = [(s + block_size - 1) // block_size for s in seq_lens]
    total_blocks = sum(num_blocks_per_seq) + 4  # a little slack
    # logical (B, H, N, 2*hs) with NHD physical layout, exactly like vLLM
    logical = (total_blocks, num_kv_heads, block_size, 2 * head_size)
    stride_order = (0, 2, 1, 3)  # NHD
    phys = tuple(logical[i] for i in stride_order)
    inv = [stride_order.index(i) for i in range(len(stride_order))]
    kv_cache = torch.zeros(phys, dtype=dtype, device=DEV).permute(*inv)

    indptr = [0]
    indices = []
    last_page_len = []
    blk = 0
    for s, nb in zip(seq_lens, num_blocks_per_seq):
        indices.extend(range(blk, blk + nb))
        blk += nb
        indptr.append(len(indices))
        lpl = s - (nb - 1) * block_size
        last_page_len.append(lpl)

    # reference KV, and scatter it into the paged cache
    ks, vs = [], []
    for i, s in enumerate(seq_lens):
        k = torch.randn(s, num_kv_heads, head_size, dtype=dtype, device=DEV)
        v = torch.randn(s, num_kv_heads, head_size, dtype=dtype, device=DEV)
        ks.append(k)
        vs.append(v)
        base = indptr[i]
        for t in range(s):
            b = indices[base + t // block_size]
            off = t % block_size
            kv_cache[b, :, off, :head_size] = k[t]
            kv_cache[b, :, off, head_size:] = v[t]

    return (
        kv_cache,
        torch.tensor(indptr, dtype=torch.int32, device=DEV),
        torch.tensor(indices, dtype=torch.int32, device=DEV),
        torch.tensor(last_page_len, dtype=torch.int32, device=DEV),
        ks,
        vs,
    )


def as_tuple(kv_cache, head_size):
    """(B,H,N,2*hs) -> ((B,N,H,hs),(B,N,H,hs)) — what vLLM hands the wrapper."""
    return kv_cache.transpose(1, 2).split(head_size, dim=-1)


def sdpa_ref(q, k, v, causal, q_offset):
    """q:(nq,H,D) k/v:(s,Hkv,D). GQA via repeat_interleave."""
    nq, H, D = q.shape
    s, Hkv, _ = k.shape
    rep = H // Hkv
    kk = k.repeat_interleave(rep, dim=1)
    vv = v.repeat_interleave(rep, dim=1)
    attn_mask = None
    if causal:
        qi = torch.arange(nq, device=DEV) + q_offset
        ki = torch.arange(s, device=DEV)
        attn_mask = (ki[None, :] <= qi[:, None])[None, :, :]
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0).float(),
        kk.transpose(0, 1).unsqueeze(0).float(),
        vv.transpose(0, 1).unsqueeze(0).float(),
        attn_mask=attn_mask,
    )
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


def check(name, got, want, dtype):
    atol, rtol = (5e-3, 1e-2) if dtype == torch.float16 else (2e-2, 3e-2)
    ok = torch.allclose(got.float(), want.float(), atol=atol, rtol=rtol)
    diff = (got.float() - want.float()).abs()
    print(
        f"  {name:<34} {'PASS' if ok else 'FAIL'}  "
        f"max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}"
    )
    return ok


def run_prefill(
    dtype, backend, num_qo_heads=32, num_kv_heads=8, head_size=128, block_size=16
):
    seq_lens = [37, 128, 5]
    kv_cache, indptr, indices, lpl, ks, vs = make_paged_kv(
        seq_lens, num_kv_heads, head_size, block_size, dtype
    )
    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int32, device=DEV
    )
    q = torch.randn(sum(seq_lens), num_qo_heads, head_size, dtype=dtype, device=DEV)

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    kwargs = {}
    if backend is not None:
        kwargs["backend"] = backend
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", **kwargs)
    w.plan(
        qo_indptr,
        indptr,
        indices,
        lpl,
        num_qo_heads,
        num_kv_heads,
        head_size,
        block_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out = w.run(q, as_tuple(kv_cache, head_size))

    ok = True
    for i, s in enumerate(seq_lens):
        lo, hi = qo_indptr[i].item(), qo_indptr[i + 1].item()
        ref = sdpa_ref(q[lo:hi], ks[i], vs[i], causal=True, q_offset=0)
        ok &= check(f"prefill[seq{i} len={s}]", out[lo:hi], ref, dtype)
    return ok


def run_decode(
    dtype, backend, num_qo_heads=32, num_kv_heads=8, head_size=128, block_size=16
):
    seq_lens = [37, 128, 5]
    kv_cache, indptr, indices, lpl, ks, vs = make_paged_kv(
        seq_lens, num_kv_heads, head_size, block_size, dtype
    )
    q = torch.randn(len(seq_lens), num_qo_heads, head_size, dtype=dtype, device=DEV)

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    sig = inspect.signature(flashinfer.BatchDecodeWithPagedKVCacheWrapper.__init__)
    kwargs = {}
    if backend is not None and "backend" in sig.parameters:
        kwargs["backend"] = backend
    if "use_tensor_cores" in sig.parameters:
        kwargs["use_tensor_cores"] = False
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", **kwargs)
    w.plan(
        indptr,
        indices,
        lpl,
        num_qo_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out = w.run(q, as_tuple(kv_cache, head_size))

    ok = True
    for i, s in enumerate(seq_lens):
        ref = sdpa_ref(q[i : i + 1], ks[i], vs[i], causal=False, q_offset=0)
        ok &= check(f"decode[seq{i} len={s}]", out[i : i + 1], ref, dtype)
    return ok


def main():
    print(f"flashinfer {flashinfer.__version__}")
    print(f"gpu {torch.cuda.get_device_properties(0).gcnArchName}")
    # 0.5.3+amd.1 exposed HAS_AITER; later builds replaced it with
    # is_aiter_supported(), the helper the README always documented.
    try:
        from flashinfer.aiter_utils import HAS_AITER

        print(f"HAS_AITER={HAS_AITER}")
    except ImportError:
        from flashinfer.aiter_utils import is_aiter_supported

        print(f"is_aiter_supported={is_aiter_supported(torch.device('cuda'))}")
    dec_sig = inspect.signature(flashinfer.BatchDecodeWithPagedKVCacheWrapper.__init__)
    print(f"decode __init__ params: {list(dec_sig.parameters)}")

    results = {}
    for dtype in (torch.float16, torch.bfloat16):
        for backend in (None, "fa2", "aiter"):
            tag = f"{str(dtype).split('.')[-1]}/{backend or 'default'}"
            print(f"\n=== PREFILL {tag} ===")
            try:
                results[f"prefill {tag}"] = run_prefill(dtype, backend)
            except Exception as e:
                print(f"  EXCEPTION {type(e).__name__}: {str(e)[:200]}")
                results[f"prefill {tag}"] = None
            print(f"=== DECODE {tag} ===")
            try:
                results[f"decode {tag}"] = run_decode(dtype, backend)
            except Exception as e:
                print(f"  EXCEPTION {type(e).__name__}: {str(e)[:200]}")
                results[f"decode {tag}"] = None

    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(f"  {k:<32} {'PASS' if v else ('FAIL' if v is False else 'ERROR')}")


if __name__ == "__main__":
    main()
