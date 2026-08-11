"""Is fa2 prefill ignoring causal=True? Compare against BOTH references."""

import flashinfer
import torch
import torch.nn.functional as F

torch.manual_seed(7)
DEV, DT, HS, NQ, NKV = "cuda", torch.bfloat16, 128, 32, 8
seq = 128

q = torch.randn(seq, NQ, HS, dtype=DT, device=DEV)
k = torch.randn(seq, NKV, HS, dtype=DT, device=DEV)
v = torch.randn(seq, NKV, HS, dtype=DT, device=DEV)


def sdpa(causal):
    rep = NQ // NKV
    m = None
    if causal:
        i = torch.arange(seq, device=DEV)
        m = (i[None, :] <= i[:, None])[None, :, :]
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0).float(),
        k.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0).float(),
        v.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0).float(),
        attn_mask=m,
    )
    return o.squeeze(0).transpose(0, 1).to(DT)


ref_causal = sdpa(True)
ref_full = sdpa(False)
dd = lambda a, b: (a.float() - b.float()).abs().max().item()  # noqa: E731

print(f"flashinfer {flashinfer.__version__}")
print(f"sanity: causal vs non-causal refs differ by {dd(ref_causal, ref_full):.3e}\n")

for be in ("fa2", "aiter"):
    for causal in (True, False):
        try:
            o = flashinfer.single_prefill_with_kv_cache(
                q, k, v, causal=causal, backend=be
            )
            print(
                f"  single_prefill backend={be:<6} causal={str(causal):<5} "
                f"vs_ref_causal={dd(o, ref_causal):>9.3e}   "
                f"vs_ref_full={dd(o, ref_full):>9.3e}"
            )
        except Exception as e:
            print(f"  single_prefill backend={be:<6} causal={causal} ERR {e}")
