"""Does backend="auto" resolve to fa2 on ROCm prefill?

Bitwise comparison, not log-scraping: if auto's output is byte-identical to
fa2's and differs from aiter's, auto resolved to fa2.
"""

import flashinfer
import torch
import torch.nn.functional as F

DEV, DT, HS, NQ, NKV, S = "cuda", torch.bfloat16, 128, 32, 8, 128
torch.manual_seed(11)

q = torch.randn(S, NQ, HS, dtype=DT, device=DEV)
k = torch.randn(S, NKV, HS, dtype=DT, device=DEV)
v = torch.randn(S, NKV, HS, dtype=DT, device=DEV)

rep = NQ // NKV
i = torch.arange(S, device=DEV)
ref = (
    F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0).float(),
        k.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0).float(),
        v.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0).float(),
        attn_mask=(i[None, :] <= i[:, None])[None, :, :],
    )
    .squeeze(0)
    .transpose(0, 1)
    .to(DT)
)

outs = {}
# Note: omitting `backend` entirely, to exercise the true library default
# rather than an explicitly-passed "auto".
outs["<omitted>"] = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
for be in ("auto", "fa2", "aiter"):
    outs[be] = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend=be)

print(f"flashinfer {flashinfer.__version__}")
print(f"gpu {torch.cuda.get_device_properties(0).gcnArchName}\n")

d = lambda a, b: (a.float() - b.float()).abs().max().item()  # noqa: E731
for name, o in outs.items():
    print(f"  backend={name:<11} max_abs_vs_sdpa = {d(o, ref):.4e}")

print("\n  bitwise identity:")
for name in ("<omitted>", "auto"):
    print(
        f"    {name:<11} == fa2   : {torch.equal(outs[name], outs['fa2'])}"
        f"     == aiter : {torch.equal(outs[name], outs['aiter'])}"
    )
