# Handoff: amd-flashinfer HIP `fa2` prefill returns incorrect results on gfx942

**Component:** `amd-flashinfer` 0.5.3+amd.1 ([AMD-Ecosystem/flashinfer](https://github.com/AMD-Ecosystem/flashinfer), branch `amd-integration`)
**Hardware:** AMD Instinct MI300X, `gfx942:sramecc+:xnack-`
**Found:** 2026-07-31, while integrating amd-flashinfer as a vLLM attention backend
**Severity:** High — wrong numerical output, no error raised, on the **default** code path
**Status:** findings #1 and #2 **FIXED** upstream; #3 open; two new regressions
opened. See "Retest" below before acting on anything in this document.

---

## Retest against a newer build (2026-08-12)

Everything below describes `0.5.3+amd.1`, the published wheel. It was re-run
against a source build of `amd-integration` at commit `afc7a613`
(`0.5.3+amd.2.dev55`), on MI300X / ROCm 7.2.3 / torch 2.12.0.

| paged prefill | `+amd.1` | `+amd.2.dev55` |
|---|---|---|
| `fa2` fp16 / bf16 | wrong (3.5) | **correct (1.6e-2)** ✅ |
| `aiter` fp16 | correct | **wrong (3.5)** ⚠️ new |
| `aiter` bf16 | correct | **fails to compile** ⚠️ new |
| `auto` resolves to | `fa2` | `aiter` |
| decode, all routes | correct | correct |

- **Finding #1 (fa2 prefill wrong) is FIXED.**
- **Finding #2 (`auto` never reaches AITER) is FIXED** — `auto` now routes to
  AITER, confirmed bitwise.
- **Finding #3 (`-DUSE_ROCM`) is still unfixed**; the retest required applying
  `scripts/patch_use_rocm.py` locally before anything would JIT on torch 2.12.
- **Two new regressions on the AITER prefill path.** bf16 fails to compile —
  the generated `batch_prefill_aiter_config.inc` uses `__hip_bfloat16`, which
  does not exist in ROCm 7.2.3 (`hip_bfloat16`, no leading underscores, does).
  fp16 compiles but returns max-abs ≈ 3.5 vs SDPA. Because `auto` now routes
  to AITER, the default path is still broken — the failure moved rather than
  closed.
- API change: `aiter_utils.HAS_AITER` was replaced by `is_aiter_supported()`.

Caveat: the retest ran on a locally patched build (`-DUSE_ROCM`). The bf16
compile error is clearly independent of that patch; the fp16 numerical result
would ideally be reproduced without it, on torch ≤ 2.10.

---

## Four findings in this document

1. **HIP `fa2` prefill returns wrong results.** Silent, no error. *High.*
2. **`auto` never routes prefill to AITER**, contrary to the README's feature
   matrix — the router has no AITER branch, so every caller who does not pass
   `backend="aiter"` explicitly gets finding #1. The ragged wrapper discards
   `"aiter"` even when passed. *High, and it is what makes #1 reachable.*
3. **JIT cannot compile against torch ≥2.11** — one missing `-DUSE_ROCM`.
   Blocks amd-flashinfer and current vLLM from coexisting at all. *Trivial fix.*
4. **No `fast_decode_plan` on the ROCm path** — missing optimization, costs
   ~37–42% on small-batch decode under CUDA graphs. *Enhancement.*

#1 and #2 compound and should probably be filed together: a wrong kernel that
nothing selects would be low severity, and a routing gap that lands on a
correct kernel would be cosmetic. #3 and #4 are independent.

---

## TL;DR

The HIP `fa2` **prefill** kernels produce numerically wrong output. The error is
structural (max abs ≈ 3.5 against a `torch.nn.functional.scaled_dot_product_attention`
reference), not a tolerance issue, and it is silent — no exception, no warning.

`backend="auto"` resolves to `fa2` on ROCm, so **the default route is the affected
one**. `backend="aiter"` prefill is correct. **Decode is correct on every route.**

This blocks any use of amd-flashinfer's own prefill kernels for LLM inference and
forces consumers to pin `backend="aiter"`, which in turn means correct prefill on
ROCm is AITER underneath rather than FlashInfer's own kernels.

---

## Affected surface

| Entry point | `backend="fa2"` | `backend="aiter"` |
|---|---|---|
| `single_prefill_with_kv_cache` | **wrong** | correct |
| `BatchPrefillWithPagedKVCacheWrapper` | **wrong** | correct |
| `BatchPrefillWithRaggedKVCacheWrapper` | **wrong** | **wrong** — `"aiter"` is coerced to `fa2` in `__init__` |
| `BatchDecodeWithPagedKVCacheWrapper` | correct | correct |

Independent of: dtype (fp16 and bf16 both affected), page size (1, 16, 32, 64),
whether the last page is full or partial, and `causal` True/False.

---

## Reproduction

Self-contained; no external files needed. Run on a gfx942 host with
`amd-flashinfer` installed.

```python
import torch
import torch.nn.functional as F
import flashinfer

torch.manual_seed(7)
DEV, DT, HS, NQ, NKV, S = "cuda", torch.bfloat16, 128, 32, 8, 128

q = torch.randn(S, NQ, HS, dtype=DT, device=DEV)
k = torch.randn(S, NKV, HS, dtype=DT, device=DEV)
v = torch.randn(S, NKV, HS, dtype=DT, device=DEV)

def sdpa(causal):
    rep = NQ // NKV
    m = None
    if causal:
        i = torch.arange(S, device=DEV)
        m = (i[None, :] <= i[:, None])[None, :, :]
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0).float(),
        k.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0).float(),
        v.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0).float(),
        attn_mask=m,
    )
    return o.squeeze(0).transpose(0, 1).to(DT)

ref_causal, ref_full = sdpa(True), sdpa(False)
d = lambda a, b: (a.float() - b.float()).abs().max().item()

for be in ("fa2", "aiter"):
    for causal in (True, False):
        o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal, backend=be)
        print(f"{be:<6} causal={causal!s:<5} "
              f"vs_causal_ref={d(o, ref_causal):>9.3e}  vs_full_ref={d(o, ref_full):>9.3e}")
```

### Observed output

```
fa2    causal=True  vs_causal_ref=3.746e+00   vs_full_ref=1.947e+00
fa2    causal=False vs_causal_ref=3.675e+00   vs_full_ref=1.159e+00
aiter  causal=True  vs_causal_ref=1.562e-02   vs_full_ref=3.816e+00
aiter  causal=False vs_causal_ref=3.816e+00   vs_full_ref=3.906e-03
```

Read this as: `aiter` lands on the correct reference in both mask modes
(1.6e-2 causal, 3.9e-3 non-causal) and is far from the wrong one, exactly as it
should. `fa2` is far from **both** references in **both** modes.

### `auto` never reaches AITER on prefill — the documented routing is absent

This is the second half of the problem, and arguably the more important half.
The README's feature matrix says prefill auto-routes to AITER for
fp16/bf16 + NHD + `pos_encoding_mode="NONE"`. **The shipped code does not do
that.** AITER is not reachable through `auto` at all, for any input.

Behaviour differs by entry point, all in `flashinfer/prefill_rocm.py`:

| entry point | how `backend` is resolved | effect of `auto` |
|---|---|---|
| `single_prefill_with_kv_cache` | `determine_attention_backend(...)` | → `fa2` |
| `BatchPrefillWithPagedKVCacheWrapper` | `if backend not in ("fa2","aiter"): backend = "fa2"` | → `fa2`, router never consulted |
| `BatchPrefillWithRaggedKVCacheWrapper` | `if backend != "fa2": backend = "fa2"` | → `fa2`, **and explicit `"aiter"` is silently discarded too** |

The router the first row calls is the unmodified upstream CUDA one
(`flashinfer/utils.py`):

```python
def determine_attention_backend(...) -> str:
    if is_sm90a_supported(device) and is_fa3_backend_supported(...):
        return "fa3"
    else:
        return "fa2"
```

`"aiter"` is not among its possible return values, and `is_sm90a_supported` is
False on ROCm, so it returns `"fa2"` unconditionally. No dtype, layout,
head-dim or positional-encoding property of the caller's input can change
this.

Confirmed bitwise rather than from logs — omitting `backend` entirely, which
is what real callers do, gives output byte-identical to `backend="fa2"`:

```
backend=<omitted>   max_abs_vs_sdpa = 3.5469e+00
backend=auto        max_abs_vs_sdpa = 3.5469e+00
backend=fa2         max_abs_vs_sdpa = 3.5469e+00
backend=aiter       max_abs_vs_sdpa = 1.5625e-02

torch.equal(omitted, fa2)   -> True
torch.equal(omitted, aiter) -> False
```

So the two defects compound: the kernel `auto` always lands on is the broken
one, and the correct kernel cannot be selected without passing
`backend="aiter"` explicitly — which is undocumented as a requirement, and
which the ragged wrapper ignores even when you do.

For scale, the two references differ from each other by 3.816e+00 — so the `fa2`
error is the same order as the entire causal-vs-non-causal difference.

### Paged prefill, same picture

With a paged KV cache laid out NHD and passed as a `(k_cache, v_cache)` tuple:

```
page=1   seq=128  (  exact) fa2_vs_sdpa=3.480e+00  aiter_vs_sdpa=1.562e-02  fa2_vs_aiter=3.480e+00
page=1   seq=130  (  exact) fa2_vs_sdpa=3.480e+00  aiter_vs_sdpa=1.562e-02  fa2_vs_aiter=3.480e+00
page=16  seq=128  (  exact) fa2_vs_sdpa=3.480e+00  aiter_vs_sdpa=1.562e-02  fa2_vs_aiter=3.480e+00
page=16  seq=130  (partial) fa2_vs_sdpa=3.480e+00  aiter_vs_sdpa=1.562e-02  fa2_vs_aiter=3.480e+00
page=32  seq=128  (  exact) fa2_vs_sdpa=3.480e+00  aiter_vs_sdpa=1.562e-02  fa2_vs_aiter=3.480e+00
page=32  seq=130  (partial) fa2_vs_sdpa=3.480e+00  aiter_vs_sdpa=1.562e-02  fa2_vs_aiter=3.480e+00
page=64  seq=128  (  exact) fa2_vs_sdpa=3.480e+00  aiter_vs_sdpa=1.562e-02  fa2_vs_aiter=3.480e+00
page=64  seq=130  (partial) fa2_vs_sdpa=3.480e+00  aiter_vs_sdpa=1.562e-02  fa2_vs_aiter=3.480e+00
```

The error magnitude is *constant* across every page-size and sequence-length
combination, which is what argues for a structural defect rather than an
accumulation or edge-case bug.

### Decode control (same harness, same cache, same reference)

```
decode backend=None   max_abs_vs_sdpa=7.761e-03
decode backend=fa2    max_abs_vs_sdpa=7.761e-03
```

Decode passing on the `fa2` route with the identical paged cache and reference
function is what establishes that the harness, the NHD layout, and the reference
are all correct — and isolates the defect to the `fa2` prefill kernel.

---

## Ruled out

- **Causal mask ignored.** The most natural explanation, and it is wrong: `fa2`
  matches neither the causal nor the non-causal reference, in either mask mode.
- **Bad test harness / layout / reference.** `aiter` passes on the same inputs,
  same cache, same reference; `fa2` decode also passes on the same cache.
- **Page-size or partial-last-page handling.** Error is identical for page sizes
  1/16/32/64 and for sequence lengths that are and are not exact multiples.
- **A paging bug.** Non-paged `single_prefill_with_kv_cache` is equally affected.
- **Dtype.** fp16 and bf16 both affected.
- **An artifact of one container image or torch build.** Reproduced on three
  independent environments (below).

---

## Environments reproduced on

All on the same MI300X (gfx942), all with `amd-flashinfer` 0.5.3+amd.1 from
`https://pypi.amd.com/rocm-7.2.0/simple`, all producing byte-identical results:

| # | Image | ROCm | torch | Ubuntu |
|---|---|---|---|---|
| 1 | local ROCm + flashinfer image | 7.2.0 | 2.9.1+rocm7.2.0 | 24.04 |
| 2 | local ROCm inference image | 7.2 (hip 7.2.53211) | 2.10.0+git8514f05 | 24.04 |
| 3 | `rocm/vllm-dev:base` (public) | 7.2.3 | 2.11.0+gitd0c8b1f | 22.04 |

Python 3.12 throughout. Environment 3 additionally requires the JIT fix below.

---

## Secondary blocker: amd-flashinfer cannot JIT against torch 2.11

Anyone reproducing on a current vLLM base image will hit this *first*, so it is
included here.

PyTorch 2.11 moved the entire `c10::hip` hipify-v2 backward-compat namespace
behind `#ifdef USE_ROCM` (`torch/include/c10/hip/HIPStream.h`, comment: *"hipify
v2 backward compat in external projects"*). amd-flashinfer's generated kernels
call `c10::hip::getCurrentHIPStream()` but the JIT does not pass `-DUSE_ROCM`, so
**every** JIT compile fails:

```
single_prefill.cu:77:30: error: no member named 'getCurrentHIPStream' in
namespace 'c10::hip'; did you mean 'c10::cuda::getCurrentCUDAStream'?
```

This is not visible on torch 2.9/2.10, where the compat namespace was unguarded.

**Fix:** add `"-DUSE_ROCM"` to `COMMON_HIPCC_FLAGS` in
`flashinfer/compilation_context_hip.py`. That single flag is sufficient; with it,
all kernels build and behave exactly as on torch 2.9/2.10 (including the prefill
defect above, unchanged).

This matters beyond convenience: upstream vLLM main **requires** torch 2.11
(`csrc/libtorch_stable/hip_view.hip` uses `torch::stable::Tensor::layout()`, and
`_C_stable_libtorch` is not an optional target), so without this flag
amd-flashinfer and current vLLM cannot coexist at all.

---

## Third finding: no `fast_decode_plan` on the ROCm path

Not a bug — a missing optimization, but one that costs real throughput, so it
belongs in the same conversation.

`BatchDecodeWithPagedKVCacheWrapper.plan()` runs on the host, outside any
captured CUDA graph, once per decode step. Upstream FlashInfer ships
`flashinfer.decode.fast_decode_plan`, a cudagraph-aware variant that avoids
device-to-device index copies, and the CUDA vLLM backend uses it for exactly
this reason. The ROCm build routes decode through `decode_rocm.py`, which does
not define it:

```
ImportError: cannot import name 'fast_decode_plan' from 'flashinfer.decode_rocm'
```

Measured cost on MI300X (TinyLlama-1.1B, bf16, decode, CUDA graphs enabled,
`ROCM_FLASHINFER` vs vLLM's `ROCM_AITER_FA`, which has no planning stage):

| batch size | AITER | FlashInfer | gap |
|---|---|---|---|
| 1 | 0.724 s | 0.990 s | +36.7% |
| 8 | 0.975 s | 1.388 s | +42.4% |
| 32 | 1.627 s | 1.617 s | −0.6% |

The gap vanishing by batch 32 is the tell: it is a fixed per-step host cost,
hidden once GPU work per step is large enough. Without graphs the two backends
are within noise of each other at every batch size — the overhead only becomes
decisive once graphs remove everything else.

A ROCm `fast_decode_plan` would close this. Full numbers and method:
[`benchmark-results.md`](./benchmark-results.md).

## Impact on the vLLM integration

The `ROCM_FLASHINFER` backend in this branch
(`vllm/v1/attention/backends/rocm_flashinfer.py`) pins the prefill wrapper to
`backend="aiter"` and never uses `auto`. Consequences:

1. Correct prefill on ROCm is AITER underneath, so prefill throughput should be
   near-neutral against the existing `ROCM_AITER_FA` backend by construction.
   The decode path is the genuinely independent comparison.
2. Every AITER-route restriction becomes a hard requirement rather than a
   preference: NHD layout, fp16/bf16, `q_dtype == kv_dtype`,
   `head_dim_qk == head_dim_vo`, `pos_encoding_mode="NONE"`,
   `use_tensor_cores=False`.
3. Things the AITER route accepts and silently ignores — ALiBi slopes, RoPE
   scaling, attention sinks, fp8 dequant scales — must be rejected outright by
   the backend, since ignoring them yields plausible-but-wrong output.
4. The ragged prefill wrapper is unusable: it is wrong even with
   `backend="aiter"`, so only the paged wrapper is wired up.

---

## Suggested next steps

1. File against [AMD-Ecosystem/flashinfer](https://github.com/AMD-Ecosystem/flashinfer):
   findings #1 and #2 together (the broken kernel and the routing that makes it
   unavoidable), then #3 and #4 separately.
2. **Decide which of #1 and #2 is the intended fix.** They admit different
   answers, and it matters:
   - If `fa2` prefill is *meant* to work, it is a kernel bug and the README is
     right.
   - If AITER is the only supported prefill route on gfx942 today, then the
     kernel bug is lower priority but the README's feature matrix is wrong
     (it lists `fa2` prefill as supported ✅), and `auto` should route to
     AITER — or `fa2` prefill should raise rather than return wrong numbers.
3. Confirm whether `BatchPrefillWithRaggedKVCacheWrapper` is meant to support
   AITER at all. Today its `__init__` coerces any backend to `fa2`, so the
   ragged path has no correct configuration.
4. Consider making an unsupported/unimplemented route fail loudly. Every one of
   these findings cost far more to diagnose than it would have if the library
   had raised instead of silently substituting a kernel.
5. Re-run the reproduction above against any new amd-flashinfer build before
   trusting prefill.

## Related

- The scripts used to produce these numbers are in [`scripts/`](./scripts/):
  `probe.py` (full matrix), `probe2.py` (localization), `probe3.py` (the
  causal-hypothesis test), `patch_use_rocm.py` (the JIT fix). The reproduction
  above is self-contained and does not depend on them.
- Integration plan and current status: [`flashinfer-as-vllm-backend.md`](./flashinfer-as-vllm-backend.md)
