# Phase 4 — `ROCM_FLASHINFER` vs `ROCM_AITER_FA` vs `TRITON_ATTN`

Measured 2026-07-31, MI300X (gfx942), one GPU, TP=1.
Model: TinyLlama-1.1B-Chat-v1.0, bf16, `kv_cache_dtype=auto`, block size 16.
`vllm bench latency`, 2 warmup + 5 measured iterations, **`--enforce-eager`**.
Driver: [`scripts/bench_backends.sh`](./scripts/bench_backends.sh).

## Headline

**`ROCM_FLASHINFER` is on par with `ROCM_AITER_FA`** (every delta inside
measurement noise except one at 4.1%) **and consistently 8–16% faster than
`TRITON_ATTN`.**

That is the expected shape: the backend pins prefill to the AITER route
(because the HIP `fa2` prefill kernels are numerically wrong — see
[`handoff-amd-flashinfer-fa2-prefill.md`](./handoff-amd-flashinfer-fa2-prefill.md)),
so prefill is AITER underneath and cannot diverge much. Decode is the
genuinely independent path, and it also lands neutral.

## Results — average latency, seconds, lower is better

| config | isl/osl/bs | ROCM_AITER_FA | ROCM_FLASHINFER | TRITON_ATTN | FI vs AITER | FI vs Triton |
|---|---|---|---|---|---|---|
| decode_bs1 | 128/512/1 | 4.660 | **4.601** | 5.407 | −1.3% | −14.9% |
| decode_bs32 | 128/512/32 | **7.185** | 7.214 | 8.026 | +0.4% | −10.1% |
| prefill_bs1 | 1024/32/1 | 0.2874 | **0.2870** | 0.3433 | −0.2% | −16.4% |
| prefill_bs8 | 1024/32/8 | **0.4616** | 0.4806 | 0.5206 | +4.1% | −7.7% |
| mixed_bs8 | 512/128/8 | 1.7434 | **1.7188** | 1.9366 | −1.4% | −11.2% |

Negative = FlashInfer faster.

## Noise floor

`decode_bs1` was repeated at the end of the sweep as a drift check:

| backend | first | repeat | drift |
|---|---|---|---|
| ROCM_AITER_FA | 4.660 | 4.714 | +1.2% |
| ROCM_FLASHINFER | 4.601 | 4.624 | +0.5% |
| TRITON_ATTN | 5.407 | 5.541 | +2.5% |

**Treat anything under ~3% as a tie.** By that standard the only real
AITER-vs-FlashInfer difference in the table is `prefill_bs8` (+4.1%), and every
FlashInfer-vs-Triton difference is real.

## Read these numbers with the caveats

1. **`--enforce-eager` throughout.** `ROCM_FLASHINFER` currently declares
   `AttentionCGSupport.NEVER`, so eager is the only apples-to-apples
   comparison. It is *not* the default production configuration:
   `ROCM_AITER_FA` supports CUDA graphs (`UNIFORM_BATCH`) and would pull ahead
   in a default run, most visibly at small batch. **Enabling CUDA graphs
   (Phase 6 item 1) is the single highest-value remaining work.**
2. **TinyLlama-1.1B is small.** Attention is a smaller fraction of total step
   time than in a 7B+ model, which compresses the differences between
   backends. A larger model would separate them more.
3. TP=1, single GPU, bf16, no fp8 KV, no sliding window — all outside the
   backend's currently supported surface.
4. Latency only. No `vllm bench throughput` or `serve` (TTFT/ITL/P99) yet.

## Why an earlier sweep was discarded

A first pass produced `ROCM_FLASHINFER` at 9.81s vs `ROCM_AITER_FA` at 4.57s on
`decode_bs1` — a 2.1× gap that looked like real per-step `plan()` overhead. It
was not. Another tenant on this shared box was acquiring and releasing ~180 GiB
of VRAM during the sweep; the `vram_pct` column swung 0→88% between runs, and
one run aborted with vLLM's "other processes sharing the same container release
GPU memory while vLLM is profiling" assertion. Re-running on an idle GPU
inverted the result to a 1.3% FlashInfer *win*.

This is why `bench_backends.sh` samples GPU use and VRAM around every run and
writes them next to the timings: on a shared machine a benchmark without that
context cannot be distinguished from noise. Discard any row whose
`gpu_use`/`vram_pct` are non-zero and re-run.
