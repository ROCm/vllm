# Integrate rocm-flashinfer as a vLLM attention backend (`ROCM_FLASHINFER`)

## Status (updated 2026-08-11)

| Phase | State |
|---|---|
| 0 — env + baseline | Env **done**; flashinfer validated **done** (major finding below); AITER baseline **done** (captured in Phase 4); vendor test suite **not run** |
| 1 — register backend | **GATE PASS** — 30/30 checks (`gate_phase1.py`) |
| 2 — builder + forward | **GATE PASS** — end-to-end coherent text on TinyLlama, eager |
| 3 — correctness | **PASS** — SDPA 18/18, greedy match vs AITER, GSM8K within 0.005 |
| 4 — benchmarks | **DONE (latency, eager)** — on par with AITER, 8–16% faster than Triton *in eager mode only*; the ranking changes once graphs are on, see Phase 6 and [`benchmark-results.md`](./benchmark-results.md). Throughput/serve not run. |
| 5 — Dockerfile | **DONE** — `docker/Dockerfile.rocm_flashinfer`; gates re-verified inside the built image |
| 6 — optimization | **CUDA graphs DONE** (4.6× on decode_bs1). Remaining gap vs AITER at small batch is host-side `plan()`; the CUDA fix `fast_decode_plan` **does not exist in the ROCm build** |
| 7 | Not started |

Working branch: `rocm-flashinfer-backend`, based on upstream `bebf918044`
(v0.26.1rc0). Build with `docker/Dockerfile.rocm_flashinfer`.
Gate scripts: `plan/scripts/gate_phase{1,2,3}.py` and `gate_phase3_gsm8k.py`.

### Correctness: settled

**Layer 1 — vs SDPA reference: 18/18 PASS.** vLLM's own
`_test_backend_correctness` harness, `atol=rtol=1e-2`, across
single/small/medium/large decode, prefill, and mixed batch specs, with
`TRITON_ATTN` run alongside as a control. Driven via `gate_phase3.py` because
the packaged test parametrizes over gated `meta-llama/Meta-Llama-3-8B`;
`large_prefill` is skipped since it needs `max_model_len=4096` and TinyLlama
caps at 2048. Re-verified inside the built image on torch 2.12.

**Layer 2 — vs AITER end-to-end:** 2/3 prompts byte-identical greedy. The
third was **investigated and cleared**: at the diverging step the top-2
candidates are an *exact bf16 tie* under `ROCM_FLASHINFER`
(`' C'` and `' The'`, both -2.095223, gap 0.00e+00) and 6.25e-02 apart under
`ROCM_AITER_FA`. That is argmax sensitivity between kernels with different
accumulation orders, not incorrectness. Earlier read — that AITER and
TRITON agreeing meant ours was wrong — did not survive the measurement.

**Layer 3 — GSM8K: PASS.** Mistral-7B-Instruct-v0.3, 400 questions, 5-shot,
greedy, bf16, via `gate_phase3_gsm8k.py` (which drives the same
`evaluate_gsm8k` the packaged test uses, against a server per backend —
the packaged test parametrizes over quantized models and exposes no
attention-backend knob).

| backend | accuracy | invalid |
|---|---|---|
| `ROCM_FLASHINFER` | 0.4650 | 0.0075 |
| `ROCM_AITER_FA` | 0.4700 | 0.0050 |

|delta| = **0.0050**, against the plan's gate of ≤ 0.08. Both land where
Mistral-7B is expected to on 5-shot GSM8K, so this is two working backends
agreeing, not two broken ones agreeing.

Caveat on precision: 400 questions puts the standard error on each accuracy
near 2.5%, so this bounds the difference rather than resolving it — it rules
out a meaningful quality regression, not a sub-1% one.

### Environment: a three-way version squeeze (resolved)

- `amd-flashinfer` 0.5.3+amd.1 emits `c10::hip::getCurrentHIPStream()`.
  **torch 2.11 moved that whole compat namespace behind `#ifdef USE_ROCM`**
  (`c10/hip/HIPStream.h`, "hipify v2 backward compat"), so every JIT compile
  fails on torch 2.11 unless `-DUSE_ROCM` is passed.
- Upstream vLLM main **requires** torch ≥2.10 (`compressed-tensors 0.17.0`)
  and in practice torch 2.11 (`torch::stable::Tensor::layout()` in
  `csrc/libtorch_stable/hip_view.hip`; `_C_stable_libtorch` is not optional).
- Resolution: base on **`rocm/vllm-dev:base`** (ROCm 7.2.3, Ubuntu 22.04,
  py3.12, torch 2.11.0+gitd0c8b1f — matches `Dockerfile.rocm_base`) and apply
  `plan/scripts/patch_use_rocm.py`, which appends
  `-DUSE_ROCM` to `COMMON_HIPCC_FLAGS` in flashinfer's
  `compilation_context_hip.py`. **Report this to AMD.**
- **All of this now lives in `docker/Dockerfile.rocm_flashinfer`.** The earlier
  `vllm-fi:*` `docker commit` snapshots were pruned off the host, which is
  what forced Phase 5. The base image has since moved to **torch 2.12.0** and
  every gate still passes, so the backend is not pinned to one torch build.
- Build gotchas the Dockerfile handles: `requirements/rocm.txt` replaces the
  ROCm torch with a CUDA wheel unless the torch stack is pinned via `-c`;
  setuptools-scm needs `git safe.directory`; and `/.deps` must be kept out of
  the build context, since CMakeCache.txt hard-codes absolute paths.

### Two vLLM defects found and fixed on this branch

1. **Installing amd-flashinfer broke vLLM on ROCm entirely** — including
   `ROCM_AITER_FA`. `allreduce_rms_fusion.py` gated on a raw
   `find_spec("flashinfer")` and imported CUDA-only `flashinfer.comm`, which
   asserts `libcudart is not loaded` at import. Fixed with a
   `current_platform.is_cuda()` guard and a broader `except`. This is
   independently upstreamable.
2. **GQA group size** — the HIP decode kernel's `DISPATCH_GQA_GROUP_SIZE`
   only handles `{1, 2, 3, 4, 8}`; anything else aborts inside `plan()` with
   "Unsupported group_size: N". Qwen2.5-0.5B (14/2 = 7) hit this. Now a clean
   `supports_combination()` rejection.

### Corrections to the original plan

- KV writes go through `do_kv_cache_update()`, not `forward()`.
- `envs.py` needs **two** edits, not three (`compile_factors()` hashes every
  known env var unless explicitly ignored).
- **`VLLM_ATTENTION_BACKEND` no longer exists** — selection is
  `--attention-backend` / `AttentionConfig.backend` / the `attention_backend`
  EngineArgs field. Phase 4's benchmark sweep must use that, not the env var.
- KV cache shape is the packed `(B, H, N, 2*hs)` with an NHD stride order,
  not the old 5-D `(B, 2, N, H, hs)`.

---

## Context

**Goal.** Register AMD's ROCm port of FlashInfer ([AMD-Ecosystem/flashinfer](https://github.com/AMD-Ecosystem/flashinfer), pip package `amd-flashinfer`, module `flashinfer`) as a first-class vLLM V1 attention backend on AMD GPUs, prove its numerics match the incumbent `ROCM_AITER_FA` backend, and benchmark it against that backend.

**Why.** vLLM on ROCm today selects among `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `TRITON_ATTN`. FlashInfer's wrapper API (plan/run, paged KV, CUDA-graph-friendly persistent buffers) is the interface upstream vLLM is increasingly built around on NVIDIA, and the AMD port ships HIP FA2 kernels plus AITER dispatch. A working `ROCM_FLASHINFER` backend gives ROCm users the same backend surface as CUDA and a second attention path.

**What already exists (do not rebuild).**
- `vllm/v1/attention/backends/flashinfer.py` — the CUDA backend, 2405 lines. Structural template only: TRT-LLM, cuDNN, FP4, DCP, batch-invariance and `fast_decode_plan` have no HIP equivalent.
- `vllm/v1/attention/backends/rocm_aiter_fa.py` — ROCm reference for the class-attribute / `supports_*` contract.
- `vllm/v1/attention/backends/registry.py` — `AttentionBackendEnum`; one member per backend mapping to a class path.
- `vllm/platforms/rocm.py` — `_get_backend_priorities()` + `get_valid_backends()`; selection is priority-list + `backend_class.validate_configuration(...)`.
- `vllm/v1/attention/backend.py` — the `AttentionBackend` ABC (note: `backend.py`, singular, not `attention/backends/abstract.py`).

---

## Constraints (these drive the whole plan)

### The prefill route is not optional

Measured 2026-07-31 on MI300X (gfx942), flashinfer 0.5.3+amd.1, reproduced on
three independent images. Probes: `plan/scripts/`.
Full write-up: [`handoff-amd-flashinfer-fa2-prefill.md`](./handoff-amd-flashinfer-fa2-prefill.md).

- **HIP `fa2` prefill is numerically wrong.** max_abs ≈ 3.5 vs a torch SDPA
  reference. Reproduces on paged, ragged and `single_prefill_with_kv_cache`;
  fp16 and bf16; page sizes 1/16/32/64; exact and partial last page.
- **`backend="aiter"` prefill is correct** (~1.5e-2 bf16 causal, 3.9e-3
  non-causal) on identical inputs against the same reference — which is what
  validates the harness.
- **`backend="auto"` resolves to fa2 on ROCm** (logged: "auto backend not
  supported on ROCm. Selecting FA2 as the backend"), so the *default* route is
  the broken one.
- Ruled out: causal-mask-ignored (fa2 matches neither the causal nor the
  non-causal reference), page-size effects, partial-page handling.
- **Decode is correct on every route**, both dtypes.
- `BatchPrefillWithRaggedKVCacheWrapper` is wrong even with `backend="aiter"`.

→ Pin `backend="aiter"` on the prefill wrapper. Use the paged wrapper, never
the ragged one. **Report this upstream to AMD.**

### Absent from the HIP build (vs the README)
`MultiLevelCascadeAttentionWrapper`, `BatchMLAPagedAttentionWrapper`,
`flashinfer.aiter_utils.is_aiter_supported` (only `HAS_AITER` and
`get_aiter_mha_module` exist), and all `trtllm_*` entry points.

### AITER routing requirements (hard, because prefill must use AITER)
NHD layout, fp16/bf16, `q_dtype == kv_dtype`, `head_dim_qk == head_dim_vo`,
`pos_encoding_mode="NONE"`, `use_tensor_cores=False`.
**Silently ignored** on that route: ALiBi slopes, RoPE scaling, attention
sinks, fp8 dequant scales → must be hard rejections, not warnings.

### Versions

| | amd-flashinfer 0.5.3+amd.1 | first bring-up image | upstream `Dockerfile.rocm_base` |
|---|---|---|---|
| GPU | gfx942, gfx950 | gfx942 | gfx90a…gfx1201 |
| ROCm | 7.0.2 / 7.1.1 / 7.2 | 7.2.0 | 7.2.3 |
| Ubuntu | 24.04 validated | 24.04 | 22.04 |
| Torch | 2.8.0 / 2.9.1 | 2.9.1+rocm7.2 | from `ROCm/pytorch@d0c8b1f3` |
| Python | 3.12 | 3.12 | 3.12 |

That image already satisfies every amd-flashinfer requirement, so first
bring-up used it directly. It was later abandoned for `rocm/vllm-dev:base`,
because upstream vLLM main requires torch 2.11 — see the Status section.

**Scope:** attention only — paged batch prefill + batch decode, fp16/bf16, NHD.
No MLA, no fp8 KV, no fused RMSNorm/RoPE/sampling.
**Target hardware:** MI300X (gfx942), 1 GPU on this host.

---

## Phase 0 — Environment and baseline

1. ~~Rebase onto upstream main.~~ **Done** — branch `rocm-flashinfer`.
2. ~~Validate flashinfer standalone before involving vLLM.~~ **Done** — see the
   constraints section. Re-run `plan/scripts/probe.py` against
   any new amd-flashinfer build before trusting prefill.
3. ~~Unblock the vLLM build.~~ **Done** — builds in `vllm-fi:dev`.
4. **Capture the AITER baseline** on the same image so later comparisons are
   apples-to-apples:
   ```bash
   VLLM_ATTENTION_BACKEND=ROCM_AITER_FA vllm bench latency    --model ... 
   VLLM_ATTENTION_BACKEND=ROCM_AITER_FA vllm bench throughput --model ...
   ```
   Save raw JSON to `bench/baseline-aiter/`.

---

## Phase 1 — Register the backend — code written

Files touched:

- **`vllm/v1/attention/backends/rocm_flashinfer.py`** (new) — `RocmFlashInferBackend`:
  - `supported_dtypes` fp16/bf16; `supported_kv_cache_dtypes` auto/float16/bfloat16
  - `get_name()` → `"ROCM_FLASHINFER"` (must match the enum member exactly)
  - `get_supported_head_sizes()` → `[64, 128, 256]`; `get_supported_kernel_block_sizes()` → `[16, 32, 64]`
  - `get_kv_cache_shape()` → `(num_blocks, num_kv_heads, block_size, 2 * head_size)` — the **packed** layout current upstream uses, *not* the old 5-D `(B, 2, N, H, hs)`
  - `get_kv_cache_stride_order()` → NHD only: `(0, 2, 1, 3)`, or `(1, 0, 3, 2, 4)` with the layers dim
  - `get_required_kv_cache_layout()` → `"NHD"` — the clean hook for pinning layout; do **not** call `set_kv_cache_layout()`
  - `supports_compute_capability()` → `get_cdna_version() >= 3` (gfx942/gfx950)
  - `supports_sink()`, `supports_sliding_window()`, `supports_non_causal()` → `False`
  - `forward_includes_kv_cache_update = False`
- **`registry.py`** — added `ROCM_FLASHINFER` enum member.
- **`platforms/rocm.py`** — `_get_backend_priorities()` prepends `ROCM_FLASHINFER` when `envs.VLLM_ROCM_USE_FLASHINFER` is set; default order unchanged for existing users.
- **`envs.py`** — `VLLM_ROCM_USE_FLASHINFER`, default `False`, in **two** places:
  the type-hint block and the `environment_variables` dict.
  *(Correction to the original plan: `compile_factors()` now hashes every known
  env var unless explicitly listed in `ignored_factors`, so there is no third
  edit and no stale-compile-cache hazard.)*

**Gate:** `--attention-backend ROCM_FLASHINFER` selects on gfx942 and raises a
clear `ValueError` naming the reason on gfx90a/gfx1100.

---

## Phase 2 — Metadata builder and forward — code written

**Correction to the original plan:** KV cache writes no longer happen inside
`forward()`. Backends set `forward_includes_kv_cache_update = False` and
implement `do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)`,
which `torch.ops.vllm.unified_kv_cache_update` calls before attention. The op
still ends in `reshape_and_cache_flash`, but on `(B,H,N,2*hs) → transpose(1,2)
→ split(head_size, -1)` k/v views.

- **Builder** — persistent `CpuGpuBuffer` indptr / indices / last_page_len;
  a `_copy_page_indices_kernel` Triton kernel (copied locally rather than
  imported, since `backends/flashinfer.py` hard-imports CUDA-only symbols and
  cannot be imported on ROCm); `split_decodes_and_prefills` with decodes first;
  `plan()` per half of the batch, re-basing `qo_indptr`/`indptr`/`indices` onto
  the prefill sub-batch.
- **Wrappers** — `BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="aiter")`
  and `BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=False,
  backend="auto")`. 256 MB workspace (CUDA-tuned constant, unverified here).
- **CUDA graphs deferred** — `_cudagraph_support = NEVER`, run `--enforce-eager`.
  Raise to `UNIFORM_SINGLE_TOKEN_DECODE` only after Phase 3 is green. Do not
  debug graphs and numerics in the same session.
- **Impl** — rejects ALiBi, sinks and sliding window in `__init__`; rejects
  non-`DECODER` attn types; `use_cascade_attention()` → `False`.

---

## Phase 3 — Correctness vs AITER

Three layers, cheapest first; do not skip to layer 3.

1. **Backend vs SDPA.** Add `ROCM_FLASHINFER` to the backend map in
   `tests/v1/attention/utils.py` and to `BACKENDS_TO_TEST` in
   `tests/v1/attention/test_attention_backends.py` (guarded by
   `current_platform.is_rocm()`). That harness already asserts
   `allclose(rtol=1e-2, atol=5e-3)` across decode/prefill/mixed `BATCH_SPECS`.
2. **E2E equivalence vs AITER.** Follow `tests/v1/e2e/test_cascade_attention.py`'s
   `m.setenv("VLLM_ATTENTION_BACKEND", backend)` parametrization; compare greedy
   token IDs and top-k logprobs against `ROCM_AITER_FA`. This is the check that
   answers "are the results aligned with the aiter backend".
3. **Accuracy eval.** `tests/evals/gsm8k/test_gsm8k_correctness.py`, `RTOL=0.08`,
   once per backend; confirm both land within noise of each other.

**Debug ordering when numerics disagree:** confirm which route flashinfer took
(`FLASHINFER_LOGGING_LEVEL=DEBUG`; force `backend="fa2"` vs `"aiter"`) *before*
suspecting vLLM. A route regression and a metadata bug look identical from the
output tensor — and one of the two routes is known-broken.

---

## Phase 4 — Benchmarking vs AITER

Sweep `{ROCM_AITER_FA, ROCM_FLASHINFER, TRITON_ATTN}` via `VLLM_ATTENTION_BACKEND`
(no per-benchmark flag exists) across `vllm bench latency` / `throughput` /
`serve`, ISL/OSL `{128, 2048, 8192} × {128, 1024}`, batch `{1, 8, 32}`, TP 1.

**Read the results with this in mind:** because correct prefill is forced onto
the AITER route, `ROCM_FLASHINFER` prefill *is* AITER plus wrapper overhead —
expect near-neutral-to-slightly-worse, and treat a large prefill gap as a bug in
the plan/metadata path rather than a kernel result. **Decode is the genuinely
independent comparison** and is where a real win or loss would show up.

Hold constant: model, dtype, `--max-model-len`, `--gpu-memory-utilization`,
eager/graph state, block size. Record `rocm-smi` clocks and power per run —
MI300X numbers are not comparable across thermal states.

---

## Phase 5 — Docker image

`docker/Dockerfile.rocm_flashinfer` layered on the vLLM ROCm base:

```
ARG BASE_IMAGE=rocm/vllm-dev:base        # ROCm 7.2.3, Ubuntu 22.04, py3.12
ARG FLASHINFER_REPO=https://github.com/AMD-Ecosystem/flashinfer.git
ARG FLASHINFER_BRANCH=amd-integration    # pin to the 0.5.3+amd.1 tag once validated
```

Build amd-flashinfer from source in its own stage (`pip wheel . --wheel-dir=
/app/install --no-deps --no-build-isolation`), install the wheel in the final
stage, append the commit to `/app/versions.txt` — mirroring how
`Dockerfile.rocm_base` stages hipblaslt/triton/aiter. Building from source
sidesteps the Ubuntu 22.04-vs-24.04 glibc risk of the prebuilt wheel.

Leave `VLLM_ROCM_USE_FLASHINFER` unset (opt-in). Consider
`FLASHINFER_DISABLE_JIT=1` for reproducible startup once AOT covers the needed
kernels — first-request JIT otherwise shows up as a spurious TTFT regression in
Phase 4.

---

## Phase 6 — Optimization (only after Phase 3 is green and Phase 4 has numbers)

1. **CUDA graphs** — `UNIFORM_SINGLE_TOKEN_DECODE` + per-batch-size decode
   wrapper cache with persistent buffers (`use_cuda_graph=True` is supported).
   Largest win for small-batch decode.
2. **Plan overhead** — `.plan()` runs per forward; the CUDA backend needed
   `fast_decode_plan` precisely because plan cost dominated at small batch.
3. **Workspace sizing** — 256 MB is a CUDA-tuned guess; measure.
4. **Block size** — verify 16 is optimal on gfx942 for decode.
5. **Decode route** — compare `auto` vs pinned `fa2` vs `aiter` per shape.

## Phase 7 — Upstreaming

Reviewable PRs: (1) enum + env var + platform wiring, (2) backend impl,
(3) tests, (4) Dockerfile + docs. Keep `VLLM_ROCM_USE_FLASHINFER` defaulting to
`False` throughout. Separately, file the fa2 prefill bug against
AMD-Ecosystem/flashinfer with the probe scripts attached.

---

## Verification

| Phase | Gate |
|---|---|
| 0 | flashinfer probes pass ✅; vLLM builds in the image ⬜; AITER baseline saved ⬜ |
| 1 | `--attention-backend ROCM_FLASHINFER` selects on gfx942; clear `ValueError` on unsupported arch |
| 2 | `vllm serve` + one completion returns coherent text with `--enforce-eager` |
| 3 | `test_backend_correctness` green; greedy tokens match `ROCM_AITER_FA`; GSM8K within `RTOL=0.08` |
| 4 | Latency/throughput/serve tables vs AITER and Triton, clocks recorded |
| 5 | Image builds from committed Dockerfile; `import flashinfer, vllm` works; Phase 3 green inside it |
| 6 | Phase 3 still green after optimization |

**Smoke command for the end-to-end claim:**
```bash
VLLM_ROCM_USE_FLASHINFER=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --attention-backend ROCM_FLASHINFER --enforce-eager --max-model-len 8192
# then the identical run with --attention-backend ROCM_AITER_FA and diff outputs
```

## Risks

- **fa2 prefill is broken** → mitigated by pinning `backend="aiter"`, but it
  means a future amd-flashinfer bump could silently change routing. The probe
  scripts are the regression guard; run them on every version bump.
- **Value proposition is narrower than assumed** — prefill is AITER underneath,
  so the case for this backend rests on decode, API parity with CUDA, and
  AMD's investment direction rather than on prefill throughput.
- **Silently-ignored kwargs** (ALiBi, sinks, rope scaling) → hard rejections in
  `validate_configuration` / `Impl.__init__`.
- **Upstream churn** — the attention registry was refactored recently and
  `_get_backend_priorities` is actively changing; expect to rebase.
- **Single GPU on this host** — TP>1 paths cannot be exercised locally.
