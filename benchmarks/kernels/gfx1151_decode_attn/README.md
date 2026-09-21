# HIP decode-attention kernel for gfx1151 (Strix Halo) — investigation record

The kernel this directory produced is now **integrated**, and lives elsewhere:

| what | where |
| --- | --- |
| kernel source | `csrc/rocm/rdna35_decode_attn.cu` |
| JIT loader | `vllm/v1/attention/ops/rdna35_hip_decode.py` |
| backend | `vllm/v1/attention/backends/rdna35_hip_attn.py` (`RDNA35_HIP_ATTN`) |
| tests | `tests/kernels/attention/test_rdna35_hip_decode.py` |

What remains here is the measurement record, plus one piece of unported work.

## Measuring it

Through vLLM's own attention benchmark, the same path TRITON_ATTN takes — same
KV tensors, same metadata, same `do_bench` under CUDA graphs:

```bash
cd benchmarks/attention_benchmarks
VLLM_KV_CACHE_LAYOUT=HND python benchmark.py \
    --backends TRITON_ATTN RDNA35_HIP_ATTN \
    --batch-specs q4s1k --dtype float16 \
    --num-q-heads 32 --num-kv-heads 16 --head-dim 256 \
    --block-size 16 --min-working-set-mb 96
```

`--min-working-set-mb 96` is not optional: without it a short context stays
resident in the 32 MiB MALL, reads ~34 % fast, and the kernel appears to exceed
100 % of roofline.

If a shape does not fit the kernel it **falls back to Triton and says so** as a
warning. Take that warning seriously: it means the row labelled
`RDNA35_HIP_ATTN` is reporting Triton's time.

## ⚠ The numbers in `reports/` are superseded

Two measurement errors were found after those reports were written, both of
which flattered the HIP kernel:

1. **The timed region excluded `reduce_segments`**, which normalises by `l` and
   writes the output — without it there is no result. Worth +20.8 % at S=128,
   +4.6 % at S=32768, so the bias is largest exactly where the reported margin
   was largest.
2. **Everything labelled fp16 was actually bf16.** The benchmark built its model
   config with `dtype="auto"` and ignored `BenchmarkConfig.dtype`, so it used
   the stand-in model's native bfloat16.

The reports are kept as the record of how the kernel was arrived at — the
design reasoning and the refuted hypotheses remain valid. **The timing tables do
not.** They are pending regeneration through the unified harness.

## What the kernel does

Online softmax with split-KV. Each of `NSEG` segments walks its slice of the
sequence keeping `(m, l, acc)`; a reduction merges them with
`alpha = exp(m_seg - m_global)`.

Design decisions that mattered, all measured:

| Decision | Value | Why |
| --- | --- | --- |
| Lane ownership | `acc[M][8]` — one K/V row = one `global_load_b128` per lane | **2.67×**, the bulk of the gain |
| `NSEG` | **1** | 16 costs 700 µs; the penalty grows with S |
| Pipeline depth | 4 loads in flight | 6 and 8 are worse |
| K/V caching | **normal, not non-temporal** | non-temporal is **1.84× worse** |
| LDS | **none**, not even for Q | 0 barriers in the inner loop |
| `P@V` | dot2/VOPD over WMMA | 1.57× on useful work, 83 vs 153 VGPR |

The counter-intuitive ones share a cause: with GQA=2 two q-head blocks read the
same KV head, so 64 MiB is issued for 32 MiB unique and the cache recovers half
the traffic. Anything that breaks that reuse — more segments, non-temporal
hints, larger blocks — costs more than it gives. **In this regime the scarce
resource is cache locality, not grid parallelism.**

## Correctness

`max_abs` alone is **not** sufficient, and this bit us:

| mutant | S | max_abs | max_rel | verdict at `max_abs <= 2e-2` |
| --- | --- | --- | --- | --- |
| causal mask off by one key | 2048 | 1.2e-03 | 1.065 | **PASSES** ⚠ |
| causal mask off by one key | 48 | 5.2e-02 | 4.0e+01 | fails |

A one-key error moves the softmax by ~1/S while the tolerance is fixed, so
longer contexts hide more bugs. The test suite therefore checks `max_rel`, a
short-`S` case, and a negative control built with `-DMUTATE=1` that it asserts
must fail.

## Limitations

- **fp16 only.** The inner product is `__builtin_amdgcn_fdot2` and the loads
  reinterpret the cache as `_Float16`, so bf16 would return finite nonsense; the
  op refuses it and the backend falls back.
- **`D=256` only**, and one sequence per call.
- `kernels/decode_attn_v6.hip` generalises to `D ∈ {64, 128, 256}` but sits on
  the **dense** layout and has not been ported to the paged one. It is kept here
  because that port has not been done — it is not redundant with the integrated
  kernel.
- `kv_cache_dtype=fp8` untested.
- Tuned for `H_kv >= 8`. Below that the layout and segment choices stop paying.
- `M`, the head counts and `D` are compile-time constants, so one build serves
  one shape.
- The op is compiled JIT, not by CMake, so it is a development path rather than
  something a wheel ships.

## Notes on the reports

`reports/` is the raw investigation, including the dead ends and corrections.
Several design-document predictions were **refuted by measurement** —
non-temporal loads, high `NSEG`, Q-in-LDS, large tiles, `global_load_lds` (which
does not exist on gfx1151). `reports/00-design.md` §0.1 has the verdict table;
`reports/00-protocol.md` has the measurement rules, including the requirement
that a single configuration win across the whole context sweep.

AI assistance was used throughout: the kernel, the harness and the sweeps were
produced with Claude. Every number here was measured on the target board.
