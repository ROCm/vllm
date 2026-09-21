# HIP decode-attention kernel for gfx1151 (Strix Halo)

Standalone HIP kernel for the decode / speculative-decode regime
(`num_seqs=1`, `M ∈ {1..5}`), plus the harness and measurement reports that
produced it.

**This is exploratory work, not an integrated backend.** The kernels build to a
self-contained binary with `main()`; there is no Torch op, no bindings and no
`AttentionBackend`. Everything here is reproducible on a gfx1151 board.

## Result

Radeon 8060S (gfx1151), `Hq=32, Hkv=16, D=256, M=4`, fp16, KV working set forced
past the 32 MiB MALL. Roofline = KV bytes / 230 GiB/s.

| S | Triton NHD | Triton HND | HIP NHD | **HIP HND** |
|---|---|---|---|---|
| 128 | 23.80 (35.7 %) | 22.98 (36.9 %) | **14.10 (60.2 %)** | 14.44 (58.8 %) |
| 1024 | 111.02 (61.2 %) | 103.89 (65.4 %) | 85.21 (79.7 %) | **80.26 (84.6 %)** |
| 2048 | 199.15 (68.2 %) | 185.71 (73.2 %) | 162.61 (83.6 %) | **153.16 (88.7 %)** |
| 4096 | 376.52 (72.2 %) | 350.97 (77.4 %) | 316.60 (85.8 %) | **295.75 (91.9 %)** |
| 8192 | 737.07 (73.7 %) | 689.69 (78.8 %) | 618.70 (87.8 %) | **585.46 (92.8 %)** |
| 16384 | 1441.03 (75.4 %) | 1353.00 (80.3 %) | 1220.56 (89.1 %) | **1157.49 (93.9 %)** |
| 32768 | 2838.81 (76.6 %) | 2644.74 (82.2 %) | 2421.45 (89.8 %) | **2308.08 (94.2 %)** |

µs (% of roofline). The kernel wins at every context length; the margin is
largest at S=128 (**1.69×**) because its fixed overhead is low, and settles to
~1.17× at 32k where both are bandwidth-bound.

`D=128` (`Hq=32, Hkv=8`): **40.70 µs vs 54.95 µs** for Triton = 1.35×
(83.5 % vs 61.8 % of roofline).

## What the kernel does

Online softmax with split-KV. Each of `NSEG` segments walks its slice of the
sequence keeping `(m, l, acc)`; a reduction merges them with
`alpha = exp(m_seg - m_global)`.

Design decisions that mattered, all measured:

| Decision | Value | Why |
|---|---|---|
| Lane ownership | `acc[M][8]` — one K/V row = one `global_load_b128` per lane | **2.67×**, the bulk of the gain |
| `NSEG` | **1** (2 at D=128) | 16 costs 700 µs; the penalty grows with S |
| Pipeline depth | 4 loads in flight | 6 and 8 are worse |
| K/V caching | **normal, not non-temporal** | non-temporal is **1.84× worse** |
| LDS | **none**, not even for Q | 0 barriers in the inner loop |
| `P@V` | dot2/VOPD over WMMA | 1.57× on useful work, 83 vs 153 VGPR |

129 VGPR, zero spills, zero `s_barrier`, zero LDS.

The counter-intuitive ones share a cause: with GQA=2 two q-head blocks read the
same KV head, so 64 MiB is issued for 32 MiB unique and the cache recovers half
the traffic. Anything that breaks that reuse — more segments, non-temporal
hints, larger blocks — costs more than it gives. **In this regime the scarce
resource is cache locality, not grid parallelism.**

## Layout

```
kernels/     decode_attn*.hip   kernel versions, v1 -> v6
             *.s                dumped ISA for the winning variants
harness/     check.py           validator (max_abs AND max_rel)
             ref_attn.py        PyTorch fp32 reference
             *.sh               build / sweep / ASM-stat scripts
reports/     00-design.md       design document with measured verdicts
             00-protocol.md     measurement protocol
             f0..f9-*.md        one report per investigation
```

Kernel versions, in order:

| File | What it adds |
|---|---|
| `decode_attn.hip` | first correct version (445 µs, 30.5 % of roofline) |
| `decode_attn_v2..v4.hip` | intermediate steps |
| `decode_attn_v5.hip` | lane-ownership redesign — the one that works (dense layout) |
| `decode_attn_paged.hip` | v5 re-hosted on vLLM's paged KV layout |
| `decode_attn_paged_mutants.hip` | deliberately broken variants for negative controls |
| `decode_attn_v6.hip` | generalises to `D ∈ {64, 128, 256}` |

## Reproducing

```bash
VENV=/path/to/rocm/venv
$VENV/bin/hipcc -O3 --offload-arch=gfx1151 \
    kernels/decode_attn_paged.hip -o /tmp/attn \
    -DNSEG=1 -DKPW=4 -DLAYOUT=0
/tmp/attn
```

Validate against the PyTorch reference:

```bash
$VENV/bin/python harness/check.py
```

**Measurement hygiene matters more than usual here** — see
`reports/00-protocol.md`:

- Force the KV working set past the 32 MiB MALL. Without it, S=128 reads
  **33.8 % fast** and the kernel appears to exceed 100 % of roofline.
- Time with the realtime clock, not `SHADER_CYCLES` (it stops during L2+ stalls
  and wraps at 2²⁰).
- Sustained clock under load is ~2100 MHz, not the 2900 MHz boost.

## Correctness

`max_abs` alone is **not** a sufficient criterion, and this bit us:

| mutant | S | max_abs | max_rel | verdict at `max_abs <= 2e-2` |
|---|---|---|---|---|
| causal mask off by one key | 2048 | 1.2e-03 | 1.065 | **PASSES** ⚠ |
| causal mask off by one key | 48 | 5.2e-02 | 4.0e+01 | fails |

A one-key error moves the softmax by ~1/S while the tolerance is fixed, so
**longer contexts hide more bugs**. Validate with `max_rel <= 1e-3`, a short-`S`
case, and negative controls — `decode_attn_paged_mutants.hip` carries three
(causal mask, slot↔head index swap, K/V packing offset).

Caveat found late: `MUTATE=2` only patches the `LAYOUT==0` branch, so under
`-DLAYOUT=1` it is a no-op and passes silently. Run it with `-DLAYOUT=0`.

## Limitations

- **Not integrated.** Standalone binary; no Torch op, no backend, not reachable
  from `vllm serve`.
- `decode_attn_v6.hip` (the `D ∈ {64,128,256}` generalisation) is on the **dense**
  layout; it has not been ported to the paged one.
- fp16/bf16 only — `kv_cache_dtype=fp8` untested.
- Tuned for `H_kv >= 8`. Below that the layout and segment choices stop paying.
- `M` is a compile-time constant, as are the head counts and `D`.

## Notes on the reports

`reports/` is the raw investigation, one file per agent, including the dead ends
and the corrections. Several design-document predictions were **refuted by
measurement** — non-temporal loads, high `NSEG`, Q-in-LDS, large tiles,
`global_load_lds` (which does not exist on gfx1151) — and the reports keep the
evidence rather than only the conclusions. `reports/00-design.md` §0.1 has the
verdict table.

AI assistance was used throughout: the kernel, the harness and the sweeps were
produced with Claude. Every number here was measured on the target board.
