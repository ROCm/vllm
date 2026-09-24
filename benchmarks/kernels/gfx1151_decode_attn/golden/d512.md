# Golden — D=512

Best measured result for the D=512 configurations, at both query-token counts.
Replace only when a run beats this one, and say in the commit message what
changed to earn it.

| | |
| --- | --- |
| date | 2026-09-24 |
| commit | `6ee57eb65e` + OPTIMIZATIONS.md 001 (BFLY split), 002 (assume), 003 (LDS pad), 005 (GRIDT), 007 (DPL), 008 (BFLY retune) |
| kernel | `csrc/rocm/rdna35_decode_attn.cu` |
| knobs | NSEG/MSPLIT from the heuristics; BFLY, GRIDT and DPL/LDSPLIT per configuration from `_TUNED` |
| roofline | 1 dispatch (1.48 us) + Q + KV + output + block table |
| host | Radeon 8060S (gfx1151), batch = 1, fp16, HND, block size 16 |
| torch | 2.12.0+rocm10.1.0a20260803 |

**M=1** (plain decode): geomean **2.963x** vs Triton, worst 1.63x, best 5.07x, 0 losses in 35 cells, median 89.4 % of roofline.

**M=4** (speculative / MTP): geomean **3.081x** vs Triton, worst 2.12x, best 6.45x, 0 losses in 35 cells, median 61.1 % of roofline.

## What changed since the previous golden

`BFLY` re-swept under the four-stage butterfly that `DPL=32` introduced, which
moved it on two of the four rows: `8/1` and `16/2` at M=4 go from 3 to **0**.
Worth 2.7-3.2 % geomean each, winning at all seven contexts in two independent
runs and regressing nothing. `BFLY=0` puts the whole butterfly on the LDS pipe,
which the main loop still never touches; a four-stage reduction leaves the VALU
with more register pressure per element than a five-stage one did, so the idle
pipe is worth more than it was. `16/1` and `32/4` re-confirmed their existing 4.

This is why a knob is re-swept after a change to its neighbourhood, not
assumed: `16/1` had looked 6.7 % better at `bfly=0` in a three-point sweep on
cells carrying 10-13 % spread, and a clean seven-context pass puts it at 0.999
-- the earlier reading was noise.

Before that, in the same golden:

`DPL=32` with `LDSPLIT=2` on four of the five M=4 configurations. A lane now
carries 32 fp16 of a row instead of 16, so `LPR` halves to 16 and the score
butterfly reduces over 16 lanes in **four dependent stages instead of five**.
`LDSPLIT=2` comes along because `SUB` doubles with `DPL` and the partials stop
fitting the 64 KiB LDS; it is an enabler and measures neutral on its own.

This came out of ablating the kernel's three VALU blocks — thinning each to one
iteration, which returns wrong numbers but bounds what restructuring it could
buy. The result was not what the shape of the problem suggested:

| ablated block | S=8192 | S=32768 | |
| --- | --- | --- | --- |
| none | 192.33 | 771.35 | |
| P@V accumulation | 215.71 | 837.84 | **+12 % slower** |
| Q@K dot product | 212.66 | 821.77 | **+11 % slower** |
| score butterfly | 163.29 | 684.24 | -15 % / -11 % |

Removing arithmetic makes the kernel *slower* in two of three cases: that work
is hiding memory latency for free. Only the butterfly costs real time, because
it is a dependency chain — stride 2 consumes stride 1 — that nothing can hide.
Shortening it by one stage is what these rows buy.

**M=4 only.** At M=1 the kernel already streams at 94 % of the bus, so a shorter
butterfly buys nothing while the extra registers (193 against 127) and the
chunked epilogue cost 2.5-6.6 % — it loses on all five M=1 configurations.
`(8,2,512,4)` is excluded as well; it regresses five of seven contexts.

## The tuning rule changed with this entry

These rows regress `S=128` by 4-9 %, which §4.1's percentage rule forbade. In
absolute time that is **0.47-0.69 us** against **137-203 us** saved at S=32768 —
a 200-400x asymmetry that a percentage threshold cannot see. The rule now reads
in microseconds; see HANDOFF §4.1.

`KPW=8` was re-examined under the same criterion, since it had been rejected
for a +4.6 % / +0.38 us regression at S=128. It stays rejected on its own
merits: it does not combine with `DPL=32` (SUB doubles, so KPWE does too) and
regresses up to +676 us, and where they overlap `DPL=32` is simply better
(geomean 0.915 against 0.981 on `16/1`).

## How this is measured

Rows are ordered by `(S, M)` and then configuration, so every configuration sits
side by side at the same context and token count. That order also walks the
contexts upward across the whole run, which is what warms the allocator -- a
pass starting at a large context reads its first cell per configuration up to
10x slow.

`--reps 1`: `do_bench` already medians many iterations inside one call, so a rep
only resamples allocation and graph placement. The `spread` column is
`do_bench`'s own dispersion within the cell -- read it before believing any
small difference.

No warm-up calls. Every variant is precompiled and the loader is then *sealed*,
so a build that would land inside the timed region raises instead. An unsealed
first call measures 104.59 us against a true 29.99 -- a 258 % error, silently.

The roofline counts Q in, KV in, output out and the block table. It excludes
the split-KV partials our kernel pushes through global memory: at S=128 those
are 514 KiB against 512 KiB of KV, so counting them would double the
denominator on exactly the shapes that look worst.

```bash
export PATH=/scratch/rogarcia/vllm/.venv/bin:$PATH PYTHONPATH=$PWD \
    VLLM_KV_CACHE_LAYOUT=HND
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py \
    --head-dim 512
```

## Per configuration

| M | modelos | Hq | Hkv | GQA | BFLY | GRIDT | DPL | geomean | peor | %roof @32k | us @32k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | gemma-4-E2B-it | 8 | 1 | 8 | 4 | 0 | 16 | 3.00x | 2.85x | 95.3 % | 286.8 |
| 1 | gemma-4-E4B-it | 8 | 2 | 4 | 1 | **1** | 16 | 2.81x | 2.66x | 95.7 % | 569.4 |
| 1 | gemma-4-12b-it | 16 | 1 | 16 | 4 | 0 | 16 | 3.89x | 3.47x | 91.6 % | 298.6 |
| 1 | gemma-4-26B-A4B-it | 16 | 2 | 8 | 4 | 0 | 16 | 3.05x | 2.91x | 99.0 % | 550.6 |
| 1 | gemma-4-31B-it-AWQ +1 | 32 | 4 | 8 | 4 | 0 | 16 | 2.28x | 1.63x | 72.4 % | 1504.8 |
| 4 | gemma-4-E2B-it | 8 | 1 | 8 | 0 | 0 | **32** | 3.14x | 2.20x | 78.1 % | 350.0 |
| 4 | gemma-4-E4B-it | 8 | 2 | 4 | 2 | 0 | 16 | 3.23x | 2.59x | 93.1 % | 585.8 |
| 4 | gemma-4-12b-it | 16 | 1 | 16 | 4 | 0 | **32** | 3.51x | 2.37x | 40.9 % | 669.8 |
| 4 | gemma-4-26B-A4B-it | 16 | 2 | 8 | 0 | 0 | **32** | 3.14x | 2.58x | 73.1 % | 745.8 |
| 4 | gemma-4-31B-it-AWQ +1 | 32 | 4 | 8 | 4 | 0 | **32** | 2.49x | 2.12x | 56.0 % | 1947.3 |

`%roof` still falls roughly as `1 / GQA` across the M=1 rows, so it does not
rank them by quality; compare a row against its own previous value, not against
its neighbours. Two rows no longer fit that pattern: `8/2` at M=1, which GRIDT
moved from 53.2 % to 95.8 %, and `16/1` at M=4, which DPL moved from 33.6 % to
40.8 %.

## Full matrix

| modelos | S | M | Hq | Hkv | D | roofline | Triton | nuestro | vs | %roof | spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-4-E2B-it | 128 | 1 | 8 | 1 | 512 | 2.61 | 14.99 | 4.91 | **3.05x** | 53.2 % | 4.7 % |
| gemma-4-E4B-it | 128 | 1 | 8 | 2 | 512 | 3.67 | 16.96 | 6.09 | **2.78x** | 60.3 % | 5.2 % |
| gemma-4-12b-it | 128 | 1 | 16 | 1 | 512 | 2.67 | 18.08 | 5.21 | **3.47x** | 51.2 % | 3.0 % |
| gemma-4-26B-A4B-it | 128 | 1 | 16 | 2 | 512 | 3.74 | 17.67 | 6.07 | **2.91x** | 61.6 % | 4.3 % |
| gemma-4-31B-it-AWQ +1 | 128 | 1 | 32 | 4 | 512 | 5.99 | 24.79 | 7.88 | **3.15x** | 76.0 % | 2.7 % |
| gemma-4-E2B-it | 128 | 4 | 8 | 1 | 512 | 2.81 | 17.49 | 7.94 | **2.20x** | 35.4 % | 6.3 % |
| gemma-4-E4B-it | 128 | 4 | 8 | 2 | 512 | 3.87 | 21.53 | 8.32 | **2.59x** | 46.5 % | 3.3 % |
| gemma-4-12b-it | 128 | 4 | 16 | 1 | 512 | 3.07 | 27.50 | 8.82 | **3.12x** | 34.8 % | 3.5 % |
| gemma-4-26B-A4B-it | 128 | 4 | 16 | 2 | 512 | 4.13 | 24.15 | 9.35 | **2.58x** | 44.2 % | 3.2 % |
| gemma-4-31B-it-AWQ +1 | 128 | 4 | 32 | 4 | 512 | 6.79 | 31.32 | 11.68 | **2.68x** | 58.1 % | 1.2 % |
| gemma-4-E2B-it | 512 | 1 | 8 | 1 | 512 | 5.79 | 25.40 | 8.35 | **3.04x** | 69.3 % | 2.5 % |
| gemma-4-E4B-it | 512 | 1 | 8 | 2 | 512 | 10.04 | 35.32 | 12.65 | **2.79x** | 79.4 % | 0.9 % |
| gemma-4-12b-it | 512 | 1 | 16 | 1 | 512 | 5.86 | 42.44 | 8.37 | **5.07x** | 70.0 % | 2.0 % |
| gemma-4-26B-A4B-it | 512 | 1 | 16 | 2 | 512 | 10.11 | 37.85 | 12.65 | **2.99x** | 79.9 % | 1.7 % |
| gemma-4-31B-it-AWQ +1 | 512 | 1 | 32 | 4 | 512 | 18.73 | 56.50 | 21.81 | **2.59x** | 85.9 % | 1.7 % |
| gemma-4-E2B-it | 512 | 4 | 8 | 1 | 512 | 5.99 | 41.48 | 11.70 | **3.55x** | 51.2 % | 3.0 % |
| gemma-4-E4B-it | 512 | 4 | 8 | 2 | 512 | 10.24 | 50.32 | 15.06 | **3.34x** | 68.0 % | 1.2 % |
| gemma-4-12b-it | 512 | 4 | 16 | 1 | 512 | 6.26 | 101.49 | 15.74 | **6.45x** | 39.8 % | 2.6 % |
| gemma-4-26B-A4B-it | 512 | 4 | 16 | 2 | 512 | 10.50 | 59.10 | 17.15 | **3.45x** | 61.2 % | 2.0 % |
| gemma-4-31B-it-AWQ +1 | 512 | 4 | 32 | 4 | 512 | 19.53 | 97.73 | 31.94 | **3.06x** | 61.1 % | 1.4 % |
| gemma-4-E2B-it | 1024 | 1 | 8 | 1 | 512 | 10.04 | 38.38 | 12.61 | **3.04x** | 79.6 % | 1.7 % |
| gemma-4-E4B-it | 1024 | 1 | 8 | 2 | 512 | 18.53 | 59.54 | 21.21 | **2.81x** | 87.4 % | 0.9 % |
| gemma-4-12b-it | 1024 | 1 | 16 | 1 | 512 | 10.11 | 57.25 | 12.75 | **4.49x** | 79.3 % | 2.7 % |
| gemma-4-26B-A4B-it | 1024 | 1 | 16 | 2 | 512 | 18.60 | 65.18 | 21.26 | **3.07x** | 87.5 % | 1.0 % |
| gemma-4-31B-it-AWQ +1 | 1024 | 1 | 32 | 4 | 512 | 35.71 | 101.92 | 39.96 | **2.55x** | 89.4 % | 1.8 % |
| gemma-4-E2B-it | 1024 | 4 | 8 | 1 | 512 | 10.24 | 59.02 | 16.98 | **3.48x** | 60.3 % | 1.1 % |
| gemma-4-E4B-it | 1024 | 4 | 8 | 2 | 512 | 18.73 | 85.41 | 23.73 | **3.60x** | 78.9 % | 2.8 % |
| gemma-4-12b-it | 1024 | 4 | 16 | 1 | 512 | 10.50 | 129.82 | 25.73 | **5.05x** | 40.8 % | 1.2 % |
| gemma-4-26B-A4B-it | 1024 | 4 | 16 | 2 | 512 | 19.00 | 99.25 | 27.93 | **3.55x** | 68.0 % | 1.4 % |
| gemma-4-31B-it-AWQ +1 | 1024 | 4 | 32 | 4 | 512 | 36.51 | 162.58 | 59.77 | **2.72x** | 61.1 % | 1.7 % |
| gemma-4-E2B-it | 4096 | 1 | 8 | 1 | 512 | 35.52 | 119.11 | 37.96 | **3.14x** | 93.6 % | 1.6 % |
| gemma-4-E4B-it | 4096 | 1 | 8 | 2 | 512 | 69.49 | 204.74 | 77.06 | **2.66x** | 90.2 % | 0.8 % |
| gemma-4-12b-it | 4096 | 1 | 16 | 1 | 512 | 35.58 | 154.30 | 39.41 | **3.92x** | 90.3 % | 3.1 % |
| gemma-4-26B-A4B-it | 4096 | 1 | 16 | 2 | 512 | 69.55 | 227.62 | 76.08 | **2.99x** | 91.4 % | 0.9 % |
| gemma-4-31B-it-AWQ +1 | 4096 | 1 | 32 | 4 | 512 | 137.62 | 336.22 | 153.35 | **2.19x** | 89.7 % | 0.7 % |
| gemma-4-E2B-it | 4096 | 4 | 8 | 1 | 512 | 35.72 | 165.65 | 49.14 | **3.37x** | 72.7 % | 0.8 % |
| gemma-4-E4B-it | 4096 | 4 | 8 | 2 | 512 | 69.68 | 263.42 | 80.27 | **3.28x** | 86.8 % | 1.6 % |
| gemma-4-12b-it | 4096 | 4 | 16 | 1 | 512 | 35.98 | 292.63 | 86.36 | **3.39x** | 41.7 % | 1.9 % |
| gemma-4-26B-A4B-it | 4096 | 4 | 16 | 2 | 512 | 69.95 | 302.64 | 95.06 | **3.18x** | 73.6 % | 2.2 % |
| gemma-4-31B-it-AWQ +1 | 4096 | 4 | 32 | 4 | 512 | 138.42 | 542.25 | 230.62 | **2.35x** | 60.0 % | 1.7 % |
| gemma-4-E2B-it | 8192 | 1 | 8 | 1 | 512 | 69.49 | 219.00 | 76.76 | **2.85x** | 90.5 % | 1.3 % |
| gemma-4-E4B-it | 8192 | 1 | 8 | 2 | 512 | 137.42 | 417.82 | 146.05 | **2.86x** | 94.1 % | 0.6 % |
| gemma-4-12b-it | 8192 | 1 | 16 | 1 | 512 | 69.56 | 280.96 | 78.50 | **3.58x** | 88.6 % | 2.3 % |
| gemma-4-26B-A4B-it | 8192 | 1 | 16 | 2 | 512 | 137.49 | 453.38 | 150.93 | **3.00x** | 91.1 % | 2.1 % |
| gemma-4-31B-it-AWQ +1 | 8192 | 1 | 32 | 4 | 512 | 273.49 | 652.89 | 303.19 | **2.15x** | 90.2 % | 0.8 % |
| gemma-4-E2B-it | 8192 | 4 | 8 | 1 | 512 | 69.69 | 298.77 | 94.78 | **3.15x** | 73.5 % | 0.7 % |
| gemma-4-E4B-it | 8192 | 4 | 8 | 2 | 512 | 137.62 | 505.87 | 156.70 | **3.23x** | 87.8 % | 4.3 % |
| gemma-4-12b-it | 8192 | 4 | 16 | 1 | 512 | 69.95 | 497.82 | 163.69 | **3.04x** | 42.7 % | 0.6 % |
| gemma-4-26B-A4B-it | 8192 | 4 | 16 | 2 | 512 | 137.89 | 583.27 | 187.79 | **3.11x** | 73.4 % | 1.6 % |
| gemma-4-31B-it-AWQ +1 | 8192 | 4 | 32 | 4 | 512 | 274.29 | 1063.54 | 450.80 | **2.36x** | 60.8 % | 3.5 % |
| gemma-4-E2B-it | 16384 | 1 | 8 | 1 | 512 | 137.43 | 422.60 | 145.80 | **2.90x** | 94.3 % | 0.8 % |
| gemma-4-E4B-it | 16384 | 1 | 8 | 2 | 512 | 273.30 | 823.36 | 284.56 | **2.89x** | 96.0 % | 0.8 % |
| gemma-4-12b-it | 16384 | 1 | 16 | 1 | 512 | 137.50 | 531.97 | 152.83 | **3.48x** | 90.0 % | 1.2 % |
| gemma-4-26B-A4B-it | 16384 | 1 | 16 | 2 | 512 | 273.37 | 897.83 | 280.76 | **3.20x** | 97.4 % | 1.5 % |
| gemma-4-31B-it-AWQ +1 | 16384 | 1 | 32 | 4 | 512 | 545.24 | 1254.21 | 619.37 | **2.02x** | 88.0 % | 3.5 % |
| gemma-4-E2B-it | 16384 | 4 | 8 | 1 | 512 | 137.63 | 575.34 | 177.97 | **3.23x** | 77.3 % | 4.1 % |
| gemma-4-E4B-it | 16384 | 4 | 8 | 2 | 512 | 273.50 | 993.05 | 294.82 | **3.37x** | 92.8 % | 0.5 % |
| gemma-4-12b-it | 16384 | 4 | 16 | 1 | 512 | 137.90 | 858.95 | 324.98 | **2.64x** | 42.4 % | 0.6 % |
| gemma-4-26B-A4B-it | 16384 | 4 | 16 | 2 | 512 | 273.77 | 1138.01 | 354.62 | **3.21x** | 77.2 % | 0.6 % |
| gemma-4-31B-it-AWQ +1 | 16384 | 4 | 32 | 4 | 512 | 546.04 | 2083.40 | 927.95 | **2.25x** | 58.8 % | 1.6 % |
| gemma-4-E2B-it | 32768 | 1 | 8 | 1 | 512 | 273.32 | 847.64 | 286.75 | **2.96x** | 95.3 % | 3.7 % |
| gemma-4-E4B-it | 32768 | 1 | 8 | 2 | 512 | 545.06 | 1644.66 | 569.40 | **2.89x** | 95.7 % | 0.8 % |
| gemma-4-12b-it | 32768 | 1 | 16 | 1 | 512 | 273.38 | 1038.72 | 298.59 | **3.48x** | 91.6 % | 4.4 % |
| gemma-4-26B-A4B-it | 32768 | 1 | 16 | 2 | 512 | 545.12 | 1779.60 | 550.58 | **3.23x** | 99.0 % | 0.3 % |
| gemma-4-31B-it-AWQ +1 | 32768 | 1 | 32 | 4 | 512 | 1088.74 | 2450.03 | 1504.77 | **1.63x** | 72.4 % | 18.8 % |
| gemma-4-E2B-it | 32768 | 4 | 8 | 1 | 512 | 273.52 | 1117.37 | 350.04 | **3.19x** | 78.1 % | 11.6 % |
| gemma-4-E4B-it | 32768 | 4 | 8 | 2 | 512 | 545.26 | 1952.21 | 585.77 | **3.33x** | 93.1 % | 0.8 % |
| gemma-4-12b-it | 32768 | 4 | 16 | 1 | 512 | 273.78 | 1588.78 | 669.79 | **2.37x** | 40.9 % | 0.9 % |
| gemma-4-26B-A4B-it | 32768 | 4 | 16 | 2 | 512 | 545.52 | 2219.34 | 745.82 | **2.98x** | 73.1 % | 1.2 % |
| gemma-4-31B-it-AWQ +1 | 32768 | 4 | 32 | 4 | 512 | 1089.53 | 4125.70 | 1947.29 | **2.12x** | 56.0 % | 5.8 % |

## Caveats

The two `gemma-4-31B-it` rows share one configuration, so the matrix measures
it once and labels it `+1`.

Every gemma-4 model in `tools/shapes.csv` carries a second, conflicting entry
at D=256 with a different Hkv. If those turn out to be the real shapes, this
whole file describes configurations no deployed model runs. Unresolved.

`GRIDT` and `DPL` were measured on D=512 only. Their effect on D=64, D=128 and
D=256 is unknown; both default to the old behaviour there.

Two cells carry a spread that makes them unusable for small comparisons:
`32/4` at M=1 and S=32768 (20.8 %) and `8/1` at M=4 and S=32768 (10.1 %). Both
were already the noisiest cells in the previous golden.
