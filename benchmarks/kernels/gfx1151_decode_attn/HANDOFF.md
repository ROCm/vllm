# Handoff — gfx1151 decode attention

Written for whoever picks this up next, human or agent. Read this before the
reports: it says where things stand, what is still open, and which of the open
items is actually worth doing.

Everything here was measured on a Radeon 8060S (gfx1151), batch = 1 sequence,
fp16, HND. Nothing is estimated.

Three files carry the record and they do different jobs:

| file | what it is |
| --- | --- |
| this one | state, open work, traps |
| `OPTIMIZATIONS.md` | one entry per optimisation landed **or rejected**, with before/after ISA, the C++ diff and the numbers |
| `golden/` | best measured result per head size; replace only when beaten |
| `reports/` | the original investigation record. **Its `%roof` numbers are superseded** -- see §3 |

---

## 1. Where things stand

The kernel is integrated and selected by a backend, not a standalone
experiment. `RDNA35_HIP_ATTN` serves 49 of the 50 shapes in `tools/shapes.csv`;
the one it refuses is D=96, which is three fp16 per lane and has no single load
width. One kernel, `csrc/rocm/rdna35_decode_attn.cu`.

`tools/shapes.csv` also carries nine sliding-window rows (gemma-3, gemma-4,
paligemma2). The kernel is full-context, so every tool skips them and says so.

### The honest performance picture

Full matrix, all 27 configurations x 7 contexts x M in {1,4}, against the
Triton kernel vLLM ships:

| M | geomean vs Triton | losses | median %roof |
| --- | --- | --- | --- |
| 1 (plain decode) | 1.176x | 61/182 | 71.3 % |
| 4 (speculative) | 0.975x | 80/182 | 47.7 % |

That average hides a very wide spread by head size:

| D | M=1 | M=4 | tuned configs |
| --- | --- | --- | --- |
| 64 | 0.67x | **0.47x** | 0/4 |
| 128 | 0.86x | 0.67x | 1/10 |
| 256 | 1.33x | 1.15x | 1/7 |
| 512 | **2.89x** | **2.91x** | 5/5 |

D=512 is fully tuned and wins everywhere with zero losses. D=64 and D=128 are
where we lose, and they are almost entirely untuned. **The single largest
available win is calibrating the other head sizes the way D=512 was**; see §4.1.

These numbers predate the LDS change in `OPTIMIZATIONS.md` 003, which is worth
another ~2-3 % across the board. `golden/d512.md` is current.

---

## 2. The loop

```text
matrix.py / roofline.py  ->  worst shape  ->  ISA, profiler, measured wiki
        ^                                             |
        +----------  golden + OPTIMIZATIONS  <-  validate  <-+
```

| tool | what it answers | cost |
| --- | --- | --- |
| `tools/matrix.py` | what we ship: every configuration x context x M vs Triton | 1m52s for D=512, ~13 min for all |
| `tools/roofline.py` | the regression gate, and where to look next | minutes |
| `tools/sweep.py` | one configuration, a matrix of knobs x contexts | seconds |
| `tools/tune.py` | searches (NSEG, MSPLIT) scoring the **worst** context | long |
| `tools/check.py` | correctness, with `--repeat` and `--mutate` | seconds |
| `tools/shapeset.py` | shared shape loading, filtering and the roofline | -- |
| `tools/dump_asm.sh` | ISA for one variant | seconds |

All of them share `shapeset.py`, so `--hkv 8`, `--gqa 4`, `--head-dim 512` and
`--filter <model>` select the same rows everywhere. Call them **without
arguments** unless you mean something specific -- the defaults are the
considered choice. `matrix.py` keeps all seven contexts; the tuning tools use
128 / 16384 / 32768, which is the shortest context where fixed cost dominates
plus the two longest where bandwidth does.

`roofline.py`'s `ran` column is not optional reading. The backend falls back to
Triton silently, and a harness that does not check reports Triton as ours.

### Measurement protocol, which changed

**`--reps` defaults to 1.** `do_bench` already medians many iterations inside
one call; a rep only resamples allocation and graph-capture placement. Measured
drift of single-rep against the old `--reps 3`: 0.33 % median, 1.66 % p90,
4.43 % worst -- the same envelope, at a third of the cost.

**Every tool prints a `spread` column**, which is `do_bench`'s own dispersion
inside the cell (1.5 % median, 6.4 % worst typically). Read it before believing
any small difference. Cells occasionally report 100 %+ spread; those are not
measurements.

**No warm-up calls.** Every variant is precompiled and the loader is then
*sealed*: `load()` raises `UnexpectedBuildError` rather than building something
inside the timed region. This replaced a discarded call per cell and halved
every run. It has since caught four real precompile/runtime divergences that
the warm-up had been silently absorbing.

---

## 3. The roofline, which also changed

`shapeset.roofline_us()` counts one kernel dispatch (1.48 us, the measured
per-dispatch overhead on this SKU under a HIP graph), plus Q in, KV in, output
out, and the block table. It previously counted **KV alone**.

It deliberately excludes the split-KV partials our kernel pushes through global
memory: at S=128 those are 514 KiB against 512 KiB of KV, so counting them
would double the denominator on exactly the shapes that look worst. It counts
one dispatch, not NSEG of them, for the same reason -- the decomposition is
ours, the single launch is not.

The correction is +25 % at S=128/M=4 and under +0.1 % at S=32768, and it is
larger at M=4 than at M=1, so **it was systematically biasing M=4 to look
worse**. Any `%roof` in `reports/` or in an old golden is understated at short
context. `vs Triton` is unaffected.

---

## 4. What to do next, in order

### 4.1 Calibrate the other head sizes

D=512 is done: 10 rows in `_TUNED`, zero losses, 2.9x. D=64, D=128 and D=256
are essentially untuned and are where every loss is. D=256 is the best target
-- 14 configuration/M pairs, only one tuned, already 1.33x/1.15x with 91 % of
cells winning, and one real loser (`Hq=24/Hkv=4/M=4` at 0.72x).

D=64 at 0.47x is the worst but is only 4 configurations, `DPL=2` is the
narrowest load in the table, and the kernel comment already flags it as
latency-bound. Expect a structural answer there, not a knob.

The loop per configuration: `sweep.py --hq .. --hkv .. --head-dim .. --m ..
--bfly 0 1 2 3 4`, pick by best geomean, add the row, move on. **Add the row
before moving to the next configuration.** Nothing measured may stay on the
fallback.

#### The acceptance rule reads in microseconds, not percent

This rule used to be "regress no cell by more than the ~3.3 % harness noise".
It was changed when `DPL=32` (OPTIMIZATIONS 007) hit it: that knob regresses
S=128 by 4-9 % and wins 9-17 % at long context. In absolute time the regression
is **0.47-0.69 us** and the win is **137-203 us** — a 200-400x asymmetry a
percentage threshold cannot see. Cells are three orders of magnitude apart in
absolute cost across the context range, so equal percentages are not equal
costs.

A candidate is accepted when the geomean improves **and** no cell regresses by
a large absolute amount. That second half is what still rejects things: `KPW=8`
on `(32,4,512,4)` has a better geomean than the default and regresses S=32768
by +843 us, and `DPL=32` combined with `KPW=8` regresses up to +676 us. Both
are refused.

Report both numbers when proposing a row. A percentage alone at S=128 means
almost nothing, and a percentage alone at S=32768 hides how much time it is.

### 4.2 Find out why Triton beats us at long context

Unchanged and still the least-charted territory. Triton's split-KV partials are
fp32 and the same shape as ours (`triton_attn.py:211`, allocated once in the
metadata builder), so they pay the same overhead we do and still win. The
interesting question is not "can we cut partials" but "why is their
identical-shaped overhead cheaper".

### 4.3 Make the per-workgroup fixed path cheaper

At `S=128, M=4` the call is **81 % fixed cost** (fit over the context range;
the per-key slope is flat at ~11.2 ns/key from 128 to 32768). Of that, 1.48 us
is dispatch and the rest is prologue, LDS reduction and partial publication.
§6 shows this cannot be amortised by more parallelism or removed by layout, so
it needs the fixed path itself to get cheaper -- a restructure, not a knob.

### 4.4 Support D=96, and the batch axis

One shape, Phi-3.5-vision; 96/32 is three fp16 per lane. And the kernel serves
one sequence -- the backend rejects batch > 1 to Triton, so in real batched
decode we do not participate. B and NSEG are substitutes, so introducing B
pushes NSEG towards 1 and makes most of the split-KV machinery inert. A
different regime, not an extension of this one.

---

## 5. Traps that cost us time

### 5.1 Measuring one context and believing it

`roofline.py` measures S=128. Right for finding fixed cost, wrong for judging
whether we ship. `reports/f11-shape-coverage.md` claims we win every shape;
that is true only at S=128.

### 5.2 Profiling on a hot cache

The harness sets `min_working_set_mb=96`, which at `S=128, D=512, Hkv=1` means
`layers_for_working_set` picks **384 layers** so every layer's KV is cold. A
micro-driver reusing one 256 KiB buffer runs hot and measures a different
kernel. The LDS fix looked like -21.7 % hot and is ~2-3 % in the harness; a
rejected variant looked 28 % worse hot and is ~1.5 % worse in the harness.
**Profile with the harness's working set or every number is flattered.**

### 5.3 Overriding a knob the backend would not have chosen

`self._variant` then disagrees with `_prepare` forever. Two consequences: the
scratch is reallocated every forward (memoise it -- every tool does), and
`replace()` carries forward a MSPLIT that `__post_init__` already raised for
the *old* BLOCK. `sweep.py` rebuilds from `_knobs_for` before applying an
override for this reason; anything else that overrides knobs must do the same.

### 5.4 Compiling immediately before measuring

Saturates the cores and moves the SoC clock. The tools settle for 5 s. See
`reports/00-protocol.md`.

### 5.5 Parallel builds

Bounded by memory (a build peaks at 1.15 GB), and previously racy: torch
hipifies on ROCm keyed on the **absolute source path** in a process-global
dict, so every variant collided on one key and ~10 % of builds silently
produced unhipified CUDA. `load()` now stages a private copy of the source per
variant. If builds start vanishing again, look there first.

### 5.6 `.gitignore` eats new files

`*.csv` and `*_hip*` both match things we add. `shapes.csv` needed `git add -f`.

### 5.7 An identifier the preprocessor has not seen is 0 inside `#if`

The MSPLIT fallback read `NWAVE` before it was defined and always chose 1.

### 5.8 `max_abs` is not a sufficient correctness criterion

A causal off-by-one gives `max_rel = 1.2e-03` at S=2048 and passes a 2e-2
threshold. Error scales as ~1/S while tolerance is fixed, so **longer contexts
hide more bugs**. Validate with `max_rel <= 1e-3`, a short S, the S=50 partial
tile, `--repeat`, and the `--mutate 1` negative control.

### 5.9 Harness flakiness

Treat anything under ~2 % as noise. Do not compare a cell against a run that
starts at a large context: a narrow `--contexts 8192 16384 32768` pass inflated
the first cell of each configuration by up to 10x -- 161 us read as 1094 --
because walking up from S=128 is what warms the allocator.

---

## 6. Refuted — do not re-derive

`OPTIMIZATIONS.md` carries the full tables. Summary:

| idea | verdict |
| --- | --- |
| WGP alignment as a criterion | refuted three times |
| **Occupancy is the problem** | refuted from three directions -- entry 004 |
| NSEG up for more occupancy | occupancy 9.1 -> 13.8 %, time +19 % short / +59 % long |
| BLOCK down for more workgroups | equal waves, better CU spread, 47 % slower |
| BLOCK up | 10.9 % better at S=128, 12 % worse at 32k, unstable |
| Zero LDS bank conflicts | rotate swizzle reaches 0.00 % and is slower: it costs the wide store |
| A pad that keeps `ds_store_b128` | `pad=4` ties `pad=1` and costs 18 % more LDS |
| DPP for the score butterfly | -34 % at 32k; it cannot pair in VOPD |
| `__builtin_assume` for speed | -26 instructions, no measurable time. Kept for codegen only |
| `bound_ctrl` on permlane | -79 instructions, no time, VGPR up. Reverted |
| Separate `reduce_segments` kernel | 3.3 % worse at S=128; the in-kernel tail beats a second dispatch |
| Contiguous KV runs (`ILV=0`) | 658 -> 1032 us at 32k |
| Non-temporal K/V loads | 1.84x worse |
| Instruction count predicts time | **failed four times**; measure |

Two sentences worth internalising. The design report's:

> In this regime the scarce resource is cache locality, not grid parallelism.

and its corollary from this session: **the obvious extremum keeps losing to a
middle value.** All-VALU butterfly lost to three-to-one; zero bank conflicts
lost to a partial fix; maximum occupancy lost to the default.

---

## 7. Reproducing

### Environment

There is **no `.venv` in the worktree**; everything runs from the main tree's.

| | |
| --- | --- |
| venv | `/scratch/rogarcia/vllm/.venv` |
| torch | 2.12.0+rocm10.1.0a20260803 |
| compiled `.so` | `/scratch/rogarcia/vllm/vllm/*.so`, symlinked into the worktree |

`PYTHONPATH=$PWD` makes `import vllm` resolve to the worktree. Without it you
benchmark the installed tree -- 281 us instead of 199, no error, no warning.
Two agents hit that independently. Verify:

```bash
PYTHONPATH=$PWD python -c "import vllm; print(vllm.__file__)"
```

The `.so` files are gitignored, so a fresh worktree needs them linked:

```bash
for f in /scratch/rogarcia/vllm/vllm/*.so; do ln -sf "$f" vllm/; done
```

`amd-gpu-lock` needs `amd-smi`, which only appears with the venv on PATH. The
login shell is csh, so `source`/`export` go inside `bash -c`.

### Commands

```bash
cd <worktree>
export PATH=/scratch/rogarcia/vllm/.venv/bin:$PATH PYTHONPATH=$PWD \
    VLLM_KV_CACHE_LAYOUT=HND

# the gate, and where to look next
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/roofline.py

# what we ship
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py

# one configuration against a knob
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/sweep.py \
    --hq 8 --hkv 1 --head-dim 512 --m 4 --bfly 0 1 2 3 4

# correctness: partial tile, repeated launches, negative control
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/check.py \
    --hq 32 --hkv 8 --head-dim 128 --repeat 10
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/check.py \
    --hq 32 --hkv 8 --head-dim 128 --mutate 1 --layouts 1

amd-gpu-lock python -m pytest tests/kernels/attention/test_rdna35_hip_decode.py
```

### Profiling

`rocprofv3` is in the same venv and works with torch loaded, despite the
wiki's warning. Counters that earned their keep: `LDSBankConflict`,
`MemUnitBusy`, `OccupancyPercent`, `MeanOccupancyPerActiveCU`, `SQ_WAVES`,
`SQ_INSTS_VALU`. One counter group per pass; filter with
`--kernel-include-regex decode_attn`. `--kernel-trace` gives per-dispatch
kernel duration and the gap to the next dispatch, which is how we separated
kernel time from launch overhead. Re-read §5.2 before trusting any of it.
