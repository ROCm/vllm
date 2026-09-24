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

By head size, against the Triton kernel vLLM ships, geomean over 7 contexts:

| D | M=1 | M=4 | tuned configs | measured |
| --- | --- | --- | --- | --- |
| 64 | 0.67x | **0.47x** | 0/4 | before 2026-09-24 |
| 128 | 0.86x | 0.67x | 1/10 | before 2026-09-24 |
| 256 | 1.33x | 1.15x | 1/7 | before 2026-09-24 |
| 512 | **2.96x** | **3.08x** | 5/5 | 2026-09-24, `golden/d512.md` |

D=512 is fully tuned and wins everywhere with zero losses in 70 cells. D=64 and
D=128 are where we lose, and they are almost entirely untuned. **The single
largest available win is calibrating the other head sizes the way D=512 was**;
see §4.1.

**Only the D=512 row is current.** The other three are from an earlier full-
matrix pass and have not been re-measured since. Everything landed on
2026-09-24 (`GRIDT`, `DPL`/`LDSPLIT`, the `BFLY` re-sweep) is scoped to
`_TUNED` rows that name D=512 configurations, so those head sizes are
unaffected by construction -- with one exception worth knowing: restoring
`LDSPLIT` moved the m/l stores out of the accumulator loop in the shared
epilogue, which changes codegen on **every** shape (1340 -> 1329 instructions
at D=512). It is semantically identical and measured neutral on D=512, but it
was not re-measured on D=64/128/256.

There is deliberately no whole-matrix aggregate here any more. The old one
(1.176x at M=1, 0.975x at M=4) averaged a tuned head size with three untuned
ones, which made it move for reasons that had nothing to do with the change
being evaluated. Read the per-head-size row, or `golden/`.

---

## 2. The loop

```text
matrix.py / roofline.py  ->  worst shape  ->  ISA, profiler, measured wiki
        ^                                             |
        +----------  golden + OPTIMIZATIONS  <-  validate  <-+
```

| tool | what it answers | cost |
| --- | --- | --- |
| `tools/matrix.py` | what we ship: every configuration x context x M vs Triton. `--gridt/--kpw/--dpl/--ldsplit` force a knob across the whole run | ~2 min for D=512, ~13 min for all |
| `tools/roofline.py` | the regression gate, and where to look next | minutes |
| `tools/sweep.py` | one configuration, a matrix of knobs x contexts | ~1 min warm, ~2 cold |
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

D=512 is done: 10 rows in `_TUNED`, zero losses, **2.96x at M=1 and 3.08x at
M=4**. D=64, D=128 and D=256 are essentially untuned and are where every loss
is. D=256 is the best target -- 14 configuration/M pairs, only one tuned,
already 1.33x/1.15x with 91 % of cells winning, and one real loser
(`Hq=24/Hkv=4/M=4` at 0.72x).

Four knobs now carry D=512 and **none of them has been measured anywhere else**:
`GRIDT` (entry 005), `DPL`/`LDSPLIT` (007) and the re-swept `BFLY` (008). All
default to the old behaviour outside the rows that name them, so the other head
sizes are unaffected -- but that also means the first thing to try on D=256 is
the knobs that already paid, not a fresh search. `DPL` in particular is a rule,
not a table entry, for every shape that does not override it: at D=256 the
default is already `DPL=8, LPR=32`, five butterfly stages, so `DPL=16` is the
analogous move and costs `SUB=2` in LDS the same way.

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

The *per-key* path, by contrast, is now charted. Entry 007 ablated it -- each
VALU block thinned to one iteration, which returns wrong numbers and so bounds
what restructuring it could buy:

| ablated | S=8192 | S=32768 | |
| --- | --- | --- | --- |
| nothing | 192.33 | 771.35 | |
| P@V | 215.71 | 837.84 | **+12 % slower** |
| Q@K | 212.66 | 821.77 | **+11 % slower** |
| butterfly | 163.29 | 684.24 | -15 % / -11 % |

**Removing arithmetic makes the kernel slower in two of three cases**: P@V and
Q@K are hiding memory latency for free. Do not try to cut them. Only the
butterfly costs real time, because its strides are a dependency chain nothing
can overlap -- and entry 007 has already taken one stage out of it. What is
left there is one more stage (`DPL=64` would need `LPR=8`, and `SUB=4` puts the
partials far over the LDS ceiling even chunked), so expect little.

`ABLATE` is still in the kernel. It is measurement scaffolding: every bit must
make `check.py` fail, and if one passes, that block was already dead and the
number means nothing.

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

**The negative control is vacuous at M=1.** `MUTATE=1` admits one key past the
causal bound, but at M=1 the single query token already attends to every key:
`ctx + m` is `S-1`, the extra index is `S`, which does not exist and whose V is
zeroed. So the mutated kernel returns the correct answer and `check.py` reports
`FAIL` -- meaning the control did not fire, not that the kernel is wrong.
Validate an M=1 shape's causal masking at `--m 4` on the same configuration.

### 5.9 Harness flakiness

Treat anything under ~2 % as noise. Do not compare a cell against a run that
starts at a large context: a narrow `--contexts 8192 16384 32768` pass inflated
the first cell of each configuration by up to 10x -- 161 us read as 1094 --
because walking up from S=128 is what warms the allocator.

### 5.10 The sweep's label is not what got compiled

`KernelVariant.__post_init__` silently raises MSPLIT until the partials fit
LDS, so `--msplit 1` can build `ms2` and `--msplit 2` can build `ms4`. Two rows
of a sweep then show the same binary under different labels, and the small gap
between them reads as a knob effect. This happened three times in one session:
`block=512 msplit=2` was really `ms4`, `msplit=1 ldsplit=1` was really `ms2`,
and `dpl=32 ldsplit=1` was really `ms4` -- the last one nearly landed as
evidence that DPL was slow when it was measuring MPW=1.

**Check the variant suffix in the compile log, not the sweep's column header.**
Two rows with near-identical times are the tell.

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
| Put M on the grid instead of in the workgroup | proxy with the identical grid, per-WG work and working set: 376 us against 194, even dispatched head-fastest as intended. M inside the workgroup buys K/V reuse *in registers*; moving it out turns that into memory traffic |
| Occupancy, again, at M=4 | the proxy gets 96 VGPRs and the full 16 waves/SIMD and is the worst row measured; the three M=4 decompositions all sit at 224 VGPRs and 6 waves yet differ. Occupancy orders nothing here either |
| Cutting VGPRs to raise occupancy | wrong direction: registers are the reuse mechanism. Splitting M=4 into two M=2 launches does not even lower them (MPW is what costs, and MSPLIT already halved it) and is 5-8 % worse |
| Driving the KV re-read factor to 1.00x | a M=4 configuration with *perfect* reuse exists (MPW=1) and is the slowest of four, at 70 GB/s. Reuse and memory-level parallelism are one dial: NSLICE 2/4/8 gives reuse 1.00x/1.96x/7.57x and 71/148/412 GB/s |
| `time ~ max(bytes/BW, M*c)` | fitted M=1, M=2 and M=4 to within 10 % and was refuted by the ablation an hour later. A three-point fit is not a mechanism |
| KPW=8 | entry 006. Flattered by a three-context sweep, killed by the full matrix |

Two sentences worth internalising. The design report's:

> In this regime the scarce resource is cache locality, not grid parallelism.

and its corollary: **the obvious extremum usually loses to a middle value** --
all-VALU butterfly lost to three-to-one, zero bank conflicts lost to a partial
fix, maximum occupancy lost to the default. *Usually*, not always: entry 008
moved two rows to `BFLY=0`, an extremum, once entry 007 changed the shape it
was balancing. A heuristic about knobs is not a law about them, and the only
way to tell is to re-sweep after the neighbourhood changes.

### A note on `%roof` at M=4

It is a poor guide there and has cost time twice. The denominator counts KV
once, but M=4 reads the same bytes as M=1 and does four times the arithmetic
over them, so a low `%roof` at M=4 says the shape is compute-heavy, not that
bandwidth is being wasted. `16/1/512` at M=4 read 37 % and was never
bandwidth-limited. Compare an M=4 candidate by achieved GB/s and by absolute
microseconds, and compare a row against its own previous value.

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
`SQ_INSTS_VALU`, `FETCH_SIZE`. The CSV also reports `VGPR_Count`,
`LDS_Block_Size` and the kernel name, which is the cheapest way to confirm
which variant actually ran (see §5.10).

**`FETCH_SIZE` is not DRAM traffic.** It matches the distinct bytes exactly on
a clean streaming case (16.02 MiB measured against 16.00 expected at M=1), but
elsewhere it implies rates above the 247 GB/s bus -- 776 GB/s on one shape --
so it counts request volume including cache hits. Use it for direction and for
comparing two variants of the same shape, never as an absolute byte count, and
never compare it across contexts: it runs 1.42x at S=8192 and 4.36x at S=32768
on the same configuration purely because 16 MiB of KV fits the 32 MiB MALL and
64 MiB does not. One counter group per pass; filter with
`--kernel-include-regex decode_attn`. `--kernel-trace` gives per-dispatch
kernel duration and the gap to the next dispatch, which is how we separated
kernel time from launch overhead. Re-read §5.2 before trusting any of it.
