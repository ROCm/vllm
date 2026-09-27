# Handoff — gfx1151 decode attention

Written for whoever picks this up next, human or agent. Read this before the
reports: it says where things stand, what is still open, and which of the open
items is actually worth doing.

Everything here was measured on a Radeon 8060S (gfx1151), batch = 1 sequence,
fp16 unless it says bf16, **HND** (see §5.1). Nothing is estimated.

| file | what it is |
| --- | --- |
| this one | state, open work, traps |
| `OPTIMIZATIONS.md` | one entry per optimisation landed **or rejected**, with the numbers. 001-008 describe the previous (dot) kernel; 009 is the rewrite, 010-023 the commits below |
| `reference/` | the previous per-q-head dot kernel and its D=512 golden, kept for comparison only |
| `golden/` | best measured result per head size for the WMMA kernel, with each configuration's ceiling; replace only when beaten. `bf16.md` is the same for bf16, `swa.md` for the sliding-window configurations |
| `reports/` | the original investigation record, about the dot kernel. Its `%roof` numbers are superseded |

---

## 1. Where things stand

`RDNA35_HIP_ATTN` serves 49 of the 50 shapes in `tools/shapes.csv`, one
sequence or a batch of them (027), with one kernel, `csrc/rocm/rdna35_decode_attn.cu`, in fp16 and in bf16; it refuses
D=96 (three elements per lane) and falls back to Triton.  The nine
sliding-window rows (Gemma 3/4, PaliGemma 2; six distinct configurations)
are served too, with rows of their own keyed by window (`_TUNED_SWA`, 022);
the tools select them with `--windowed`.

The kernel is the WMMA rewrite of OPTIMIZATIONS 009: one workgroup per
`(kv head, row group, KV segment)`, both products on
`v_wmma_f32_16x16x16_f16` (`_bf16` in bf16), knobs
`NSEG / RG / MINB / NW / DSPL / RSPL / PF / CPUB / VINLDS` per configuration in `_TUNED`
(`vllm/v1/attention/backends/rdna35_hip_attn.py`).

**One configuration per `(Hq, Hkv, D, M)`, chosen for the whole context
range.**  S is a runtime value: it may bound the work (active segments,
masking) but never select knobs or a decomposition.  `DOT=1` picks the
per-q-head dot-product decomposition of reference/ (018) instead of WMMA,
for the whole configuration, when it wins over the full range.

Commits on top of it, 2026-09-25:

| commit | what |
| --- | --- |
| `e79bc3cb18` (010) | preamble and softmax: LDS-only barriers, unconditional Q load, permlane fetch-inactive, causal mask only on the tail tile. S=128 +1 to +10 % |
| `c17a8bda13` (011) | split-KV merge shared across the segments when a group's partials reach 64 KiB. D=512 S=128 up to +56 % |
| `16621e4856` (012) | `16/2/512` M=1 re-tuned to `rg=1` (§5.2) |
| `88fb3dc37b` (013) | bf16: a compile-time dtype, bf16 WMMAs, `--dtype` in the tools (§1, bf16) |
| `83415bcc26` (014, 015) | S read on the device (a host S was frozen into full CUDA graphs); RSPL, rows split inside the workgroup over a shared tile |
| `9ae12f67da` (016) | two decompositions per build switched by S -- **removed** in `1e3e0ade83` (020): knobs may not depend on S |
| `8cd55c5497` (017) | PF, a second tile in flight where it fits |
| `c219ce6083`, `5d1204997f` (018) | the dot decomposition as a mode; short mode of all twelve D=256/512 M=1 rows (S=128 1.12-1.34x); D=64 PF rows |
| `2c7b87b6e8` (021, 022) | full-range re-tune (dot rows at D=256/512 M=1, PF at D=64), `_TUNED_BF16`, sliding window in kernel/backend/tests |
| `82ef301b1d` (022) | `_TUNED_SWA` rows; the harness allocates only a window's blocks |
| `6f8a9c9f6c` (025) | CPUB: split-KV partials staged in LDS and written a line at a time |
| `29feb4a997` (026) | the backend requires HND; gemma-4-E2B end to end |
| `813f1f5b95` (027) | batched decode: grid.y per sequence, uniform query lengths up to 8 |
| `23751209a2` (028) | mixed batches: decodes on the kernel, prefills on Triton |
| `f7c00e52d0` (031) | VINLDS: V in LDS, next tile issued before this one's compute |

### The performance picture

`matrix.py`, HND, all 52 configuration/M pairs, geomean over the seven
contexts, measured 2026-09-26 at `2c7b87b6e8` (`golden/`; bf16 in
`golden/bf16.md`, whose nine dot shapes use their WMMA rows):

| D | M | configs | vs Triton | median configuration %roof | >= 90 % roof | S=128 median %roof |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 1 | 4 | 1.27x | 87.1 % | 1 | 56.0 % |
| 64 | 4 | 4 | 1.29x | 87.2 % | 1 | 57.6 % |
| 128 | 1 | 10 | 1.20x | 88.5 % | 1 | 69.0 % |
| 128 | 4 | 10 | 1.26x | 88.0 % | 1 | 63.1 % |
| 256 | 1 | 7 | 1.39x | 86.0 % | 1 | 62.5 % |
| 256 | 4 | 7 | 1.53x | 81.2 % | 0 | 53.5 % |
| 512 | 1 | 5 | 2.95x | 85.0 % | 1 | 58.0 % |
| 512 | 4 | 5 | 3.93x | 78.5 % | 0 | 48.5 % |

Against the golden from before 014 (same harness): median cell 0.998x;
the dot rows at D=256/512 M=1 1.03-1.13x (S=128 up to 1.25x), the PF and
re-tuned WMMA rows 1.01-1.04x; unchanged rows 0.99x, the device-side S of
014 (1-2 % at S=128) and noise.  Eleven long cells (S >= 16k) are under
90 % of roof, all of them already under it before: `32/2/128` M=4 16k/32k,
`8/1/256` M=4 16k, `16/2/64` M=4 16k/32k, `14/2/64` M=4 16k/32k,
`16/1/512` M=4 16k, `16/2/256` M=4 16k, `16/2/128` M=4 16k and M=1 16k --
85.0-89.8 %, see §4.5.

**90 % of roof is not reachable everywhere.** `tools/floor.py` times a kernel
that does nothing but stream the same bytes after one dependent page-table
read, under the same harness. Its score is the ceiling for any kernel: 19 of
the 52 pairs have a ceiling under 90 %, mostly few-kv-head shapes whose short
contexts are all dispatch and latency.

### bf16

Same kernel, same `_TUNED` knobs, bf16 WMMAs (OPTIMIZATIONS 013). An
interleaved A/B against the fp16 build of every pair puts it at fp16's speed
at long context and 0.3 % behind at S=128 (median; worst 1.1 %), the cost of
rounding the output. Correctness is bounded at 8e-3 relative, not 1e-3:
rounding the output to bf16 alone costs up to 3.9e-3.

`matrix.py --dtype bf16`, all 52 pairs, Triton in bf16 too (`golden/bf16.md`).
The nine D=256/512 M=1 shapes whose fp16 row is the dot decomposition run
their WMMA row in bf16 (`_TUNED_BF16`, 021): the dot rows lose up to 0.73x at
32k in bf16.

| D | M | configs | vs Triton | median configuration %roof | fp16 (`golden/d*.md`) | >= 90 % roof | S=128 median %roof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 1 | 4 | 1.28x | 86.9 % | 87.1 % | 1 | 55.9 % |
| 64 | 4 | 4 | 1.30x | 87.4 % | 87.2 % | 1 | 57.5 % |
| 128 | 1 | 10 | 1.20x | 88.2 % | 88.5 % | 1 | 68.9 % |
| 128 | 4 | 10 | 1.27x | 87.9 % | 88.0 % | 1 | 63.0 % |
| 256 | 1 | 7 | 1.32x | 80.9 % | 86.0 % | 0 | 50.9 % |
| 256 | 4 | 7 | 1.53x | 81.2 % | 81.2 % | 0 | 53.2 % |
| 512 | 1 | 5 | 2.89x | 81.9 % | 85.0 % | 0 | 54.0 % |
| 512 | 4 | 5 | 3.89x | 78.4 % | 78.5 % | 0 | 48.2 % |

Four cells of 364 trail Triton by 1-2 %, D=128 M=1 at S=32768.

### Sliding window

`matrix.py --windowed` (`golden/swa.md`), fp16; bf16 within 0.6 %.  Past
S = window every context reads the same bytes, so the plateau is what a
long conversation sees; `ceiling` is `floor.py --windowed` at those bytes.

| configuration | window | M=1 plateau | M=4 plateau | ceiling | vs Triton (M=1 / M=4) |
| --- | --- | --- | --- | --- | --- |
| 32/16/256 | 1024 | 91.9 % | 91.7 % | 92.4 % | 1.40x / 1.49x |
| 16/8/256 | 1024 | 95.2 % | 91.5 % | 97.1 % | 1.62x / 1.73x |
| 8/4/256 | 4096 | 90.4 % | 87.6 % | 92.4 % | 1.32x / 1.31x |
| 8/4/256 | 1024 | 88.3 % | 81.5 % | 93.9 % | 1.72x / 1.83x |
| 8/2/256 | 512 | 75.6 % | 64.2 % | 84.6 % | 2.84x / 3.50x |
| 8/1/256 | 512 | 64.0 % | 48.1 % | 80.8 % | 3.76x / 3.25x |

The small windows are short contexts forever: their gap is the fixed cost of
§4.7, not the window.

---

## 2. The loop

```text
matrix.py  ->  worst shape  ->  timeline (TIMING), ISA, counters
    ^                                         |
    +----  OPTIMIZATIONS + golden  <-  interleaved A/B  <-+
```

| tool | what it answers |
| --- | --- |
| `tools/matrix.py` | what we ship: every configuration x context x M vs Triton. `--nseg/--rg/--minb/--nw/--dspl` force a knob across the run, `--no-triton` halves it |
| `tools/tune.py` | coordinate descent over the knobs for one configuration, scored by geomean over the seven contexts; prints `_TUNED` rows |
| `tools/sweep.py` | one configuration, knobs x contexts |
| `tools/floor.py` | the ceiling per configuration (§1) |
| `tools/roofline.py` | S=128 gate |
| `tools/check.py` | correctness, with `--repeat` and `--mutate` |
| `tools/dump_asm.sh` | ISA for one variant |

All share `shapeset.py`, so `--hq/--hkv/--head-dim/--gqa/--filter` select the
same rows everywhere, and `--dtype fp16|bf16` the element type (default fp16). The full matrix with Triton takes about an hour.

### Measurement protocol

- **Interleave A/B.** A and B alternate within one process, several rounds,
  median per side. The session's scratch tooling did this (`/tmp/gqa/abs.py`,
  outside the repo and not guaranteed to survive); anything equivalent works.
  A/A noise at S=128 is +-0.2 %. Two matrices taken at different times are
  **not** an A/B (§5.5).
- **Read `spread`.** It is `do_bench`'s own dispersion inside the cell: median
  ~1.7 %, p90 ~5 %, and up to 20-30 % at S=32768 on some shapes.
- **No warm-up calls.** Variants are precompiled and the loader is sealed:
  `load()` raises rather than building inside the timed region.
- **Validate at long context too.** Every loss found on 2026-09-25 hid at
  S >= 8192 behind a gain at S=128 (§5.2, §5.3).

`amd-gpu-lock` only polls for other processes, so two jobs of the same session
that poll together both start and contaminate each other. Serialise your own
jobs with `flock` around it.

---

## 3. The roofline

`shapeset.roofline_us()` counts one dispatch (1.48 us under a HIP graph) plus
Q in, KV in, output out and the block table. It deliberately excludes the
split-KV partials and counts one dispatch, not NSEG. An empty kernel measures
1.74 us on the same harness, so no kernel reaches 100 %, and at S=128 the
floor alone is 65-85 % of roof depending on bytes (`floor.py`).

---

## 4. What to do next, in order

### 4.1 Keep the record current

OPTIMIZATIONS 010-023 describe the state above, and `golden/` (fp16, bf16,
sliding window) was measured on it.

### 4.2 Re-tune what is left

The 29 formerly two-mode rows were re-tuned over the full range (021); ten of
them kept their earlier values because the tuned point lost a cell.  The rest
of `_TUNED` predates RSPL, PF and DOT.  ~25 minutes per configuration on five
contexts; land a row only if a `matrix.py` run beats golden/ on it (the
tuner's single sample picks within noise), and confirm it in bf16 -- the dot
rows did not hold there (`_TUNED_BF16`).

### 4.3 The KV layout is decided: HND

The backend requires HND (`get_required_kv_cache_layout`, 026), so vLLM
allocates it for this backend whatever `VLLM_KV_CACHE_LAYOUT` says.  The
Triton paths it keeps are faster on HND too (prefill 3-5 %, batched decode
13-27 %).  NHD is still accepted by the kernel (a tensor handed in by other
code), untuned: under NHD with shuffled pages it was 4-8 % slower than the
pre-commit kernel at S >= 16384 on some D=256 configurations, never bisected.

### 4.4 The ISA work that is still open

Measured but not landed, in rough order of expected value:

- WMMA operand bank conflicts: the compiler puts A, B and C in bank 0, 34
  cycles per WMMA instead of 32.
- At D=512 M=4 (16/1/512, S=128): ~600 ns from KV landing to the first Q@K
  (the DSPL exchange's two barriers wait on the slowest wave), ~500-700 ns
  writing the partials, ~700 ns in the merge.
- The PPACK fold at the loop exit (64 permlane16), the loop-exit waitcnt
  chain.

The loop is **not** VALU-bound: lazy rescaling removed 64 multiplies per tile
and measured neutral even at S=16384 on M=4.

### 4.5 The long cells still under 90 % of roof

Eleven cells, 85.0-89.8 %, all but one at M=4 with one or two kv heads and
most of them exactly 16 MiB of KV -- where even a pure stream reaches only
92.2 % of roof (019): `32/2/128` M=4 16k/32k, `8/1/256` M=4 16k, `16/2/64`
M=4 16k/32k, `14/2/64` M=4 16k/32k, `16/1/512` M=4 16k, `16/2/256` M=4 16k,
`16/2/128` M=4 16k and M=1 16k.  What is known about them:

- not P@V (dropping P's low half is neutral), not memory parallelism (more
  segments or waves make loads-only worse);
- on the RSPL path, loads alone reach 88.4 % at 16k: the per-tile barrier of
  the shared tile, not bytes in flight (a second tile in flight is neutral);
- the fixed tail: wave skew at the loop end (0.4-1.3 us), the split-KV
  publish, atomic and merge (~1.5-2 us) -- 2-4 % of an 80 us call.

029 settles both halves: the 16 MiB step is in a pure stream whatever the
working set, and on `32/2/128` M=4 loads alone reach it while each of Q@K
and P@V costs ~5 % -- exposed WMMA latency, one tile in flight, with PF
spilling at those row counts.
030 tried to hide them: producer/consumer waves feeding an LDS ring
(loads were never the limit: 1.00-1.04x long, 0.74-0.86x short) and two
Q@K accumulators (no gain on the matrix's contiguous pages).
031 did: VINLDS stages V in the wave's LDS so the next tile's loads go out
before this tile's compute -- 1.01-1.035x on eight rows, and 32/2/128 M=4
32k, 16/2/64 M=4 16k and 16/2/128 M=4 16k now over 90 %.  The rows it has
not reached are being re-tuned with it in the search space.

The dot-against-WMMA comparison this section used to propose is done (018): dot wins short contexts at D=256/512 M=1 and loses long ones, so over the full range it rarely wins.

### 4.6 D=96, and the batch axis

D=96 is three elements per lane, still Triton.  Batches of equal query
length up to 8 run the kernel (027), 1.02-2.9x Triton except 32/8/128 at
64 x 1k (0.96x).  Open: the batch runs the single-sequence row -- a row
tuned per batch size (B is a shape, allowed to select knobs) is the next
step (not RG: 027).  Mixed batches split at D >= 256 (028): decodes on
the kernel, prefills on Triton, 1.07-4.6x; below D=256, or with few
decodes beside a short extend, the batch stays whole on Triton.

### 4.7 The fixed cost of split KV at short context

What is left at short context, and all of what is left on the small windows
(8/1 and 8/2 at w512: 64-76 % at M=1, 48-64 % at M=4, ceilings 81-85 %), is
the split-KV tail: publishing the partials, seeing the last arrival, merging
-- ~2 us of a 5 us kernel on 8/1/256 M=4 w512 (023).  An ablation that
skips the partials is 1.26x, but it is their round trip that costs, not
their bytes: 16-bit partials are no faster and 25-120x less accurate (024).
More segments, balanced segments, RG instead of RSPL, polling the arrival
counter, lighter fences and 16-wave workgroups were measured and lose
(023, 024).  The write pattern of the partials did matter: CPUB (025).
What is left would remove a round trip, not shrink one.

---

## 5. Traps that cost us time

### 5.1 The KV layout was an environment variable

Before 026 the backend took the layout vLLM chose, and without
`VLLM_KV_CACHE_LAYOUT=HND` a matrix measured NHD, 10-28 % slower for both our
kernel and Triton.  One full session of "regressions" was a matrix taken
without it compared against one taken with it.  The backend now forces HND,
but the Triton column of `matrix.py` still follows the variable: keep
`VLLM_KV_CACHE_LAYOUT=HND` so both columns see the same cache.

### 5.2 RG > 1 shares its KV through L2, and that sharing is fragile

The row groups of one kv head read the same KV and rely on L2 to read it once.
Whether they do depends on how the two workgroups drift, which nothing
controls. Replacing `__syncthreads` with LDS-only barriers broke it for
`16/2/512` M=1: `GL2C_EA_RDREQ_DRAM` at S=32768 showed the KV read 1.6 times,
-23 %. Six other RG>1 configurations stayed at 1.00-1.04x. The counter to
check is `GL2C_EA_RDREQ_DRAM` (times 128 B, against the KV bytes).

### 5.3 The compiler cannot see a wait in inline asm

The waitcnt pass ignores `s_waitcnt` written in `asm`. `__syncthreads()` gave
it a visible `vmcnt(0)` before the loop, and removing it changed codegen
across the whole kernel, not just at the barrier. Use
`__builtin_amdgcn_s_waitcnt` when the compiler needs to know. Related: a
uniform page-table address becomes an `s_load`, counted in `lgkmcnt` together
with LDS, so an `lgkmcnt(0)` barrier also waits for it.

### 5.4 HIP's occupancy query is wrong for LDS on gfx1151

`hipOccupancyMaxActiveBlocksPerMultiprocessor` assumes 64 KiB of LDS per WGP;
gfx1151 has 128 (`floor(128 KiB / LDS)` workgroups fit, measured). The shared
merge counts residency itself for this reason. Anything that waits across
workgroups must be sure the whole grid is resident.

### 5.5 Order and harness change absolute numbers

The same shape measured alone came out 1117 us, and 1404 us inside the full
NHD matrix; starting a run at a large context without walking up from S=128
measures the allocator, not the kernel (up to 10x on the first cell). `matrix.py`
goes through `benchmarks/attention_benchmarks`, whose block table is
`arange`: **contiguous** pages (earlier revisions of this file said
shuffled).  Shuffled pages cost the loads-only kernel 2-4 % at 16 MiB, so a
dev harness must use contiguous pages to agree with the matrix. Compare only runs made the same way, and
prefer interleaved A/B for any decision.

### 5.6 Profiling on a hot cache

The harness rotates a >= 96 MiB working set so every layer's KV is cold. A
driver that reuses one buffer measures a different kernel.

### 5.7 Compiling immediately before measuring

Saturates the cores and moves the SoC clock. The tools settle for 5 s.

### 5.8 Parallel builds

Bounded by memory (~1.15 GB per build). `load()` stages a private copy of the
source per variant because torch's hipify is keyed on the absolute path.
Stale lock files under `~/.cache/torch_extensions` block builds; delete them.

### 5.9 `.gitignore` eats new files

`*.csv` and `*_hip*` both match things we add.

### 5.10 `max_abs` is not a correctness criterion

Use `max_rel <= 1e-3` (8e-3 in bf16, §1) with a short S, the partial tile, repeated launches and
the `--mutate 1` negative control. The control is vacuous at M=1; validate
masking at M=4. A test for a new path must fail when that path is broken on
purpose, or it is not reaching it.

### 5.11 A define named `BF16` breaks every build

`ATen/Context.h` declares `enum class Float32Precision { ..., BF16 }`, so
`-DBF16=...` fails the torch build of every variant, fp16 included. The
kernel's dtype define is `KV_BF16`; pick names torch does not use.  `VL`
broke the same way (a hipsolver prototype has a parameter named `VL`), which
is why V-in-LDS is `VINLDS`: short all-caps knob names are risky.

### 5.12 Killing a tuner leaves build locks

A `tune.py` stopped mid-build leaves `lock` files under
`~/.cache/torch_extensions`, and the next run of the same variants waits on
them forever (GPU at 0 %, no output).  Delete the locks when no `ninja`
runs.

### 5.13 The waitcnt pass needs to count every load

A load under a branch, or a loop with an exit in the middle, and the pass
can no longer count what is outstanding: it waits `vmcnt(0)`, which drains
whatever prefetch was meant to stay in flight (015, 019).  Loads meant to
overlap must be unconditional (clamp the address instead).

---

## 6. Refuted on the WMMA kernel — do not re-derive

| idea | verdict |
| --- | --- |
| Two tiles in flight per wave (double buffer, prefetch) | 60-100 VGPRs more than a wave has; spilled, 59 % against 77 % of roof |
| LDS float atomics in the merge | ~17 us per call; never |
| Lazy rescale (only when the max grows by 2^8) | neutral; the loop is not VALU-bound |
| One select for fully masked rows | neutral to -2.7 % |
| Unrolling the split-KV merge loop | neutral; it is bandwidth-bound per CU, hence §1's shared merge |
| Segment-fastest grid order | -2 to -3 % on 16/2/512 |
| `__syncthreads` back at the Q barrier | fixes §5.2 by accident and costs 3-8 % at S=128 elsewhere |
| K issued before V | neutral |
| Forcing the page load to VMEM, a visible vmcnt(0) before the loop | removes the symptoms of §5.3, not the §5.2 regression |
| RG = GQA, V row duplication, arrive-first merge, Q first | measured neutral or worse during the rewrite |

The dot kernel's refutations (occupancy, bank conflicts, DPP butterflies, a
separate reduce kernel, non-temporal loads...) are in OPTIMIZATIONS 001-008.
The one that carried over: **instruction count does not predict time here**.

---

## 7. Reproducing

### Environment

There is **no `.venv` in the worktree**; everything runs from the main tree's.

| | |
| --- | --- |
| venv | `/scratch/rogarcia/vllm/.venv` |
| torch | 2.12.0+rocm10.1.0a20260803 |
| compiled `.so` | `/scratch/rogarcia/vllm/vllm/*.so`, symlinked into the worktree |

`PYTHONPATH=$PWD` makes `import vllm` resolve to the worktree. Scripts run as
files need it; `python -m pytest` from the worktree root does not, because
`-m` puts the working directory on the path. Verify:

```bash
PYTHONPATH=$PWD python -c "import vllm; print(vllm.__file__)"
```

A fresh worktree needs the `.so` files linked:

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

# what we ship
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py

# the same in bf16 (Triton column in bf16 too)
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py \
    --dtype bf16

# bf16 correctness, with the negative control
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/check.py \
    --dtype bf16 --hq 16 --hkv 1 --head-dim 512 --m 4 --repeat 2
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/check.py \
    --dtype bf16 --hq 16 --hkv 1 --head-dim 512 --m 4 --mutate 1

# one configuration
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py \
    --hq 16 --hkv 2 --head-dim 512 --m 1

# tune one configuration
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/tune.py \
    --hq 32 --hkv 8 --head-dim 128 --m 1

# the ceiling
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/floor.py

# the sliding-window configurations: matrix, tuning, ceiling (golden/swa.md)
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py --windowed
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/tune.py --windowed --hq 8 --hkv 1 --head-dim 256
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/floor.py --windowed

amd-gpu-lock python -m pytest tests/kernels/attention/test_rdna35_hip_decode.py
```

### Profiling

`rocprofv3` is in the venv and works with torch loaded. Filter with
`--kernel-include-regex decode_attn`, one counter group per pass.
`GL2C_EA_RDREQ_DRAM`, `GL2C_HIT` and `GL2C_MISS` settle whether KV is read
once (§5.2). `FETCH_SIZE` is request volume including hits, not DRAM traffic.

`TIMING=3` builds record per-workgroup timestamps (Q in LDS, KV landed, first
Q@K, loop end, merge, output, arrival, final merge); that is how the split-KV
merge cost was found.
