# Handoff — gfx1151 decode attention

Written for whoever picks this up next, human or agent. Read this before the
reports: it says where things stand, what is still open, and which of the open
items is actually worth doing.

Everything here was measured on a Radeon 8060S (gfx1151), batch = 1 sequence,
fp16 unless it says bf16, **HND** (see §5.1). Nothing is estimated.

| file | what it is |
| --- | --- |
| this one | state, open work, traps |
| `OPTIMIZATIONS.md` | one entry per optimisation landed **or rejected**, with the numbers. 001-008 describe the previous (dot) kernel; 009 is the rewrite, 010-012 the commits below |
| `reference/` | the previous per-q-head dot kernel and its D=512 golden, kept for comparison only |
| `golden/` | best measured result per head size for the WMMA kernel, with each configuration's ceiling; replace only when beaten. `bf16.md` is the same for bf16 |
| `reports/` | the original investigation record, about the dot kernel. Its `%roof` numbers are superseded |

---

## 1. Where things stand

`RDNA35_HIP_ATTN` serves 49 of the 50 shapes in `tools/shapes.csv` with one
kernel, `csrc/rocm/rdna35_decode_attn.cu`, in fp16 and in bf16; it refuses
D=96 (three elements per lane) and falls back to Triton. The nine
sliding-window rows are skipped by every tool.

The kernel is the WMMA rewrite of OPTIMIZATIONS 009: one workgroup per
`(kv head, row group, KV segment)`, both products on
`v_wmma_f32_16x16x16_f16` (`_bf16` in bf16), knobs
`NSEG / RG / MINB / NW / DSPL / RSPL / PF` per configuration in `_TUNED`
(`vllm/v1/attention/backends/rdna35_hip_attn.py`).

Since 016 a build can carry **two decompositions** of its configuration and
run the second when the S it reads on the device is `>= SW`.  The
configuration -- grid, block, both knob sets, `SW` -- is still one per
`(Hq, Hkv, D, M)`, fixed at graph capture; S only picks the code path inside
the launch, as it already picked the number of active segments.  The short
mode can be the **dot decomposition** of reference/ (`DOT=1`, 018), which
wins below ~4k keys at D=256/512 M=1.  A row with `sw` in `_TUNED` is a
two-mode build; its `...2` knobs are mode B's.

Commits on top of it, 2026-09-25:

| commit | what |
| --- | --- |
| `e79bc3cb18` (010) | preamble and softmax: LDS-only barriers, unconditional Q load, permlane fetch-inactive, causal mask only on the tail tile. S=128 +1 to +10 % |
| `c17a8bda13` (011) | split-KV merge shared across the segments when a group's partials reach 64 KiB. D=512 S=128 up to +56 % |
| `16621e4856` (012) | `16/2/512` M=1 re-tuned to `rg=1` (§5.2) |
| `88fb3dc37b` (013) | bf16: a compile-time dtype, bf16 WMMAs, `--dtype` in the tools (§1, bf16) |
| `83415bcc26` (014, 015) | S read on the device (a host S was frozen into full CUDA graphs); RSPL, rows split inside the workgroup over a shared tile |
| `9ae12f67da` (016) | two decompositions per build, switched on the device by S; `tune.py --split` |
| `8cd55c5497` (017) | PF, a second tile in flight where it fits; six two-mode rows |
| `c219ce6083`, `5d1204997f` (018) | the dot decomposition as a mode; short mode of all twelve D=256/512 M=1 rows (S=128 1.12-1.34x); D=64 PF rows |

### The performance picture

`matrix.py`, HND, all 52 configuration/M pairs, geomean over the seven
contexts, measured 2026-09-25 after the three commits (`golden/`; the
16/2/512 M=1 rows come from the run that verified its re-tune):

| D | M | configs | vs Triton | median configuration %roof | >= 90 % roof | S=128 median %roof |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 1 | 4 | 1.27x | 83.7 % | 1 | 55.6 % |
| 64 | 4 | 4 | 1.29x | 83.8 % | 1 | 56.6 % |
| 128 | 1 | 10 | 1.21x | 89.0 % | 1 | 69.0 % |
| 128 | 4 | 10 | 1.27x | 87.7 % | 1 | 64.0 % |
| 256 | 1 | 7 | 1.32x | 81.4 % | 1 | 52.2 % |
| 256 | 4 | 7 | 1.52x | 79.6 % | 0 | 51.5 % |
| 512 | 1 | 5 | 2.84x | 82.2 % | 0 | 55.3 % |
| 512 | 4 | 5 | 3.91x | 77.2 % | 0 | 46.7 % |

5 of 52 pairs reach 90 % of roof (`32/32` at D=64 and D=128, both M, and
`16/8/256` M=1). Five cells of 364 are slower than Triton, all by under 2 %
(D=128 at S=32768), which is inside the harness noise.

Against the matrix taken right before these commits (same harness, HND),
S=128 is better on all 52 pairs (median +7 % at M=1, +9 % at M=4) and long
contexts are unchanged (median +0.7 %).

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

`matrix.py --dtype bf16`, all 52 pairs, Triton in bf16 too (`golden/bf16.md`):

| D | M | configs | vs Triton | median configuration %roof | fp16 (`golden/d*.md`) | >= 90 % roof | S=128 median %roof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 1 | 4 | 1.27x | 84.0 % | 83.7 % | 1 | 55.5 % |
| 64 | 4 | 4 | 1.29x | 83.5 % | 83.8 % | 1 | 56.6 % |
| 128 | 1 | 10 | 1.21x | 88.8 % | 89.0 % | 1 | 69.2 % |
| 128 | 4 | 10 | 1.28x | 88.3 % | 87.7 % | 1 | 63.8 % |
| 256 | 1 | 7 | 1.32x | 81.2 % | 81.4 % | 1 | 51.6 % |
| 256 | 4 | 7 | 1.53x | 79.6 % | 79.6 % | 0 | 51.4 % |
| 512 | 1 | 5 | 2.88x | 82.0 % | 82.2 % | 0 | 54.7 % |
| 512 | 4 | 5 | 3.87x | 77.0 % | 77.2 % | 0 | 46.5 % |

The same five pairs reach 90 % of roof; four cells of 364 trail Triton by
1 %, D=128 at S=32768, as in fp16.

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

OPTIMIZATIONS 010-019 describe the state above.  `golden/` still holds the
2026-09-25 fp16 result from before 014; regenerate it (and `golden/bf16.md`)
from a full matrix once §4.2 is done.

### 4.2 Re-tune everything with `--split`

Two-mode rows exist for D=256/512 M=1 (all twelve, dot below 4096), six
low-kv-head configurations and five at D=64 (OPTIMIZATIONS 017-018).  The rest
of `_TUNED` is still single-mode.  `tune.py --split 4096 --contexts 128 512
2048 4096 16384 32768` takes ~25 minutes per configuration; land a row only
if a `matrix.py` run beats golden/ on it (the tuner's single sample picks
within noise: three of nine rows lost 1-5 % in the first pass).  The switch
point is fixed at 4096; where the two modes cross earlier (`32/4/512`, 0.93x
at S=1024) a per-configuration `--split` would recover it.

`_TUNED` is keyed without the dtype: bf16 moves the same bytes and measured
within 1 % of fp16 everywhere, so one table serves both. Tune in fp16 and
confirm with `matrix.py --dtype bf16`; key the table on the dtype only if a
row ever disagrees.

### 4.3 Decide the KV layout, then fix what it exposes

All the numbers here are HND. vLLM's default is NHD, and under NHD with
shuffled pages the current kernel is 4-8 % slower than the pre-commit one at
S >= 16384 on some D=256 configurations. Re-tuning recovers the M=1 ones
(`8/4`, `16/4`, `16/8` at `nw=4, dspl=4, minb=2`: +4-5 points), but not
`8/4/256` and `16/8/256` at M=4, so a kernel change is responsible there. It
has not been bisected. Nothing NHD-specific has been applied.

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

After 014-018, all at M=4 with one or two kv heads, most of them exactly
16 MiB of KV -- where even a pure stream reaches only 92.2 % of roof (019):
`32/2/128` 16k (~85 %), `16/2/128` 16k, `16/1/512` 16k/32k, `14/2/64` 32k,
`8/1/256` M=1 16k (88.9 %).  What is known about them:

- not P@V (dropping P's low half is neutral), not memory parallelism (more
  segments or waves make loads-only worse);
- on the RSPL path, loads alone reach 88.4 % at 16k: the per-tile barrier of
  the shared tile, not bytes in flight (a second tile in flight is neutral);
- the fixed tail: wave skew at the loop end (0.4-1.3 us), the split-KV
  publish, atomic and merge (~1.5-2 us) -- 2-4 % of an 80 us call.

The dot-against-WMMA comparison this section used to propose is done (018).

### 4.6 D=96, and the batch axis

Unchanged: D=96 is three elements per lane; batch > 1 falls back to Triton.

---

## 5. Traps that cost us time

### 5.1 The KV layout is an environment variable

`matrix.py` goes through the backend, which reads the layout vLLM chose.
Without `VLLM_KV_CACHE_LAYOUT=HND` it measures NHD, and both our kernel and
Triton come out 10-28 % slower on many shapes. One full session of
"regressions" was a matrix taken without it compared against one taken with
it. Check the variable before comparing any two tables.

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
kernel's dtype define is `KV_BF16`; pick names torch does not use.

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

### 5.14 Two bodies in one kernel do not compile alike

The body inside the mode branch is compiled worse than the one after it
(2-4 % and 15 % measured, 016); the long mode falls through.  A new mode
body should be measured both ways.

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

# tune it as a two-mode build switching at S=4096
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/tune.py \
    --hq 32 --hkv 8 --head-dim 128 --m 1 --split 4096 \
    --contexts 128 512 2048 4096 16384 32768

# the ceiling
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/floor.py

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
