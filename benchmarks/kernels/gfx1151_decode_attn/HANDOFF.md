# Handoff — gfx1151 decode attention

Written for whoever picks this up next, human or agent. Read this before the
reports: it says where things stand, what is still open, and which of the open
items is actually worth doing.

Everything here was measured on a Radeon 8060S (gfx1151), batch = 1 sequence,
fp16, **HND** (see §5.1). Nothing is estimated.

| file | what it is |
| --- | --- |
| this one | state, open work, traps |
| `OPTIMIZATIONS.md` | one entry per optimisation landed **or rejected**, with the numbers. 001-008 describe the previous (dot) kernel; 009 is the rewrite, 010-012 the commits below |
| `reference/` | the previous per-q-head dot kernel and its D=512 golden, kept for comparison only |
| `golden/` | best measured result per head size for the WMMA kernel, with each configuration's ceiling; replace only when beaten |
| `reports/` | the original investigation record, about the dot kernel. Its `%roof` numbers are superseded |

---

## 1. Where things stand

`RDNA35_HIP_ATTN` serves 49 of the 50 shapes in `tools/shapes.csv` with one
kernel, `csrc/rocm/rdna35_decode_attn.cu`; it refuses D=96 (three fp16 per
lane) and falls back to Triton. The nine sliding-window rows are skipped by
every tool.

The kernel is the WMMA rewrite of OPTIMIZATIONS 009: one workgroup per
`(kv head, row group, KV segment)`, both products on
`v_wmma_f32_16x16x16_f16`, knobs `NSEG / RG / MINB / NW / DSPL` per
configuration in `_TUNED` (`vllm/v1/attention/backends/rdna35_hip_attn.py`).
Commits on top of it, 2026-09-25:

| commit | what |
| --- | --- |
| `e79bc3cb18` (010) | preamble and softmax: LDS-only barriers, unconditional Q load, permlane fetch-inactive, causal mask only on the tail tile. S=128 +1 to +10 % |
| `c17a8bda13` (011) | split-KV merge shared across the segments when a group's partials reach 64 KiB. D=512 S=128 up to +56 % |
| `16621e4856` (012) | `16/2/512` M=1 re-tuned to `rg=1` (§5.2) |

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
same rows everywhere. The full matrix with Triton takes about an hour.

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

OPTIMIZATIONS 010-012 and `golden/` describe the state above. `golden/` holds
the 2026-09-25 result; replace it after the re-tune in §4.2 if that beats it.

### 4.2 Re-tune everything

`_TUNED` is still the first tuning pass, made before the three commits. A
second pass was run on the pre-commit kernel and never integrated; it is
obsolete now. The merge change in particular moves the NSEG/MINB optimum on
D=512 and D=256. One configuration takes about 3.5 minutes on two contexts,
so all 52 are ~3 hours plus a validating matrix. Validate every new row at
all seven contexts.

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

### 4.5 Dispatch dot or WMMA by model configuration

To explore. The two kernels win in different places: the WMMA kernel reads
the KV once per kv head and wins wherever GQA x M fills its 16-row tiles
(D=512 M=4 went 59.7 -> 71.1 %roof over the dot kernel), while the dot
kernel, one workgroup per q head, still beat it on D=512 M=1 at S=128-1024
before 010-012 (geomean 82.9 against 73.3 %, `reference/golden_d512_dot.md`),
where few real rows leave most of each WMMA as padding. The shared merge
closed part of that gap and the comparison has not been redone since.

The idea is to keep both and let the backend pick per configuration -- `(Hq,
Hkv, D, M)`, fixed at graph capture like the other knobs -- rather than to
grow a dot path inside the WMMA kernel. What it needs:

- the comparison redone per configuration and context with the current
  kernel, `reference/rdna35_decode_attn_dot.cu` built as a second variant;
- the rule, or a `_TUNED`-style table, saying which kernel serves which
  configuration -- one choice per configuration, since S is not known at
  capture;
- both kernels' variants precompiled and sealed, and the scratch shapes of
  both allocated.

The first step is the measurement: if no configuration has the dot kernel
ahead over the seven-context geomean, the idea is closed.

### 4.6 D=96, and the batch axis

Unchanged: D=96 is three fp16 per lane; batch > 1 falls back to Triton.

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
measures the allocator, not the kernel (up to 10x on the first cell). The
dev harness used contiguous pages and HND, `matrix.py` uses shuffled pages
and whatever layout vLLM picked. Compare only runs made the same way, and
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

Use `max_rel <= 1e-3` with a short S, the partial tile, repeated launches and
the `--mutate 1` negative control. The control is vacuous at M=1; validate
masking at M=4. A test for a new path must fail when that path is broken on
purpose, or it is not reaching it.

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

# one configuration
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py \
    --hq 16 --hkv 2 --head-dim 512 --m 1

# tune one configuration
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/tune.py \
    --hq 32 --hkv 8 --head-dim 128 --m 1

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
