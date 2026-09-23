# Handoff — gfx1151 decode attention

Written for whoever picks this up next, human or agent. Read this before the
reports: it says where things stand, what is still open, and which of the open
items is actually worth doing.

Everything here was measured on a Radeon 8060S (gfx1151), batch = 1 sequence,
fp16, HND. Nothing is estimated.

---

## 1. Where things stand

The kernel is **integrated and selected by a backend**, not a standalone
experiment. `RDNA35_HIP_ATTN` serves 49 of the 50 shapes in
`tools/shapes.csv`; the one it refuses is D=96, which is three fp16 per lane
and has no single load width.

One kernel, `csrc/rocm/rdna35_decode_attn.cu`. The small-grid fork that used to
live beside it is gone — its decomposition is the `MSPLIT` knob now.

### The honest performance picture

**We win at short context and lose at long.** Across 27 distinct
configurations and 7 contexts, geomean **0.933x** against the Triton kernel
vLLM ships:

| S | geomean | configurations we lose |
| --- | --- | --- |
| 128 | 1.605x | 0 / 27 |
| 512 | 1.268x | 8 / 27 |
| 1024 | 1.053x | 13 / 27 |
| 4096 | 0.791x | 15 / 27 |
| 8192 | 0.735x | 15 / 27 |
| 16384 | 0.713x | 15 / 27 |
| 32768 | 0.691x | 15 / 27 |

Full table in `reports/f12-matrix.md`, per-configuration geomeans below.

Be careful with `reports/f11-shape-coverage.md`: it reports winning every
shape, which is true **only at S=128**, the one context `roofline.py` measures.
That is the mistake to avoid repeating — a gate that measures one context will
tell you you are winning while you lose everywhere else.

### Which configurations lose

Worst first, geomean over the 7 contexts. `+n` is how many more models share
the configuration.

| geomean | Hq | Hkv | D | models |
| --- | --- | --- | --- | --- |
| 0.36x | 16 | 2 | 64 | MiniCPM-V-0.53B +1 |
| 0.36x | 14 | 2 | 64 | Qwen2.5-0.5B |
| 0.40x | 28 | 4 | 128 | Qwen2.5-7B **+4** |
| 0.40x | 32 | 8 | 64 | Llama-3.2-1B |
| 0.42x | 32 | 4 | 128 | Qwen3-30B-A3B +1 |
| 0.55x | 24 | 8 | 128 | Llama-3.2-3B +1 |
| 0.57x | 40 | 8 | 128 | Nemotron-3-Nano |
| 0.57x | 32 | 8 | 128 | Qwen3-4B **+7** |
| 0.61x | 32 | 2 | 128 | MiniCPM-V-custom |
| 0.72x | 24 | 4 | 256 | Qwen3.6-27B |
| 0.77x | 16 | 2 | 128 | Qwen2.5-3B +2 |
| 0.95x | 32 | 32 | 64 | SmolLM2-1.7B |
| 0.98x | 16 | 2 | 256 | Qwen3.5-35B +1 |

Everything at D=512 wins by 1.95x-2.96x. D=256 wins except the two above.
The losses are concentrated in D=64 and D=128 with small Hkv.

---

## 2. The loop

```text
roofline.py  ->  worst shape  ->  ISA / measured wiki / first principles
     ^                                          |
     +--------------  commit  <-  validate  <---+
```

| tool | what it answers | cost |
| --- | --- | --- |
| `tools/roofline.py` | where to look next; also the regression gate | minutes |
| `tools/sweep.py` | one configuration, a matrix of knobs x contexts | minutes |
| `tools/check.py` | correctness, with `--repeat` and `--mutate` | seconds |
| `tools/matrix.py` | what we ship: every configuration x context vs Triton | ~19 min |
| `tools/tune.py` | searches (NSEG, MSPLIT) scoring the **worst** context | long |
| `tools/dump_asm.sh` | ISA for one variant | seconds |

All of them build variants in parallel. A cold build is ~19.6 s of which the
kernel is 0.57 s -- the rest is `torch/extension.h` -- so compilation, not
measurement, was the cost of the loop until that was fixed.

`roofline.py`'s `ran` column is not optional reading. The backend falls back to
Triton silently, and a harness that does not check reports Triton as ours.

---

## 3. The knobs

| knob | values | chosen by | tuned? |
| --- | --- | --- | --- |
| `NSEG` | 1..32 | `_segments_for`, targets 32 workgroups | **badly** |
| `MSPLIT` | divisors of MAXM and NWAVE | `_MSPLIT_KV_HEADS`, a 4-point table | partly |
| `BLOCK` | 128, 256, 512 | fixed 256 | **no** |
| `KPW` | divisors of BS | fixed 4 | no |
| `ILV` | 0, 1 | fixed 1 | measured, 1 wins |
| `FUSEDRED` | 0, 1 | 1 | measured, 1 wins |

`_TARGET_WORKGROUPS = 32` **is the worst of four values at every head size
measured** and is the single largest known loss. For Hq=32 it picks NSEG=1 and
costs between 1.6x and 3.7x.

---

## 4. What to do next, in order

### 4.1 Retune NSEG and MSPLIT per configuration

This is the work in progress and the biggest available win. Done one
configuration at a time, not as a global sweep.

`Hq=32/Hkv=8/D=128` (8 models) is finished and is the template: tuning takes
the worst context from **0.41x to 0.89x**, but **no combination of all five
knobs wins everywhere**. At S=32768 we reach 82.6% of roofline and Triton
reaches 93.0%.

So expect retuning to convert most losses into near-parity, not into wins.
That is still worth roughly 2x on the shapes concerned.

### 4.2 Find out why Triton beats us at long context

After 4.1 this is what is left, and it is not a tuning problem. On
`Hq=32/Hkv=8/D=128` at S=32768 Triton extracts 93.0% of the KV roofline and
our best configuration extracts 82.6%. Nobody has looked at what it does
differently. This is the least-charted territory here.

### 4.3 Support D=96

One shape, Phi-3.5-vision. 96/32 is three fp16 per lane, not a power of two,
so it needs either a non-power-of-two lane group or a padded load. Deliberately
out of scope so far.

### 4.4 The batch axis

The kernel serves **one sequence**; the backend rejects batch > 1 to Triton.
Every number in this document is batch = 1. In real serving decode is batched,
so today we do not participate. Note that B and NSEG are substitutes -- both
exist to fill the machine -- so introducing B pushes NSEG towards 1 and makes
most of the split-KV machinery inert. It is a different regime, not an
extension of this one.

---

## 5. Traps that cost us time

### 5.1 Measuring one context and believing it

`roofline.py` measures S=128. It is the right choice for finding fixed cost and
the wrong one for judging whether we ship. See §1.

### 5.2 The harness times the JIT build

The kernel compiles on its first `_prepare`, which lands inside
`do_bench`'s own calibration -- and `do_bench` sizes `n_repeat` from it, so a
build measuring seconds does not just add time, it makes the whole median
wrong. Seen as 66.90 us next to a true 7.58. Every tool now does a discarded
run first.

### 5.3 An override the backend would not have chosen rebuilds every forward

`self._variant` then disagrees with what `_prepare` computes forever, and
`make_scratch` zeroes the arrival counters, which is a device memset -- an
extra kernel launch inside the timed region. It charged ~2 us to every
overridden configuration, including ones where the override changed no
generated code. Scratch is memoised in every tool for this reason.

### 5.4 Compiling immediately before measuring

Saturates the cores and moves the SoC clock. A pass taken straight after a
50-way rebuild reported three shapes falling back that are served on every
quiet run. The tools settle before timing. This is what `quiet-lock` in
`reports/00-protocol.md` exists for.

### 5.5 Parallel builds are bounded by memory, not cores

A build peaks at 1.15 GB. One per core wanted more than twice this machine's
30 GB and had to be killed.

### 5.6 `.gitignore` eats new files

`*.csv` and `*_hip*` both match things we add. `shapes.csv` needed
`git add -f`. An earlier commit silently dropped eight kernels this way.

### 5.7 An identifier the preprocessor has not seen is 0 inside `#if`

The `MSPLIT` fallback rule read `NWAVE` before it was defined, so every branch
compared `0 <= 65536` and it always chose 1. It had never worked; D=512 was the
first shape where the fallback mattered.

### 5.8 `max_abs` is not a sufficient correctness criterion

A causal off-by-one of a single key gives `max_rel = 1.2e-03` at S=2048 and
passes a 2e-2 threshold. The error scales as ~1/S while the tolerance is fixed,
so **longer contexts hide more bugs**. Validate with `max_rel <= 1e-3`, a short
S, the `S=50` partial tile, `--repeat` for state left between launches, and the
`--mutate 1` negative control.

### 5.9 Harness flakiness

Triton at S=512 sporadically faults or spikes. One cold-build pytest run failed
the two layout=0 reference tests and has never reproduced. Treat anything under
~2% as noise -- that floor is measured, from 161 repeated cells in the matrix
run: 0.2% median, 1.4% p90, 3.3% worst.

---

## 6. Refuted — do not re-derive

| idea | verdict |
| --- | --- |
| WGP alignment as a criterion | refuted three times; NSEG=5 is exactly 40 WGs and loses to 6 and 7 |
| Occupancy limits this kernel | refuted from three axes; 33 KiB of LDS and half the waves is faster |
| DPP for the score butterfly | +1.5% at S=128, **-34%** at 32k; the ISA forbids pairing DPP in VOPD |
| Contiguous KV runs (`ILV=0`) | 658 -> 1032 us at 32k; locality beats sequentiality |
| Prefetching the block table | +2.2% at 1024, -1.9% at 32k, net nothing |
| Unsigned loop induction variable | fewer instructions, **13% slower** |
| Instruction count predicts time | 67 instructions removed bought 1.5%; 16 bought 5% |
| MSPLIT gain follows bytes/token | not monotonic in Hkv: +3.8 / +7.5 / +11.1 / +3.6% |
| Non-temporal K/V loads | 1.84x worse; destroys the GQA reuse |
| `global_load_lds` | does not exist on gfx1151 |

The one sentence worth internalising is still the design report's:

> In this regime the scarce resource is cache locality, not grid parallelism.

---

## 7. Reproducing

```bash
cd <worktree>
export PATH=<venv>/bin:$PATH PYTHONPATH=$PWD VLLM_KV_CACHE_LAYOUT=HND

# the gate, and where to look next
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/roofline.py

# one configuration, a matrix of knobs against contexts
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/sweep.py \
    --hq 32 --hkv 8 --head-dim 128 --nseg 2 4 --msplit 2 4 --triton

# correctness: partial tile, repeated launches, negative control
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/check.py \
    --hq 32 --hkv 8 --head-dim 128 --repeat 10 --contexts 48 50 1020 1024
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/check.py \
    --hq 32 --hkv 8 --head-dim 128 --mutate 1 --contexts 48 --layouts 1

amd-gpu-lock python -m pytest tests/kernels/attention/test_rdna35_hip_decode.py
```
