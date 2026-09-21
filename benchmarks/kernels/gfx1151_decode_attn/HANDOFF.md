# Handoff — gfx1151 decode-attention work

Written for whoever picks this up next, human or agent. Read this before the
reports: it tells you what is done, what is not, and which of the remaining
items is actually worth doing.

Everything here was measured on a Radeon 8060S (gfx1151). Nothing is estimated.

---

## 1. Where things stand

| Artefact | Where | State |
|---|---|---|
| **HND layout default** | PR **ROCm/vllm#1315**, branch `rogarcia.gfx1151-hnd-default` | **draft**, blocked on one item |
| **HIP decode kernel** | branch `rogarcia.gfx1151-hip-decode-kernel` (this tree) | published, **not integrated** |
| Triton NSEG tuning | local worktree only, **not pushed** | see note below |

The two pushed branches are independent. The layout change touches one file in
vLLM; the kernel is standalone code under `benchmarks/`.

**On the Triton NSEG tuning**: it lives only in a local worktree
(`rogarcia.fleet-f1-triton`) and is **not on the remote**. It is worth ~7 % on
the Triton kernel for `Hq=32/Hkv=16/D=256` but the commit mixes the code change
with measurement JSONs and a `VLLM_F1_NSEG` debug hook. Before pushing, split
the `(16, 256)` table entries and the `num_stages=1` pick from the
instrumentation. Note the rule is *not* "NSEG=5" — it is **`q_blocks × H_kv ×
NSEG` a multiple of 20 (the WGP count)**; 5 only applies when the base grid is
16, and it is **worse** than the default on other shapes (+13 % and +19.5 % on
two of them). Also, `reduce_segments` required NSEG to be a power of two
(`tl.arange`), so NSEG=5 did not compile until that was fixed on the branch.

### Numbers to quote

`Hq=32, Hkv=16, D=256, M=4`, fp16, KV working set past the 32 MiB MALL.
Roofline = KV bytes / 230 GiB/s.

| S | Triton NHD | HIP+HND | speedup | % roofline |
|---|---|---|---|---|
| 128 | 23.80 µs | 14.44 µs | 1.65× | 36.9 → 58.8 |
| 2048 | 199.15 µs | 153.16 µs | 1.30× | 68.2 → 88.7 |
| 32768 | 2838.81 µs | 2308.08 µs | 1.23× | 76.6 → 94.2 |

HND alone on the Triton kernel: **1.065–1.073×**, flat from 1k to 32k.
D=128 (`Hkv=8`): **40.70 vs 54.95 µs = 1.35×**.

---

## 2. The one thing blocking the PR

**`kv_cache_dtype=fp8` is untested with HND.** It changes `padded_hs`, therefore
the strides, therefore the whole premise of the layout change. Every number
above is fp16/bf16.

This is the only item standing between PR #1315 and review. It was explicitly
excluded from the original scope, not overlooked.

**How to close it:** run the context sweep from `reports/f9-context-sweep.md`
with `--kv-cache-dtype fp8`, both layouts. If HND still wins, drop the draft
flag. If it loses, gate the default on `kv_cache_dtype != fp8` and say so in the
PR.

---

## 3. What to do next, in order

### 3.1 Integrate the kernel (the big one)

The kernel is a standalone binary with `main()`. To make it reachable from
`vllm serve`:

1. **Template the compile-time constants.** `HEAD_DIM`, `NUM_Q_HEADS`,
   `NUM_KV_HEADS`, `MAXM`, `NSEG`, `BLOCK` are all `#define`. `v6` already
   generalises `D ∈ {64,128,256}` by splitting the wave (`LPR = D/8` lanes per
   row, `SUB = WAVE/LPR` groups) — that pattern is the one to template on.
2. **Register a Torch op** in `csrc/rocm/torch_bindings.cpp`. Follow `wvSplitK`
   (same file, line 29): it is the closest existing shape, also a skinny
   decode-time kernel with compile-time batch instances.
3. **Write the `AttentionBackend`** with a fallback to Triton whenever the shape
   does not fit: `D ∉ {64,128,256}`, `H_kv < 8`, prefill, fp8.

Estimate: this is the bulk of the remaining work and touches ~1000 lines across
four files. Do **not** start it before §2 is closed — if HND loses under fp8 the
backend needs a different default and you would rewrite the gating.

### 3.2 Port v6 to the paged layout

`decode_attn_v6.hip` (the `D ∈ {64,128,256}` generalisation) is on the **dense**
layout. `decode_attn_paged.hip` is on vLLM's real paged layout but is v5-based,
so D=256 only. Merging the two is mechanical but has not been done, and the
paged one is what vLLM would use.

### 3.3 Row-pitch guard

Measured, specified, **not implemented**. See `reports/f8-stride-d128.md`.

```
if row_pitch % 2048 == 0 and num_kv_heads >= 16:
    pad by 32 bytes
```

- Row pitch is `2 * head_size * elem_size` — the extent of the last dim, **not**
  `stride_head`. Under HND `stride_head` is always a multiple of 2048 for any
  `hs >= 64`, and HND is *faster*, which is how we know the predictor is the
  pitch.
- **The padding must be a multiple of 32 B.** Pads of 8, 16 and 48 B measure
  **47 % worse than no padding at all**. A guard that "just nudges the stride"
  with naive arithmetic can land on 48 B and make things much worse.
- +32 B captures 82 % of the win for 1.6 % memory; +256 B costs 12.5 %.
- Real config affected: **Gemma4** (`transformers_utils/configs/gemma4.py:27`,
  `global_head_dim=512` → pitch exactly 2048). DeepSeek D=512 goes through MLA,
  different backend.
- **The real work is not the condition.** `kv_cache.split(hs, dim=-1)`
  (`triton_attn.py:687`) assumes the last dim is exactly `2*hs`; it needs the
  same view-slicing the per-token-head quantisation branch already does.

### 3.4 VOPD by banks, in the real kernel

`reports/f7-vopd.md` has a verified recipe that takes `P@V` pairing from 0 % to
100 %. It has **not** been applied to the shipped kernel — the current one
already emits 74 `v_dual` without it, so the marginal gain is unknown.

Worth ~1.57× on `P@V` VALU, which **will not move wall time** (the kernel is
18–90× memory-bound). It matters only if you need the registers: 83 vs 153 VGPR.

---

## 4. Traps that cost us time

### 4.1 The editable install points at a stale tree

`import vllm` resolves to `/scratch/rogarcia/vllm-build/vllm`, **655 commits
behind and without the 3D decode path for M>1**. Measuring without
`PYTHONPATH` set to your own worktree silently benchmarks the wrong kernel —
281 µs instead of 199 µs, no error, no warning.

Two agents hit this independently. Always verify:

```bash
PYTHONPATH=<your-worktree> python -c "import vllm; print(vllm.__file__)"
```

### 4.2 `*.hip` is in `.gitignore`

Line 229, meant for PyTorch-generated artefacts. Our kernel sources match it.
The first commit of this branch silently dropped **all eight kernels** and kept
only the reports. Use `git add -f` for `.hip` sources and check
`git ls-files | wc -l` against what is on disk.

### 4.3 `max_abs` is not a sufficient correctness criterion

A causal-mask off-by-one of **a single key** gives `max_abs = 1.2e-03` at
S=2048 and **passes** a 2e-2 threshold. The error scales as ~1/S while the
tolerance is fixed, so **longer contexts hide more bugs**.

Validate with all three: `max_rel <= 1e-3`, a short-`S` case (S≈48), and
negative controls. `decode_attn_paged_mutants.hip` carries three — but
`MUTATE=2` only patches the `LAYOUT==0` branch, so under `-DLAYOUT=1` it is a
no-op and **passes silently**. Run it with `-DLAYOUT=0`.

### 4.4 Cache residency inflates short-context results

Without forcing the KV working set past the 32 MiB MALL, S=128 reads **33.8 %
fast** and the kernel appears to exceed 100 % of roofline — the tell-tale sign.
The knee is exactly at 32 MiB. Use `--min-working-set-mb 96` for the vLLM
benchmark, or `ncopy` in the HIP harness.

### 4.5 Others, briefly

- **HIP resolves device functions by name process-globally.** A harness that
  `dlopen`s several variants of one source runs the *first* registered kernel
  with the *current* arguments. Gave one agent a bogus 214 % of roofline.
  Derive kernel names from the compile parameters.
- **`rocprofv3` on gfx1151** cannot arm `FETCH_SIZE` and `GL2C_*` in one pass
  (error 38), and fails if torch already initialised the GPU.
- **Per-lane addressing is mandatory** in microbenchmarks, or the compiler
  scalarises everything into `s_load_b256` and register counts are meaningless.
- `num_compute_units()` returns **WGPs (20), not CUs (40)**.
- Sustained clock under load is **~2100 MHz**, not the 2900 MHz boost. Time with
  the realtime clock; `SHADER_CYCLES` stops during L2+ stalls and wraps at 2²⁰.

---

## 5. Things that look promising and are not

Measured and refuted. Do not spend time re-deriving these.

| Idea | Verdict |
|---|---|
| Non-temporal K/V loads | **1.84× worse** — destroys the GQA cache reuse |
| High `NSEG` / split-KV | `NSEG=1` wins; 16 costs 700 µs, and the penalty **grows** with S |
| Q staged in LDS | unnecessary — the winning kernel uses **zero** LDS |
| Larger tiles / blocks | +52 % and +128 % |
| `global_load_lds` | **does not exist on gfx1151** (CDNA-only) |
| General stride padding | does not transfer to the paged layout; HND beats it for free |
| bf16 being 30–40 % slower | real impact **1.5 %** in a memory-bound regime |

They share one cause, and it is the single most useful thing to internalise:

> With GQA=2 two q-head blocks read the same KV head, so 64 MiB is issued for
> 32 MiB unique and **the cache recovers half the traffic**. Anything that
> breaks that reuse costs more than it gives.
>
> **In this regime the scarce resource is cache locality, not grid
> parallelism.**

The original design document optimised for the opposite and was wrong on seven
of ten hypotheses. `reports/00-design.md` §0.1 has the full verdict table with
the measurements that overturned each one.

---

## 6. What is left on the table

From 94.2 % of roofline there are **14.3 µs** to the theoretical ceiling at
S=2048 (150.20 → 135.87 µs).

It is **not bytes**: `FETCH_SIZE` sits at 1.03× of ideal, so the kernel already
reads the KV essentially once. What remains is effective bandwidth and latency.
Nobody has attacked this, and it is the least-charted territory here — the
obvious levers are all spent.
