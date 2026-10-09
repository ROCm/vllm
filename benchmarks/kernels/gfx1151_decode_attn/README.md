# RDNA3.5 attention: adding, tuning and checking a configuration

On RDNA3.5 `ROCM_ATTN` serves decode with the HIP kernel in
`csrc/rocm/rdna35_decode_attn.cu`, and prefills of 128 to 8192 query tokens
with the one in `csrc/rocm/rdna35_prefill_attn.cu` (see
[Prefill](#prefill)). The kernel takes every shape and launch
knob as a compile-time define, so each build serves exactly one tuple. The
builds are the rows of `vllm/v1/attention/ops/rdna35_variants.csv`. CMake
compiles every row into `_rocm_C` (`cmake/rdna35_attn.cmake`,
`csrc/rocm/generate_rdna35_attn.py`), and the backend looks each call up in the
same rows (`vllm/v1/attention/ops/rdna35_hip_decode.py`). A call without a row
falls back to Triton.

A *configuration* is `(Hq, Hkv, D, window, page size)`. Supporting one means
adding its rows. There is one row for every M = 1..8 query tokens, in fp16 and
bf16, for one sequence and for a batch: 32 rows per page size.

## Requirements

- An RDNA3.5 GPU with a vLLM build of this tree. The tools are run from the
  tree root with `PYTHONPATH=$PWD`.
- The kernel's limits: `D` in `SUPPORTED_HEAD_SIZES` (64, 128, 256, 512),
  `Hq % Hkv == 0`, and a page size that is a multiple of 16. Anything else
  needs kernel work first.
- The page size is the one vLLM uses for the model. It is 16 unless the model
  is a hybrid (attention plus linear-attention or Mamba layers). For a hybrid,
  vLLM grows the attention page until it holds as many bytes as one
  linear-attention state, e.g. 544 for Qwen3.5-0.8B; the log of a vLLM run
  prints it.
- The tools run the benchmark harness, whose stand-in model config is read with
  `HF_HUB_OFFLINE=1`. Run once online (`HF_HUB_OFFLINE=0`) to cache it.
- Nothing else should use the GPU while the tools time the kernel.

## 1. Tune

`tools/tune.py` searches the knob grid for one `(Hq, Hkv, D, window, M)`:

- every point is built with `tools/jit.py`, a torch-free build of the kernel;
- every point is checked against a float reference and then timed at 128 to
  32k context;
- the finalists are confirmed in an interleaved A/B.

One knob set must serve every context, because a CUDA graph fixes the grid
while S varies, so points are scored over the whole range. The module
docstring describes the choice rule.

Tune each M from 1 to 8 in fp16; bf16 reuses the fp16 choice:

```bash
for m in 1 2 3 4 5 6 7 8; do
  PYTHONPATH=$PWD .venv/bin/python \
      benchmarks/kernels/gfx1151_decode_attn/tools/tune.py \
      --hq 16 --hkv 2 --head-dim 256 --m $m \
      --points points.jsonl --decisions decisions.jsonl
done
```

- Add `--window W` for sliding-window attention.
- Add `--block-size P` to measure at a page size other than 16.
- `--points` collects every measured point as it goes. After an interruption,
  rerun with `--reuse` to skip the points already measured.
- `--limit 20` is a quick smoke run.

For a configuration that already has rows, the current row is the reference.
A new knob set replaces it only if it wins under the choice rule. A new
configuration has no reference, so the best point by roofline wins.

## 2. Apply

```bash
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/gfx1151_decode_attn/tools/tune.py \
    --apply decisions.jsonl --block-sizes 16
```

This writes the configuration's rows: both dtypes, one sequence and a batch,
and every page size the configuration already has, or those given in
`--block-sizes` for a new one. A batch row whose winner is the dot-product
path gets the best WMMA knob set instead. A decision that was measured
against another kernel source, or against a row that has changed since, is
refused.

Then rebuild `_rocm_C`. CMake reruns the generator whenever the CSV changes
(see `docs/contributing/incremental_build.md`).

Add the model to `tools/shapes.csv` (`model,Hq,Hkv,D,window`). `matrix.py`
measures only the shapes listed there, and `--hq`, `--hkv` and `--head-dim`
filter that list.

## 3. Check

Tests, on an RDNA3.5 GPU:

```bash
.venv/bin/python -m pytest -q \
    tests/kernels/attention/test_rdna35_hip_decode.py \
    tests/v1/attention/test_rocm_attention_backends_selection.py
```

- `test_variants_csv_is_well_formed`: every configuration has all its rows,
  and no batch row uses the dot-product path.
- `test_every_listed_variant_is_built_in`: every row is in `_rocm_C`.
- `test_built_in_variants_match_reference`: every built row, against a float
  reference. It runs one sequence at a short context and past a partial tile,
  several launches on one scratch with new inputs each time, and a batch with
  a padded sequence. Windows run past the window, and large pages end partly
  filled.

Performance, with every M and both dtypes against the baselines:

```bash
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py \
    --hq 16 --hkv 2 --head-dim 256 --m 1 2 3 4 5 6 7 8 --dtype bf16 \
    --baselines TRITON_ATTN ROCM_SEGMENTED_ATTN@autotune > matrix.md
```

- The `path` column must read `kernel`. `triton` means the call found no row.
- `%roof` is the time against a roofline (`tools/utils.py`): the call's bytes
  at 230 GiB/s or its FLOPs at 43 TFLOPS of sustained WMMA, whichever is
  longer. `AI` is the arithmetic intensity in FLOP/byte and `bound` says which
  roof applies: memory below the ridge point (~174 FLOP/byte), compute above.
  Decode is memory-bound; prefills of a few hundred tokens or more are
  compute-bound.
- Add `--windowed` for sliding-window configurations and `--block-size P` for
  other page sizes.

When the change affects a model's output, run a model eval as well
(`tests/evals/` or `lm_eval`).

## Prefill

The prefill kernel serves one sequence of M query tokens after its cached
prefix, outside CUDA graphs, so M is known on the host. Its rows are in
`vllm/v1/attention/ops/rdna35_prefill_variants.csv`: for every configuration,
in fp16 and bf16, one row per step of M, the row whose `MIN_QUERY_LEN` is the
largest at or below M. Steps start at 128, 256, 512, 1024, 2048 and 4096;
contiguous steps that share a row are one row. Below 128 query tokens a
prefill goes to Triton.

Tune every step of a configuration in one run (fp16; bf16 reuses it):

```bash
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/gfx1151_decode_attn/tools/tune.py --prefill \
    --hq 16 --hkv 2 --head-dim 256 \
    --points points.jsonl --decisions decisions.jsonl
```

- Every point of the prefill grid is checked against Triton and timed at the
  start of each step after 0, 1k and 16k cached tokens (`--prefixes`).
- Per step, the best points, the current row and the previous step's winner
  are timed through `ROCM_ATTN` against `TRITON_ATTN` and
  `ROCM_SEGMENTED_ATTN`. A point slower than the best baseline at no cell
  wins over one that is, then the best speedup weighted by sqrt(FLOPs); a step
  keeps the previous step's row within 2 %. The module docstring has the rest.
- `--window`, `--block-size`, `--points`/`--reuse` and `--limit` work as
  above.

`tune.py --apply decisions.jsonl` writes prefill decisions into the prefill
CSV like decode ones. Gemma-4 hybrids give some layers 32- or 64-token pages:
their rows carry that page size.

Check it with the tests above (`-k prefill` for the prefill ones), and measure
it against the baselines:

```bash
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py --prefill \
    --hq 16 --hkv 2 --head-dim 256 --m 128 512 2048 8192 --prefixes 0 16384 \
    --baselines TRITON_ATTN ROCM_SEGMENTED_ATTN > prefill.md
```

The `path` column must read `prefill`.
