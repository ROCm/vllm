# RDNA3.5 attention: adding, tuning and checking a configuration

`ROCM_RDNA_ATTN` (gfx1151, selected by default there) runs decodes
of up to 8 query tokens per sequence on
`csrc/rocm/rdna_decode_attn.cu` and everything else on
`csrc/rocm/rdna_prefill_attn.cu`. Its tables, in `csrc/rocm/`:

| File | Entries | Used for |
| --- | --- | --- |
| `rdna_decode_configs.h` | decode builds, one per M | one sequence |
| `rdna_decode_batch_configs.h` | decode builds, one per M | a batch of sequences |
| `rdna_prefill_configs.h` | prefill builds, one per step of M | one sequence |
| `rdna_prefill_batch_configs.h` | per batch cell, a prefill build and head group | a batch of sequences |

Calls without a build run on Triton.

A *configuration* is `(Hq, Hkv, D, window, page size)`, in fp16 and bf16.

## Requirements

- A gfx1151 GPU and a vLLM build of this tree; run from the tree root with
  `PYTHONPATH=$PWD`.
- `D` in 64, 96, 128, 256, 512, `Hq % Hkv == 0`, and a page size that is a
  multiple of 16 (the one vLLM uses for the model, e.g. 544 for Qwen3.5-0.8B).
- Nothing else running on the GPU.

## Decode

Tune each M in fp16 (bf16 reuses the choice), then apply:

```bash
for m in 1 2 3 4 5 6 7 8; do
  PYTHONPATH=$PWD .venv/bin/python \
      benchmarks/kernels/rocm_rdna_hip_attn/tools/tune.py \
      --hq 16 --hkv 2 --head-dim 256 --m $m \
      --points points.jsonl --decisions decisions.jsonl
done
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_hip_attn/tools/tune.py \
    --apply decisions.jsonl --block-sizes 16
```

- `--window W`, `--block-size P`, `--reuse` to resume, `--limit 20` for a
  smoke run.
- `--apply` refuses decisions measured on another kernel source or entry.

Rebuild `_rocm_C` and add the configuration to `tools/shapes.csv`.

## Batched decode

Tune each M over batch specs (`attention_benchmarks` grammar):

```bash
for m in 1 2 3 4 5 6 7 8; do
  PYTHONPATH=$PWD .venv/bin/python \
      benchmarks/kernels/rocm_rdna_hip_attn/tools/tune.py \
      --hq 16 --hkv 2 --head-dim 256 --m $m \
      --batch 2q${m}s4k 2q${m}s32k 4q${m}s16k 8q${m}s4k 8q${m}s8k 12q${m}s2k \
              16q${m}s2k 32q${m}s1k 64q${m}s1k q${m}s32k_15q${m}s1k \
      --points points_batch.jsonl --decisions decisions_batch.jsonl
done
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_hip_attn/tools/tune.py \
    --apply decisions_batch.jsonl --block-sizes 16
```

A batch beyond the 64 MiB scratch budget runs on Triton.

## Prefill

Tune every step of M in one run:

```bash
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_hip_attn/tools/tune.py --prefill \
    --hq 16 --hkv 2 --head-dim 256 \
    --points points.jsonl --decisions decisions.jsonl
```

An entry serves M from its `min_query_len` up to the next entry's. Steps start
at 16, 32, ..., 4096; the first entry also serves 9 to 15 tokens.

## Batched prefill

Per cell of (mean query tokens, mean keys per token), the prefill build and
head group, once the prefill entries exist:

```bash
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_hip_attn/tools/tune.py --prefill-batch \
    --hq 16 --hkv 2 --head-dim 256 --decisions decisions_prefill_batch.jsonl
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_hip_attn/tools/tune.py \
    --apply decisions_prefill_batch.jsonl
```

`--apply` refuses the cells if the prefill entries changed since.

## Check

```bash
.venv/bin/python -m pytest -q \
    tests/kernels/attention/test_rocm_rdna_hip.py \
    tests/v1/attention/test_rocm_attention_backends_selection.py
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_hip_attn/tools/matrix.py \
    --hq 16 --hkv 2 --head-dim 256 --m 1 2 3 4 5 6 7 8 \
    --baselines TRITON_ATTN ROCM_ATTN > matrix.md
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_hip_attn/tools/matrix.py \
    --hq 16 --hkv 2 --head-dim 256 --batch decode prefill-batch mixed \
    --compare > batches.md
```

Add `--prefill` (with `--m` and `--prefixes`), `--windowed`, `--block-size P`
and `--spec-tokens N` as needed. The `path` column must read `kernel` or
`prefill`. Run a model eval when outputs change.
