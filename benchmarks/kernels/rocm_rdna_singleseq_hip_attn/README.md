# RDNA3.5 attention: adding, tuning and checking a configuration

`ROCM_RDNA_SINGLESEQ_ATTN` (gfx1151, `--max-num-seqs 1`, selected by default
there; on RDNA4, gfx1200 and gfx1201, opt-in) runs calls of up to 8 query
tokens on
`csrc/rocm/rdna_singleseq_decode_attn.cu` and longer ones on
`csrc/rocm/rdna_singleseq_prefill_attn.cu`. Each entry of
`csrc/rocm/rdna_singleseq_{decode,prefill}_configs.h` is one build; calls
without one run on Triton.

A *configuration* is `(Hq, Hkv, D, window, page size)`, with decode entries
for M = 1..8 and prefill entries per step of M, in fp16 and bf16. Each GPU
family has its own complete tables (`kGfx115X*Configs` for gfx1151,
`kGfx120X*Configs` for RDNA4), built for its archs only; tune.py
measures and `--apply` writes the family of the GPU it runs on. Prefill
`v_t` 2 loads V^T with GLOBAL_LOAD_TR (gfx12; gfx11 builds it as 1).

## Requirements

- A gfx1151 or RDNA4 GPU and a vLLM build of this tree; run from the tree
  root with `PYTHONPATH=$PWD`. jit.py builds for the GPU's arch
  (`RDNA_SINGLESEQ_JIT_ARCH` overrides), and the rooflines use each GPU's
  measured peaks (tools/utils.py).
- `D` in 64, 96, 128, 256, 512, `Hq % Hkv == 0`, and a page size that is a
  multiple of 16 (the one vLLM uses for the model, e.g. 544 for Qwen3.5-0.8B).
- `HuggingFaceTB/SmolLM2-135M`'s config in the HF cache, and nothing else
  running on the GPU.

## Decode

Tune each M in fp16 (bf16 reuses the choice), then apply:

```bash
for m in 1 2 3 4 5 6 7 8; do
  PYTHONPATH=$PWD .venv/bin/python \
      benchmarks/kernels/rocm_rdna_singleseq_hip_attn/tools/tune.py \
      --hq 16 --hkv 2 --head-dim 256 --m $m \
      --points points.jsonl --decisions decisions.jsonl
done
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_singleseq_hip_attn/tools/tune.py \
    --apply decisions.jsonl --block-sizes 16
```

- `--window W`, `--block-size P`, `--reuse` to resume, `--limit 20` for a
  smoke run.
- `--apply` refuses decisions measured on another kernel source or entry.

Rebuild `_rocm_C` and add the configuration to `tools/shapes.csv`.

## Prefill

Tune every step of M in one run:

```bash
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_singleseq_hip_attn/tools/tune.py --prefill \
    --hq 16 --hkv 2 --head-dim 256 \
    --points points.jsonl --decisions decisions.jsonl
```

An entry serves M from its `min_query_len` up to the next entry's. Steps start
at 16, 32, ..., 4096; the first entry also serves 9 to 15 tokens.

## Check

```bash
.venv/bin/python -m pytest -q \
    tests/kernels/attention/test_rocm_rdna_singleseq_hip.py \
    tests/v1/attention/test_rocm_attention_backends_selection.py
PYTHONPATH=$PWD .venv/bin/python \
    benchmarks/kernels/rocm_rdna_singleseq_hip_attn/tools/matrix.py \
    --hq 16 --hkv 2 --head-dim 256 --m 1 2 3 4 5 6 7 8 \
    --baselines TRITON_ATTN ROCM_ATTN > matrix.md
```

Add `--prefill` (with `--m` and `--prefixes`), `--windowed` and
`--block-size P` as needed. The `path` column must read `kernel` or `prefill`.
Run a model eval when outputs change.
