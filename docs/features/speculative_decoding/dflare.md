# DFlare

DFlare uses DFlash's parallel query scheduling, slot mapping, rejection
sampling, and CUDA-graph boundaries with a draft model that applies learned
layer-wise fusion to target hidden states. Each draft layer has separate
target-context and draft-query key/value projections.

## Convert an AngelSlim checkpoint

```bash
python tools/convert_angelslim_dflare.py \
  /path/to/angelslim/checkpoint \
  /path/to/gemma4-dflare-vllm
```

The converter writes a vLLM draft configuration. Its `hidden_size` is the
concatenated target-feature width, while `draft_hidden_size` is the draft
transformer's internal width.

## Serve

```bash
vllm serve google/gemma-4-E4B-it \
  --speculative-config '{
    "method": "dflare",
    "model": "/path/to/gemma4-dflare-vllm",
    "num_speculative_tokens": 15,
    "draft_tensor_parallel_size": 1
  }'
```

The number of speculative tokens must not exceed the checkpoint's trained
block size minus one. DFlare currently requires
`draft_tensor_parallel_size=1`.
