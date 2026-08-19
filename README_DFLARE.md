# DFlare vLLM backend

This checkout adds a native `method: "dflare"` path for AngelSlim DFlare
checkpoints targeting Gemma-4.

The implementation reuses DFlash's parallel query scheduling, slot mapping,
rejection sampling, and CUDA-graph boundaries. The draft model differs in two
important ways:

- target hidden states remain concatenated until the draft model applies its
  learned layer-wise fusion weights;
- each draft layer has separate target-context `K/V` projections and
  draft-query `K/V` projections.

## Convert a checkpoint

```bash
python tools/convert_angelslim_dflare.py \
  /path/to/angelslim/checkpoint \
  /path/to/gemma4-e4b-dflare-vllm
```

The converter emits a vLLM-facing config with `hidden_size` equal to the
concatenated target feature width and `draft_hidden_size` equal to the actual
draft transformer width.

## Serve

```bash
vllm serve google/gemma-4-E4B-it \
  --speculative-config '{
    "method": "dflare",
    "model": "/path/to/gemma4-e4b-dflare-vllm",
    "num_speculative_tokens": 15,
    "draft_tensor_parallel_size": 1
  }'
```

The current implementation is text-first. Multimodal Gemma target hidden
states are intentionally routed through the same auxiliary-state contract, but
image-request validation should be performed before enabling production VLM
traffic.
