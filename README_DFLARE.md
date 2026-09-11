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

The runtime preserves AngelSlim's repeat-interleaved rotary layout for both
draft queries and cached target-context keys. Rejected context rows are removed
before projection, and accepted context K/V remains position-indexed across
decode steps. These details are required for serving acceptance to match
teacher-forced evaluation.

Gemma-4 multimodal requests use the target model's auxiliary hidden-state
contract. The implementation has been validated on 200 RoboVQA image/text
requests with a reduced 4K draft vocabulary, AWQ draft body, K=15, V2, and CUDA
graphs.
