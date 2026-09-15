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

AngelSlim and vLLM use two related layer-index conventions:

- AngelSlim's `dflare_config.target_layer_ids` are zero-based target-model
  layer indices.
- vLLM's `eagle_aux_hidden_state_layer_ids` are one-based hidden-state output
  indices. The converter adds one to every AngelSlim layer index.
- The converted `dflare_config.target_layer_ids` are zero-based slots in the
  concatenated target-feature tensor, not target-model layer indices. DFlare
  fusion is order-based, so these slots must remain in the same order as
  `eagle_aux_hidden_state_layer_ids`.

Checkpoints in the vLLM speculators format already carry one-based
`aux_hidden_state_layer_ids`; their config adapter performs the corresponding
conversion automatically. Do not manually shift those IDs before serving.

AngelSlim checkpoints also default to the `legacy` rotary layout. The converter
preserves `dflare_config.rope_layout`; overriding it with `neox` for a
legacy-trained checkpoint severely reduces draft acceptance.

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
`draft_tensor_parallel_size=1`. This initial backend supports Gemma 4 target
models; Qwen3 DFlare target support is not included.
