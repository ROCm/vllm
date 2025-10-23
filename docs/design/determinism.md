# Deterministic LLM serving in vLLM on ROCm

## Current state

We have achieved SGLang parity but we are a long way to go from true determinism.

We followed the RFC proposal listed here [\[Feature\]: Kernel Dispatch Overrides (in pursuit of deterministic execution) · Issue #25404 · vllm-project/vllm](https://github.com/vllm-project/vllm/issues/25404)

By that, we mean we enabled per-kernel overrides that enabled deterministic execution on ROCm.

Layernorm, topk_softmax and other basic building blocks we got thanks to the prior research done by the SGLang blogpost and the Meta folks kernel.

FlexAttention did not work out-of-the-box on ROCm but we enabled it upstream with a simple fix. All that is needed now is VLLM_ATTENTION_BACKEND=FLEX_ATTENTION

Comparing correctness of default attention backend vs FlexAttention  
lm_eval --model local-completions --model_args model=meta-llama/Llama-3.1-8B,base_url=<http://0.0.0.0:8000/v1/completions,num_concurrent=128,max_retries=5> --tasks gsm8k

**Test Result**

Default:

|Tasks|Version| Filter |n-shot| Metric | |Value | |Stderr|

|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|

|gsm8k| 3|flexible-extract| 5|exact_match|↑ |0.5011|± |0.0138|

| | |strict-match | 5|exact_match|↑ |0.5011|± |0.0138|

FlexAttention:

|Tasks|Version| Filter |n-shot| Metric | |Value | |Stderr|

|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|

|gsm8k| 3|flexible-extract| 5|exact_match|↑ |0.4882|± |0.0138|

| | |strict-match | 5|exact_match|↑ |0.4875|± |0.0138|

Just like the deterministic op support for non-ROCm target, you can enable the FlexAttention one - for example KERN_OVERRIDE_FLEX_ATTN_DETERMINISTIC_SPLIT_TILE_SIZE=4096

We support the hooks for determinism that are upstream:

- C++ hook by using bool deterministic_launch = vllm_kernel_override_determinism_all()
- Python hook by using vllm_kernel_override_determinism_all() within vllm.model_executor.layers.determinism

## Future Work

More batch invariant ops is the biggest challenge in determinism. Everyone in the community is working on this.

The operators developed by thinking machines ([thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops/tree/main)) are not license compatible with vLLM of course.

- We have dedicated an engineer for enabling such operators on ROCm as research and development matures.
- Red Hat has also dedicated an engineer to look into this full time for all HW targets in vLLM
- Meta, of course, has been leading this research

## Biggest future challenges

- MoE: This is the operator that would be trickiest to solve. No implementation under any license in any project managed to make it. For MoE models we need a deterministic MoE kernel
- High tensor parallelism: No project managed to really do TP>1. We need a deterministic all reduce or quick reduce across GPUs
