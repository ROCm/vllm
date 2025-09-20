export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_ROCM_USE_AITER_MHA=1

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_TORCH_PROFILER_DIR=./bench_results
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_WITH_FLOPS=0
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0

vllm serve /mnt/raid0/models/Qwen3-Coder-480B-A35B-Instruct-FP8-ptpc \
    --trust-remote-code \
    --max-model-len 32768 \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --max_seq_len_to_capture 32768 \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --max_num_batched_tokens 32768 \
    --kv-cache-dtype fp8

