#!/bin/bash

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

# ATTN_BACKEND="TRITON_MLA" # This does not support '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' it only supports PIECEWISE
ATTN_BACKEND="ROCM_AITER_MLA"
# ATTN_BACKEND="ROCM_AITER_TRITON_MLA"

export HF_HUB_CACHE=/app/deepep

# for profiling
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_DIR=./deepseek-r1_server_profiler_${ATTN_BACKEND}_3

# cache dirs
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

# BF16 model
# model_path=Qwen/Qwen3-235B-A22B-Instruct-2507
# FP8 model, pure TP8 unsupported due to MoE weight not being divisible by 8, so run with TP8 + EP8 first
model_path=deepseek-ai/DeepSeek-R1-0528
vllm serve $model_path \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --attention-backend ${ATTN_BACKEND} \
    --load-format fastsafetensors \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --async-scheduling \
    --port 1234 \
    2>&1 | tee deepseek-r1_server_${ATTN_BACKEND}_3.log
