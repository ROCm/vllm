#!/bin/bash

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

# for profiling
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_DIR=/root/profiler

# cache dirs
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

# BF16 model
# model_path=/data/pretrained-models/Qwen3-235B-A22B-Instruct-2507
# FP8 model, pure TP8 unsupported due to MoE weight not being divisible by 8, so run with TP8 + EP8 first
model_path=/data/pretrained-models/Qwen3-235B-A22B-Instruct-2507-FP8
vllm serve $model_path \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --enable-expert-parallel \
    --async-scheduling \
    2>&1 | tee qwen3_235b_server.log
