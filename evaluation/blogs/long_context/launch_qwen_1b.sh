#!/bin/bash

export SAFETENSORS_FAST_GPU=1
export VLLM_RPC_TIMEOUT=1800000

# Triton Unified Attention
#export VLLM_ATTENTION_BACKEND=TRITON_ATTN
#export VLLM_ROCM_USE_AITER=0
#export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0
#export VLLM_ROCM_USE_AITER_MHA=0


# Aiter Unified Attention
#export VLLM_ROCM_USE_AITER=1
#export VLLM_USE_AITER_UNIFIED_ATTENTION=1
#export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0
#export VLLM_ROCM_USE_AITER_MHA=0

# Triton Prefill-Decode Attention
export VLLM_ATTENTION_BACKEND=ROCM_ATTN
# export VLLM_ROCM_USE_AITER=1
# export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
# export VLLM_ROCM_USE_AITER_MHA=0

# AITER Multi-head Attention
# export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
#export VLLM_ROCM_USE_AITER=1
#export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0
#export VLLM_ROCM_USE_AITER_MHA=1


# for profiling
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_DIR=/home/hatwu/profiler

# cache dirs
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

# BF16 model
# model_path=Qwen/Qwen3-235B-A22B-Instruct-2507
# FP8 model, pure TP8 unsupported due to MoE weight not being divisible by 8, so run with TP8 + EP8 first
model_path=/mnt/data/pretrained_model/Qwen/Qwen2-1.5B-Instruct/
vllm serve $model_path \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --gpu_memory_utilization 0.1 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --async-scheduling \
    --port 1234 \
    2>&1 | tee qwen3_235b_server.log
