#!/bin/bash

echo "deepseek-ai/DeepSeek-R1"
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_ASMMOE=1
export SAFETENSORS_FAST_GPU=1
export MODEL_PATH=/data/models/tencent_deepseekR1/deepseek-r1-FP8-Dynamic-from-BF16/

vllm serve $MODEL_PATH \
    -tp 8 \
    --block-size 1 \
    --trust-remote-code \
    --disable-log-requests \
    --max-seq-len-to-capture 32768 \
    --max-num-batched-tokens 32768 \
    --no-enable-prefix-caching \
