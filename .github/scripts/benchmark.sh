#!/bin/bash
set -euo pipefail

# Check if model_path argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

# Set environment variables for vLLM and ROCm
export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_DEBUG=WARN
export VLLM_RPC_TIMEOUT=1800000
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_TRITON_ROPE=1 # for accuracy
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 # for accuracy, performance may not be good in some cases

# Set profiling options
export VLLM_TORCH_PROFILER_DIR="deepseek_in3k_out1k"
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1

# Get the model path from the first argument
model_path="$1"

echo "Benchmarking model at path: $model_path"
ls "$model_path"

# Launch the vLLM server (add a timeout to prevent indefinite hanging)
echo "Launching vLLM server with model at $model_path..."
timeout 2h vllm serve "$model_path" \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --gpu_memory_utilization 0.9 \
    --block-size 1

echo "Benchmark script completed."
