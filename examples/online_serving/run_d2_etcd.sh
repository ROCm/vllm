#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

set -xe

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart --no-deps
fi 

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}



MOONCAKE_CONFIG_PATH=./mooncake.json VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=4 vllm serve /hf/Meta-Llama-3.1-8B-Instruct \
    --port 8400 \
    --max-model-len 1000 \
    --gpu-memory-utilization 0.5 --tensor-parallel-size 1 \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' 2>&1 | tee d4.log &
MOONCAKE_CONFIG_PATH=./mooncake.json VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=5 vllm serve /hf/Meta-Llama-3.1-8B-Instruct \
    --port 8500 \
    --max-model-len 1000 \
    --gpu-memory-utilization 0.5 --tensor-parallel-size 1 \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'

