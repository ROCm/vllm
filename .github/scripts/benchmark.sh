#!/bin/bash
set -euo pipefail

# Usage function
usage() {
    echo "Usage: $0 <model_name> <output_path>"
    exit 1
}

# Check arguments
if [ $# -lt 2 ]; then
    usage
fi

model_name="${1:-deepseekr1_ptpc_fp8}"
output_path="$2"

# Launch vLLM server
echo "================================================"
echo "========== LAUNCHING vLLM SERVER =============="
evaluation/launch_${model_name}.sh &

VLLM_PID=$!

echo "================================================"
echo "========== WAITING FOR SERVER TO BE READY ========"
MAX_RETRIES=60
RETRY_INTERVAL=60
for ((i=1; i<=MAX_RETRIES; i++)); do
    if curl -s http://localhost:8000/v1/completions -o /dev/null; then
        echo "vLLM server is up."
        break
    fi
    echo "Waiting for vLLM server to be ready... ($i/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
done

if ! curl -s http://localhost:8000/v1/completions -o /dev/null; then
    echo "vLLM server did not start after $((MAX_RETRIES * RETRY_INTERVAL)) seconds."
    kill $VLLM_PID
    exit 1
fi

echo "================================================"
echo "========== CURLING THE REQUEST ================"
curl -X POST "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The capital of China", "temperature": 0, "top_p": 1, "top_k": 0, "repetition_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "stream": false, "ignore_eos": false, "n": 1, "seed": 123 
    }' || true

echo "================================================"
echo "========== STARTING THE TEXT MODEL EVALUATION =========="
lm_eval \
    --model local-completions \
    --tasks gsm8k \
    --model_args model="$model_path",base_url=http://127.0.0.1:8000/v1/completions \
    --batch_size 100 \
    --output_path "$output_path"

EXIT_CODE=$?
echo "================================================"
echo "========== EXIT CODE: $EXIT_CODE ==============="
if [ $EXIT_CODE -eq 0 ]; then
    echo "========== vLLM BENCHMARK COMPLETED SUCCESSFULLY =========="
    echo "========== EXIT CODE: $EXIT_CODE ==============="
else
    echo "========== vLLM BENCHMARK FAILED WITH EXIT CODE $EXIT_CODE =========="
    echo "========== EXIT CODE: $EXIT_CODE ==============="
    exit $EXIT_CODE
fi
echo "================================================"
