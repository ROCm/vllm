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
output_path="${2:-benchmark_results.json}"

model_path=$(jq -r ".${model_name}.path" .github/scripts/models_datas.json)
baseline=$(jq -r ".${model_name}.baseline" .github/scripts/models_datas.json)
baseline_strict_match_value=$(echo $baseline | jq -r ".[0].value")
baseline_flexible_extract_value=$(echo $baseline | jq -r ".[1].value")
echo "Model name: $model_name"
echo "Model path: $model_path"
echo "Output path: $output_path"
echo "Baseline strict match value: $baseline_strict_match_value"
echo "Baseline flexible extract value: $baseline_flexible_extract_value"

# Launch vLLM server
echo
echo "========== LAUNCHING vLLM SERVER =============="
./.github/scripts/launch_models.sh $model_name $model_path &

VLLM_PID=$!

echo
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

echo
echo "========== CURLING THE REQUEST ================"
curl -X POST "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The capital of China", "temperature": 0, "top_p": 1, "top_k": 0, "repetition_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "stream": false, "ignore_eos": false, "n": 1, "seed": 123 
    }' || true

echo
echo "========== STARTING THE TEXT MODEL EVALUATION =========="
# Run lm_eval and capture its output
lm_eval \
    --model local-completions \
    --tasks gsm8k \
    --model_args model="$model_path",base_url=http://127.0.0.1:8000/v1/completions \
    --batch_size 100 \
    --output_path "models_performance_test/$output_path"

find models_performance_test -name "*.json" -type f | while read file; do
    echo "----- $file -----"
    cat "$file"
    echo
done

# Parse lm_eval output and compare metrics to baseline

RESULT_FILE="models_performance_test/$output_path"
if [[ ! -f "$RESULT_FILE" ]]; then
    echo "ERROR: Could not find results file at $RESULT_FILE"
    kill $VLLM_PID
    exit 2
fi
echo "RESULT_FILE: $RESULT_FILE"

# Extract metrics from the output json using jq
STRICT=$(jq '.results.gsm8k["exact_match,strict-match"]' "$RESULT_FILE")
FLEXIBLE=$(jq '.results.gsm8k["exact_match,flexible-extract"]' "$RESULT_FILE")

echo
echo "========== RESULTS COMPARISON =============="
echo "Strict Match:    $STRICT (baseline: $BASELINE_STRICT)"
echo "Flexible Match:  $FLEXIBLE (baseline: $BASELINE_FLEXIBLE)"

# Calculate delta with baseline
DELTA_STRICT=$(awk -v current="$STRICT" -v base="$BASELINE_STRICT" 'BEGIN { d=current-base; printf "%+.6f", d }')
DELTA_FLEXIBLE=$(awk -v current="$FLEXIBLE" -v base="$BASELINE_FLEXIBLE" 'BEGIN { d=current-base; printf "%+.6f", d }')

echo "Delta Strict:    $DELTA_STRICT"
echo "Delta Flexible:  $DELTA_FLEXIBLE"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo
    echo "========== vLLM BENCHMARK COMPLETED SUCCESSFULLY =========="
else
    echo
    echo "========== vLLM BENCHMARK FAILED WITH EXIT CODE $EXIT_CODE =========="
    exit $EXIT_CODE
fi
