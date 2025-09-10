model=/docker/pretrained-models/amd/Llama-3.3-70B-Instruct-FP8-KV

# profile
export VLLM_TORCH_PROFILER_DIR=./profile
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1

vllm bench serve \
  --host localhost \
  --port 9000 \
  --model ${model} \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 8 \
  --max-concurrency 4 \
  --num-prompts 12 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  --seed 123 \
  --request-rate 2 \
  --profile \
  2>&1 | tee log.client.log
