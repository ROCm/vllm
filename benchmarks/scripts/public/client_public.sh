# bash client.sh /data/models/amd/Llama-3.1-8B-Instruct-FP8-KV

model=$1
vllm bench serve \
  --host localhost \
  --port 9000 \
  --model ${model} \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 10 \
  --max-concurrency 4 \
  --num-prompts 16 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  # --profile
  # --seed 123 \
  # --request-rate 2 \
  2>&1 | tee log.client.log