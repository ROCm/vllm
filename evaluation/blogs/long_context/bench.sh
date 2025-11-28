model="/mnt/raid0/zhangguopeng/deepseek-r1-FP8-Dynamic/"
model="/mnt/raid0/pretrained_model/deepseek-ai/DeepSeek-V3/"
model="/mnt/raid0/pretrained_model/Qwen/Qwen2-1.5B-Instruct/"
vllm bench serve \
  --host localhost \
  --port 1234 \
  --model ${model} \
  --dataset-name random \
  --random-input-len 30000 \
  --random-output-len 5 \
  --max-concurrency 32 \
  --num-prompts 32 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  --profile \
  # --seed 123 \
  # --request-rate 2 \
  #2>&1 | tee log.client.log
