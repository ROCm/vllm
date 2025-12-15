model="/mnt/data/zhangguopeng/deepseek-r1-FP8-Dynamic/"
model="/mnt/data/pretrained_model/deepseek-ai/DeepSeek-V3/"
model="/mnt/data/pretrained_model/Qwen/Qwen2-1.5B-Instruct/"
vllm bench serve \
  --host localhost \
  --port 1234 \
  --model ${model} \
  --dataset-name random \
  --random-input-len 20000 \
  --random-output-len 1 \
  --max-concurrency 1 \
  --num-prompts 3 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  --profile \
  # --seed 123 \
  # --request-rate 2 \
  #2>&1 | tee log.client.log
