# bash client.sh /data/models/gpt-oss-120b
model=$1
python3 ../../benchmark_serving.py \
  --host localhost \
  --port 9000 \
  --model ${model} \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --max-concurrency 4 \
  --num-prompts 32 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  # --profile
  # --seed 123 \
  # --request-rate 2 \