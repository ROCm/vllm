# Connect to server docker
# docker exec -it vllm-server


# Run the client benchmark
python3 ../../benchmark_serving.py \
  --host localhost \
  --port 8000 \
  --model /data/models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --max-concurrency 4 \
  --num-prompts 32 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  # --profile