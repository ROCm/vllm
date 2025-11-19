#!/bin/bash

model="/data/pretrained-models/Qwen3-VL-235B-A22B-Instruct"
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --host localhost \
  --port 8000 \
  --model ${model} \
  --dataset-name random-mm \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config "{(800, 800, 1): 1.0}" \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --num-prompts 128 \
  --max-concurrency 64 \
  2>&1 | tee qwen3vl_235b_client.log