#!/bin/bash

PORT=8000
SEED=0

CONCURRENCY=64
NREQUESTS=$(($CONCURRENCY * 10))
ISL=6144
OSL=1024
MODEL_PATH=/data/models/tencent_deepseekR1/deepseek-r1-FP8-Dynamic-from-BF16/


python3 /home/hatwu/OOB/vllm/benchmarks/benchmark_serving.py --backend vllm \
    --model ${MODEL_PATH} \
    --dataset-name random \
    --num-prompts 1 \
    --random-input ${ISL} \
    --random-output ${OSL} \
    --port ${PORT}\
    | tee sglang_benchmark_vllm_random_isl${ISL}_osl${OSL}_con${CONCURRENCY}.log
