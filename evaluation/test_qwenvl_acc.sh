#!/bin/bash
git clone https://github.com/mistralai/mistral-evals.git
pushd ./mistral-evals
python3 -m eval.run eval_vllm \
    --model_name /data/pretrained-models/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --url http://0.0.0.0:8000 \
    --output_dir ./chartqa \
    --eval_name "chartqa" 2>&1 | tee ../qwen3_vl_235b_fp8_acc.log
popd