addr=localhost
port=9000
url=http://${addr}:${port}/v1/completions
model=/data/pretrained-models/Qwen3-235B-A22B-Thinking-2507_moe_w_mxfp4_a_mxfp4_kv_fp8
bs=50
task=gsm8k

echo "url=${url}"
echo "model=${model}"
echo "task=${task}"

lm_eval \
    --model local-completions \
    --tasks ${task} \
    --model_args model=${model},base_url=${url} \
    --batch_size ${bs} \
    --seed 123 \
    2>&1 | tee log.lmeval.log
