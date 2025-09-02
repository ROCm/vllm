# https://github.com/EleutherAI/lm-evaluation-harness
# git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
# cd lm-evaluation-harness
# pip install -e .
# bash accuracy.sh /data/models/amd/Llama-3.1-8B-Instruct-FP8-KV
model=$1
lm_eval \
--model local-completions \
--tasks gsm8k \
--model_args model=${model},base_url=http://127.0.0.1:9000/v1/completions \
--batch_size 100
