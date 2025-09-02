# https://github.com/EleutherAI/lm-evaluation-harness
# git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
# cd lm-evaluation-harness
# pip install -e .
# bash accuracy.sh /data/models/amd/Llama-3.1-8B-Instruct-FP8-KV
# bash accuracy.sh /data/models/gpt-oss-120b
model=$1
lm_eval \
--model local-completions \
--tasks gsm8k \
--model_args model=${model} \
--host localhost \
--port 9000 \
--batch_size 100
