NUM_PROMPTS=$1
MAX_CONCURRENCY=$2
IN=$3
OUT=$4
python3 -m vllm.entrypoints.cli.main bench serve --backend vllm  \
    --model /mnt/raid0/models/Qwen3-Coder-480B-A35B-Instruct-FP8-ptpc \
    --dataset-name random --num-prompts ${NUM_PROMPTS} \
    --max-concurrency ${MAX_CONCURRENCY} \
    --random-input-len ${IN} \
    --random-output-len ${OUT} \
    --ignore-eos \
    #--profile \