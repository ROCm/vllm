# bash client.sh /data/models/amd/Llama-3.1-8B-Instruct-FP8-KV 1000 1024 1024

MODEL=$1
CONCUR=$2
IN=$3
OUT=$4
NUM=$(expr $CONCUR \* 10)

result_filename=CON${CONCUR}_IN${IN}_OUT${OUT}.json
result_log=CON${CONCUR}_IN${IN}_OUT${OUT}.log

if [ ! -d "bench_results" ]; then
    mkdir bench_results
fi

python3 ../../benchmark_serving.py \
    --backend vllm \
    --host 0.0.0.0 \
    --port 1119 \
    --model $MODEL \
    --dataset-name random \
    --num-prompts ${NUM} \
    --max-concurrency ${CONCUR} \
    --random-input-len ${IN} \
    --random-output-len ${OUT} \
    --save-result \
    --result-dir ./bench_results/ \
    --result-filename ${result_filename} \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    2>&1 | tee ./bench_results/${result_log}

