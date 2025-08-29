array_bs=(1)
array_in=(1024)
array_out=(1024)

#llama3.3 8B FP8 TP1
MODEL=/home/zejchen/models/amd/Llama-3.1-8B-Instruct-FP8-KV
TP=1
DTYPE=bfloat16
LOG_NAME=llama3.1-8B-FP8
for bs in ${array_bs[@]}; do
    for IN in ${array_in[@]}; do
        for OUT in ${array_out[@]}; do
            bash benchmark_throughput.sh $MODEL $TP $DTYPE $bs $IN $OUT $LOG_NAME
        done
    done
done


#llama3.3 70B BF16 TP8
MODEL=/home/zejchen/models/meta-llama/Llama-3.3-70B-Instruct
TP=8
DTYPE=bfloat16
LOG_NAME=llama3.3-70B-BF16
for bs in ${array_bs[@]}; do
    for IN in ${array_in[@]}; do
        for OUT in ${array_out[@]}; do
            bash benchmark_throughput.sh $MODEL $TP $DTYPE $bs $IN $OUT $LOG_NAME
        done
    done
done

#llama3.3 70B FP8 TP8
MODEL=/home/zejchen/models/amd/Llama-3.3-70B-Instruct-FP8-KV/
TP=8
DTYPE=bfloat16
LOG_NAME=llama3.3-70B-FP8
for bs in ${array_bs[@]}; do
    for IN in ${array_in[@]}; do
        for OUT in ${array_out[@]}; do
            bash benchmark_throughput.sh $MODEL $TP $DTYPE $bs $IN $OUT $LOG_NAME
        done
    done
done

#Qwen3 32B BF16 TP1
MODEL=/home/zejchen/models/Qwen/Qwen3-32B
TP=1
DTYPE=bfloat16
LOG_NAME=Qwen3-32B-BF16
for bs in ${array_bs[@]}; do
    for IN in ${array_in[@]}; do
        for OUT in ${array_out[@]}; do
            bash benchmark_throughput.sh $MODEL $TP $DTYPE $bs $IN $OUT $LOG_NAME
        done
    done
done

#Qwen3 32B FP8 TP1
MODEL=/home/zejchen/models/Qwen/Qwen3-32B-FP8
TP=1
DTYPE=bfloat16
LOG_NAME=Qwen3-32B-FP8
for bs in ${array_bs[@]}; do
    for IN in ${array_in[@]}; do
        for OUT in ${array_out[@]}; do
            bash benchmark_throughput.sh $MODEL $TP $DTYPE $bs $IN $OUT $LOG_NAME
        done
    done
done

#Qwen3 32B FP8 TP8
MODEL=/home/zejchen/models/Qwen/Qwen3-32B-FP8
TP=8
DTYPE=bfloat16
LOG_NAME=Qwen3-32B-FP8
for bs in ${array_bs[@]}; do
    for IN in ${array_in[@]}; do
        for OUT in ${array_out[@]}; do
            bash benchmark_throughput.sh $MODEL $TP $DTYPE $bs $IN $OUT $LOG_NAME
        done
    done
done
