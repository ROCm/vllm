# bash benchmark_throughput.sh /home/zejchen/models/amd/Llama-3.1-8B-Instruct-FP8-KV 1 bfloat16 1 1024 1024 llama3.1-8B

MODEL=$1
TP=$2
DTYPE=$3
bs=$4
IN=$5
OUT=$6
LOG_NAME=$7

export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_USE_AITER_TRITON_SILU_MUL=0
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_TRITON_FP4_GEMM_USE_ASM=1
export VLLM_USE_V1=1

echo "Here is the environment variables:"
echo "VLLM_ROCM_USE_AITER_MHA=$VLLM_ROCM_USE_AITER_MHA"
echo "VLLM_USE_AITER_TRITON_SILU_MUL=$VLLM_USE_AITER_TRITON_SILU_MUL"
echo "VLLM_ROCM_USE_AITER_PAGED_ATTN=$VLLM_ROCM_USE_AITER_PAGED_ATTN"
echo "VLLM_V1_USE_PREFILL_DECODE_ATTENTION=$VLLM_V1_USE_PREFILL_DECODE_ATTENTION"
echo "VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER"
echo "VLLM_USE_AITER_TRITON_ROPE=$VLLM_USE_AITER_TRITON_ROPE"
echo "VLLM_ROCM_USE_AITER_RMSNORM=$VLLM_ROCM_USE_AITER_RMSNORM"
echo "VLLM_TRITON_FP4_GEMM_USE_ASM=$VLLM_TRITON_FP4_GEMM_USE_ASM"
echo "VLLM_USE_V1=$VLLM_USE_V1"
echo ""

RES_PATH=./bench_results/${LOG_NAME}_TP${TP}_dtype${DTYPE}_wtype${WTYPE}_bs${bs}_input${IN}_output${OUT}
mkdir -p ${RES_PATH}
export VLLM_TORCH_PROFILER_DIR=RES_PATH


python3 ../../benchmark_throughput.py \
    --distributed-executor-backend mp \
    --dtype $DTYPE \
    --disable-detokenize \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --max-model-len 32768 \
    --max-num-batched-tokens 131072 \
    --max-seq-len-to-capture 131072 \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-prompts $bs \
    --max-num-seqs $bs \
    --compilation-config '{"full_cuda_graph": true}' \
    --profile \
    2>&1 | tee ${RES_PATH}/res.log





# ############# fp8 KV Cache ############
# export VLLM_USE_V1=1
# export VLLM_ROCM_USE_AITER=1 
# export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 
# export VLLM_ROCM_USE_AITER_MHA=0 
# export VLLM_DISABLE_COMPILE_CACHE=1 
# for IN in ${array_in[@]}; do
#     for bs in ${array_in[@]}; do
#         RES_PATH=./bench_results/${SIZE}_TP${TP}_dtype${DTYPE}_wtype${WTYPE}_bs${bs}_input${IN}_output${OUT}
#         mkdir -p ${RES_PATH}
#         export VLLM_TORCH_PROFILER_DIR=RES_PATH
#         python3 ../benchmark_throughput.py \
#             --distributed-executor-backend mp \
#             --dtype $DTYPE \
#             --disable-detokenize \
#             --gpu-memory-utilization 0.9 \
#             --trust-remote-code \
#             --model $MODEL \
#             --max-model-len 32768 \
#             --max-num-batched-tokens 131072 \
#             --max-seq-len-to-capture 131072 \
#             --kv-cache-dtype fp8 \
#             --input-len $IN \
#             --output-len $OUT \
#             --tensor-parallel-size $TP \
#             --num-prompts $bs \
#             --max-num-seqs $bs \
#             --compilation-config '{"full_cuda_graph": true}' \
#             --output-json ${RES_PATH}/res.json
#         2>&1 | tee ${RES_PATH}/res.log
#     done
# done





