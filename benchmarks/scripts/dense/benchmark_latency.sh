# bash benchmark_latency.sh /home/zejchen/models/amd/Llama-3.1-8B-Instruct-FP8-KV 1 bfloat16 1 1024 1024 llama3.1-8B

MODEL=$1
TP=$2
DTYPE=$3
array_bs=$4
array_in=$5
array_out=$6
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

RES_PATH=./bench_results/${LOG_NAME}_TP${TP}_dtype${DTYPE}_bs${bs}_input${IN}_output${OUT}
mkdir -p ${RES_PATH}
python3 ../../benchmark_latency.py \
    --distributed-executor-backend mp \
    --dtype $DTYPE \
    --disable-detokenize \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --batch-size $bs \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-iters-warmup 0 \
    --num-iters 1 \
    --compilation-config '{"full_cuda_graph": true}' \
    --output-json ${RES_PATH}/res.json \
    2>&1 | tee ${RES_PATH}/res.log

