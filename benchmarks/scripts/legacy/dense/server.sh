# bash server.sh /home/zejchen/models/amd/Llama-3.1-8B-Instruct-FP8-KV 1 bfloat16

#ROCR_VISIBLE_DEVICES=1

export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_USE_AITER_TRITON_SILU_MUL=0
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_TRITON_FP4_GEMM_USE_ASM=1
export VLLM_USE_V1=1
export VLLM_RPC_TIMEOUT=1800000

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
echo "VLLM_RPC_TIMEOUT=$VLLM_RPC_TIMEOUT"
echo ""

MODEL=$1
TP=$2
DTYPE=$3

vllm serve $MODEL \
    --distributed-executor-backend mp \
    --tensor-parallel-size $TP \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-requests \
    --max-seq-len-to-capture 131072 \
    --max-num-batched-tokens 131072 \
    --compilation-config '{"full_cuda_graph": true}' \
    --no-enable-prefix-caching \
    --dtype $DTYPE \
    --port 1119

