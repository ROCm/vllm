export VLLM_TORCH_PROFILER_DIR=/home/hatwu/OOB/vllm_bench/profile_results
echo "deepseek-ai/DeepSeek-R1"
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_ASMMOE=1
export SAFETENSORS_FAST_GPU=1
MODEL=/data/models/tencent_deepseekR1/deepseek-r1-FP8-Dynamic-from-BF16-PTPC/

IN=3500
OUT=10
python3 /home/hatwu/OOB/rocm_vllm/benchmarks/benchmark_throughput.py \
    --model $MODEL \
    -tp 8 \
    --enable-expert-parallel \
    --block-size 1 \
    --input-len $IN \
    --output-len $OUT \
    --trust-remote-code \
    --disable-log-requests \
    --max-seq-len-to-capture 32768 \
    --max-num-batched-tokens 32768 \
    --no-enable-prefix-caching \
    --profile \



