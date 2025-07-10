export HIP_VISIBLE_DEVICES=6
# export HIP_VISIBLE_DEVICES=1
# export HIP_VISIBLE_DEVICES=2


export PYTHONPATH=/data/heyanguang/code/vllm_fa_batch_prefill/rocm_vllm:$PYTHONPATH

VLLM_ROCM_USE_AITER=1 VLLM_USE_V1=1 vllm serve /data/heyanguang/code/models/Llama-3.1-8B-Instruct-FP8-KV \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code --disable-log-requests \
    --block-size 16 \
    --max-model-len 32768 \
    --dtype float16 \
    --quantization fp8 \
    --no-enable-prefix-caching \
    --port 30000
