#export HIP_VISIBLE_DEVICES=4,5,6,7
VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_RPC_TIMEOUT=18000 vllm serve /mnt/raid0/zhangguopeng/deepseek-r1-FP8-Dynamic/ \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-batched-tokens 32768 \
    --max_seq_len_to_capture 32768 \
    --disable-log-requests \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --trust-remote-code \
    --block-size 1 \
    --gpu_memory_utilization 0.9

#--compilation-config '{"compile_sizes": [56], "use_inductor":false, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [56]}' \
#--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [56]}' \
