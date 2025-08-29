HSA_NO_SCRATCH_RECLAIM=1
NCCL_MIN_NCHANNELS=112
USE_FASTSAFETENSOR=1
SAFETENSORS_FAST_GPU=1
VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1
VLLM_DISABLE_COMPILE_CACHE=1
VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1
VLLM_USE_AITER_TRITON_GEMM=1
VLLM_ROCM_USE_AITER=1
VLLM_USE_AITER_UNIFIED_ATTENTION=1
VLLM_ROCM_USE_AITER_MHA=0
TRITON_HIP_PRESHUFFLE_SCALES=1
 
vllm serve  <model>   --port 9000 --tensor-parallel-size 8 --distributed-executor-backend mp --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.95 --max-model-len 2304  --max-seq-len-to-capture 2304 --swap-space 16  --no-enable-prefix-caching  --disable-log-requests   --block-size 64 --async-scheduling --async-scheduling --compilation-config '{"compile_sizes": [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024,2048, 4096, 8192], "full_cuda_graph": true }'
