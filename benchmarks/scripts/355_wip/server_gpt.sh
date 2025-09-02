export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_MIN_NCHANNELS=112
export USE_FASTSAFETENSOR=1
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1
export VLLM_USE_AITER_TRITON_GEMM=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export TRITON_HIP_PRESHUFFLE_SCALES=1

export HIP_VISIBLE_DEVICES=3
#HIP_VISIBLE_DEVICES=4,5,6,7
model=/data/models/gpt-oss-120b/
tp=$1
vllm serve ${model} \
  --host localhost \
  --port 9000 \
  --tensor-parallel-size ${tp} \
  --distributed-executor-backend mp \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2304  \
  --max-seq-len-to-capture 2304 \
  --swap-space 16  \
  --no-enable-prefix-caching  \
  --disable-log-requests   \
  --block-size 64 \
  --async-scheduling \
  --compilation-config '{"compile_sizes": [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024,2048, 4096, 8192], "full_cuda_graph": true }'


