# export USE_FASTSAFETENSOR=1 
# export SAFETENSORS_FAST_GPU=1 
# export VLLM_USE_V1=1 
# export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 
# export AMDGCN_USE_BUFFER_OPS=1 
# export VLLM_USE_AITER_TRITON_ROPE=1 
# export VLLM_USE_AITER_TRITON_SILU_MUL=0 
# export TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 
# export TRITON_HIP_USE_ASYNC_COPY=1 
# export TRITON_HIP_USE_BLOCK_PINGPONG=1 
# export TRITON_HIP_ASYNC_FAST_SWIZZLE=1 
# export VLLM_ROCM_USE_AITER=1 
# export VLLM_TRITON_FP4_GEMM_USE_ASM=1
# export VLLM_ROCM_USE_AITER_MHA=0 
# export VLLM_ROCM_USE_AITER_PAGED_ATTN=0 
# export VLLM_ROCM_USE_AITER_RMSNORM=1 


export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 
export VLLM_DISABLE_COMPILE_CACHE=1 
export VLLM_USE_V1=1 
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 
export VLLM_TRITON_FP4_GEMM_USE_ASM=1
export AMDGCN_USE_BUFFER_OPS=1 
export VLLM_USE_AITER_TRITON_ROPE=1 
export VLLM_USE_AITER_TRITON_SILU_MUL=0 
export TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 
export VLLM_TRITON_FP4_GEMM_SPLITK_USE_BF16=1 
export TRITON_HIP_USE_ASYNC_COPY=1 
export TRITON_HIP_USE_BLOCK_PINGPONG=1 
export TRITON_HIP_ASYNC_FAST_SWIZZLE=1 
export TRITON_HIP_PRESHUFFLE_SCALES=0 
export VLLM_ROCM_USE_AITER=1 
export VLLM_ROCM_USE_AITER_MHA=0 
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0 
export VLLM_TRITON_FP4_GEMM_BPRESHUFFLE=0 

export VLLM_TORCH_PROFILER_DIR=./bench_results
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_WITH_FLOPS=0
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0
vllm serve /data/models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview \
  --host localhost \
  --port 8000 \
  --swap-space 64 \
  --disable-log-requests \
  --dtype auto \
  --max-model-len 10240 \
  --tensor-parallel-size 1 \
  --max-num-seqs 1024 \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-seq-len-to-capture 16384 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 131072 \
  --compilation-config '{"full_cuda_graph":true}' \
  # --enforce-eager 

