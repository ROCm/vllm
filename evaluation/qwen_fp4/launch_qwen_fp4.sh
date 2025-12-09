export VLLM_USE_V1=1
# export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_TRITON_FLASH_ATTN=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_TRITON_ROPE=1 # add for acc
export VLLM_DISABLE_COMPILE_CACHE=1
# FIXME: for now disable fp4 asm gemm because of running issue
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 # disable for acc

export TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1
export TRITON_HIP_USE_ASYNC_COPY=1
export TRITON_HIP_USE_BLOCK_PINGPONG=1
export TRITON_HIP_ASYNC_FAST_SWIZZLE=1
export NCCL_DEBUG=WARN
export AMDGCN_USE_BUFFER_OPS=1
export SAFETENSORS_FAST_GPU=1

# for profiling
#export VLLM_TORCH_PROFILER_DIR="deepseek_in3k_out1k"
#export VLLM_TORCH_PROFILER_WITH_STACK=1
#export VLLM_TORCH_PROFILER_RECORD_SHAPES=1

model_path=/data/pretrained-models/Qwen3-235B-A22B-Thinking-2507_moe_w_mxfp4_a_mxfp4_kv_fp8
echo "running $model_path"

vllm serve $model_path \
  --host localhost \
  --port 9000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --gpu_memory_utilization 0.8 \
  --async-scheduling \
  --block-size 16 \
  --seed 123 2>&1 | tee log.server.log

  # --load-format fastsafetensors \
  
  # --enforce-eager \
  # --data-parallel-hybrid-lb \
