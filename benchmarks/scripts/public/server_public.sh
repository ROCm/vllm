# bash server.sh /data/models/amd/Llama-3.1-8B-Instruct-FP8-KV 

model=$1
export USE_FASTSAFETENSOR=1
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_V1=1
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export AMDGCN_USE_BUFFER_OPS=1
export VLLM_USE_AITER_TRITON_ROPE=1
export VLLM_USE_AITER_TRITON_SILU_MUL=1
export TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1
export TRITON_HIP_USE_ASYNC_COPY=1
export TRITON_HIP_USE_BLOCK_PINGPONG=1
export TRITON_HIP_ASYNC_FAST_SWIZZLE=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
export VLLM_ROCM_USE_AITER_RMSNORM=1

export VLLM_TORCH_PROFILER_DIR=./bench_results
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_WITH_FLOPS=0
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0
vllm serve ${model} \
  --host localhost \
  --port 9000 \
  --swap-space 64 \
  --disable-log-requests \
  --dtype auto \
  --max-model-len 10240 \
  --tensor-parallel-size 1 \
  --max-num-seqs 1024 \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --max-seq-len-to-capture 16384 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --max-num-batched-tokens 131072 \
  2>&1 | tee log.server.log &

