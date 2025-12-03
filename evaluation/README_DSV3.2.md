# ENV Setup

aiter branch: `main`

vllm branch: `ROCm/vllm: ganyi/optimize_dsv3.2_metadata`

triton: 3.5+ (need to update to 3.5 or higher version)

pull the aot gluon kernel to local 
```bash 
git clone -b aiter_aot https://github.com/ROCm/triton-kernels.git
# copy the aot version of kernel from triton-kernels to aiter
cp -r ~/triton-kernels/kernels/configs/paged_mqa_logits/aot ~/aiter/ops/triton/configs/paged_mqa_logits

```

# Deepseek V3.2 launch script 

```bash
export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_DEBUG=WARN
export VLLM_RPC_TIMEOUT=18000000
export VLLM_ROCM_USE_AITER_ASMMOE=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_TORCH_PROFILER_DIR=./vllm_profile
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1

# model_path="/mnt/raid0/ygan/Deepseekv3.2"
model_path="/mnt/raid0/zhangguopeng/DeepSeek-V3.2-Exp"  # --- IGNORE ---
 
vllm serve $model_path \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --kv-cache-dtype bfloat16 \
  --gpu_memory_utilization 0.85 \
  --load-format fastsafetensors \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --block-size 64 \
```

# Accuracy verification
Following the upper step, you may reproduce this accuarcy resutl on dsv3.2

```
# verification script

lm_eval --model local-completions  \
   --tasks gsm8k  \
   --output_path ./results  \
   --log_samples \
   --model_args model=/data/models/deepseek-ai/DeepSeek-V3.2-Exp,base_url=http://localhost:8000/v1/completions,num_concurrent=128,max_retries=3,timeout=3000,seed=1234,temperature=0

# Verification result
# 5-shot
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9606|±  |0.0054|
|     |       |strict-match    |     5|exact_match|↑  |0.9606|±  |0.0054|

# 20-shot
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|    20|exact_match|↑  |0.9507|±  |0.0060|
|     |       |strict-match    |    20|exact_match|↑  |0.9515|±  |0.0059|
```