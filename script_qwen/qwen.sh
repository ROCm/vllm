export HF_HOME=/mnt/raid0/models/
export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_ROCM_USE_AITER_MHA=1

TP=8
EP=8
PREFIX=/home/zhiwei/
MODEL_NAME="qwen3-480b-ptpc"
LOG_PATH="${HOME}/logs/${MODEL_NAME}"
# mkdir -p ${PROFILE_RESULT_PATH}
# export VLLM_TORCH_PROFILER_DIR=${PROFILE_RESULT_PATH}
# export VLLM_TORCH_PROFILER_WITH_STACK=1
# export VLLM_TORCH_PROFILER_RECORD_SHAPES=1

mkdir -p ${PREFIX}/temp/vllm
mkdir -p ${PREFIX}/temp/inductor
rm -rf ${PREFIX}/temp/vllm/*
rm -rf ${PREFIX}/temp/inductor/*
export VLLM_CACHE_ROOT=${PREFIX}/temp/vllm
export TORCHINDUCTOR_CACHE_DIR=${PREFIX}/temp/inductor

SERVER_LOG_FILE="${LOG_PATH}/tp${TP}_ep${EP}_server.log"

#vllm serve /mnt/raid0/models/Qwen3-Coder-480B-A35B-Instruct-FP8-ptpc \
#--trust-remote-code \
#--disable-log-requests \
#--max-model-len 32768 \
#--tensor-parallel-size ${TP} \
#--max_seq_len_to_capture 32768 \
#--no-enable-prefix-caching \
#--enable-expert-parallel \
#--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
#--max_num_batched_tokens 32768 2>&1 | tee ${SERVER_LOG_FILE}


VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --port 8000 --tensor-parallel-size 4 --max-model-len 262144 --block-size=1 --enforce-eager

