# Setup environment for deepep on ROCM
This guide is based on https://amd.atlassian.net/wiki/spaces/MLSE/pages/1119057874/vLLM+deployment+with+DeepEP.

## Build docker image
```shell
DOCKER_BUILDKIT=1 docker build --no-cache -f Dockerfile.rocm_deepep_base -t rocm/vllm_deepep .
```
## Launch docker and install deepep inside container
```shell
docker run -itd --rm --cap-add=SYS_PTRACE -e SHELL=/bin/bash --network=host --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri -v /home/gyu:/home/gyu/ -v /mnt:/mnt --group-add video --ipc=host --name ${CONTAINER_NAME} rocm/vllm_deepep

docker exec -it ${CONTAINER_NAME} bash

bash install_deepep.sh
```
## Run vLLM with deepep_high_throughput mode
Currently only eager mode is verified.
```shell
# launch server
export HF_HOME=/mnt/raid0/models/
export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_DEBUG=WARN
export VLLM_ALL2ALL_BACKEND=deepep_high_throughput
export VLLM_LOGGING_LEVEL=DEBUG

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --trust-remote-code \
  --disable-log-requests \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --quantization fp8 \
  --no-enable-prefix-caching \
  --max_num_batched_tokens 32768 \
  --enable-expert-parallel \
  --enforce-eager \
  --max_seq_len_to_capture 32768
```

