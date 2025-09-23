# Setup environment for deepep on ROCM
This guide is based on https://amd.atlassian.net/wiki/spaces/MLSE/pages/1119057874/vLLM+deployment+with+DeepEP.
## Prepare nv-shmem and deepep repo
nv-shmem, https://github.com/anghostcici/nvshmem.git, is a private repo owned by Nicole.Nie. You may use this zip to acquire the latest nvshmem for its build [nvshmem-main.zip](https://amdcloud-my.sharepoint.com/:u:/g/personal/danli103_amd_com/EUi9AszhzDBNiZk0P_fF_C0BP2TW60WWaQD6dxyykzyq5w?e=PGNapB) and unzip it locally.
In case you cannot access this zip file, the alternative is to scp it:
```shell
cd vllm/docker/deepep_required
scp -r gyu@hjbog-srdc-35.amd.com:/home/gyu/nvshmem-main.zip ./
unzip nvshmem-main.zip
```
Then clone deepep:
```shell
git clone -b develop-ali https://github.com/ROCm/deepEP_private.git
```
**NOTE: Do remember to put nv-shmem and deepep under ./deepep_required/**
## Build docker image
```shell
DOCKER_BUILDKIT=1 docker build --no-cache -f Dockerfile.rocm_deepep_base -t rocm/vllm_deepep .
```
## Launch docker and install deepep inside container
```shell
docker run -itd --rm --cap-add=SYS_PTRACE -e SHELL=/bin/bash --network=host --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri -v /home/gyu:/home/gyu/ -v /mnt:/mnt --group-add video --ipc=host --name ${CONTAINER_NAME} rocm/vllm_deepep
```
nv-shmem and deepep will be **automatically built and installed** after launching the docker, which can take some time.
Please wait until you can find **.deepep_installed** file in /opt, whcih means the installation has finished.
## Run vLLM with deepep_high_throughput mode
Enter the container and run vLLM with deepep. Currently only eager mode is verified.
```shell
# enter container
docker exec -it ${CONTAINER_NAME} bash
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

