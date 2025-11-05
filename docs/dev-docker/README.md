# vLLM FP8 latency and throughput benchmarks on the AMD Instinct™ MI300X GPU

vLLM is a toolkit and library for large language model (LLM) inference and serving. It uses the PagedAttention algorithm to reduce memory consumption and increase throughput through dynamic key and value allocation in GPU memory. vLLM also incorporates the latest LLM acceleration and quantization algorithms, such as FP8 GeMM, FP8 KV cache, continuous batching, flash attention, HIP graph, tensor parallelism, GPTQ, AWQ, and token speculation. In addition, AMD implements high-performance custom kernels and modules in vLLM to further enhance performance.

This documentation ceovers running Meta’s popular Llama 3.1 series models with a pre-built AMD vLLM Docker image optimized for AMD Instinct™ MI300X or MI325X GPUs. The container is available on [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html).

The prebuilt image includes:

- ROCm™ 7.0.0
- HipBLASLt 1.0.0
- vLLM 0.11.1 RC
- PyTorch 2.9

## Pull the latest Docker image

Pull the latest validated Docker image with `docker pull rocm/vllm:latest`

## What is new

- Added support for Llama 4 FP4 and Granite 4 models.
- Updated to vLLM 0.11.1 RC.
- Enabled AITER by default.

## Known issues and workarounds

- AITER must be explicitly disabled on GPUs other than gfx942 and gfx950.
- Disable AITER for Llama 3.1 405B FP8 for better performance with a batch size of 1 (BS=1).
- Performance drops may occur with an input and output sequence length of 128/128 on the Llama4 Maverick 17B 128E FP8 model.

## Performance results

The data in the following tables serves as a reference to help you validate observed performance. It should not be interpreted as the peak performance achievable on the AMD Instinct™ MI300X GPU with vLLM. For details on MLPerf 4.1 inference results, see the MLPerf section in this document. The performance numbers were collected using the steps described below.

### Throughput measurements

The table below shows performance data where a local inference client receives requests at an infinite rate, illustrating the client-server throughput under maximum load.

| Model | Precision | TP Size | Input | Output | Num Prompts | Max Num Seqs | Throughput (tokens/s) |
|-------|-----------|---------|-------|--------|-------------|--------------|-----------------------|
| Llama 3.1 70B (amd/Llama-3.1-70B-Instruct-FP8-KV) | FP8 | 8 | 128 | 2048 | 3200 | 3200 | 13279.6  |
|       |           |         | 128   | 4096   | 1500        | 1500         | 11449.7               |
|       |           |         | 500   | 2000   | 2000        | 2000         | 11347.4               |
|       |           |         | 2048  | 2048   | 1500        | 1500         | 7651.7                |
| Llama 3.1 405B (amd/Llama-3.1-405B-Instruct-FP8-KV) | FP8 | 8 | 128 | 2048 | 1500 | 1500 | 3816.8 |
|       |           |         | 128   | 4096   | 1500        | 1500         | 3099.6                |
|       |           |         | 500   | 2000   | 2000        | 2000         | 3026.1                |
|       |           |         | 2048  | 2048   | 500         | 500          | 2196.4                |

*TP stands for Tensor Parallelism.*

Supermicro AS-8125GS-TNMR2 with 2x AMD EPYC 9554 processors, 2.25 TiB RAM, 8x AMD Instinct MI300X GPUs (192GiB, 750W), Ubuntu 22.04, and amdgpu driver 6.8.5

### Latency measurements

The table below shows latency measurements, which capture the time from when the system receives an input to when the model produces a result.

| Model | Precision | TP Size | Batch Size | Input | Output | MI300X Latency (sec) |
|-------|-----------|----------|------------|--------|---------|-------------------|
| Llama 3.1 70B (amd/Llama-3.1-70B-Instruct-FP8-KV) | FP8 | 8 | 1 | 128 | 2048 | 16.154 |
| | | | 2 | 128 | 2048 | 18.041 |
| | | | 4 | 128 | 2048 | 18.322 |
| | | | 8 | 128 | 2048 | 20.800 |
| | | | 16 | 128 | 2048 | 21.850 |
| | | | 32 | 128 | 2048 | 25.513 |
| | | | 64 | 128 | 2048 | 32.539 |
| | | | 128 | 128 | 2048 | 45.193 |
| | | | 1 | 2048 | 2048 | 16.256 |
| | | | 2 | 2048 | 2048 | 18.084 |
| | | | 4 | 2048 | 2048 | 18.851 |
| | | | 8 | 2048 | 2048 | 20.930 |
| | | | 16 | 2048 | 2048 | 23.079 |
| | | | 32 | 2048 | 2048 | 26.873 |
| | | | 64 | 2048 | 2048 | 34.585 |
| | | | 128 | 2048 | 2048 | 51.856 |
| Llama 3.1 405B (amd/Llama-3.1-405B-Instruct-FP8-KV) | FP8 | 8 | 1 | 128 | 2048 | 48.138 |
| | | | 2 | 128 | 2048 | 48.366 |
| | | | 4 | 128 | 2048 | 49.790 |
| | | | 8 | 128 | 2048 | 53.546 |
| | | | 16 | 128 | 2048 | 55.685 |
| | | | 32 | 128 | 2048 | 67.445 |
| | | | 64 | 128 | 2048 | 86.597 |
| | | | 128 | 128 | 2048 | 120.387 |
| | | | 1 | 2048 | 2048 | 48.555 |
| | | | 2 | 2048 | 2048 | 48.348 |
| | | | 4 | 2048 | 2048 | 49.828 |
| | | | 8 | 2048 | 2048 | 53.415 |
| | | | 16 | 2048 | 2048 | 57.398 |
| | | | 32 | 2048 | 2048 | 68.519 |
| | | | 64 | 2048 | 2048 | 90.234 |
| | | | 128 | 2048 | 2048 | 130.518 |

*TP stands for Tensor Parallelism.*

Supermicro AS-8125GS-TNMR2 with 2x AMD EPYC 9554 processors, 2.25 TiB RAM, 8x AMD Instinct MI300X (192GiB, 750W) GPUs, Ubuntu 22.04, and amdgpu driver 6.8.5

## Reproducing benchmark results

### Preparation- accessing the models

The vLLM Docker image supports any model compatible with vLLM.  When running with FP8,AMD provides quantized versions of several popular models, or you can quantize models yourself using Quark.  The vLLM benchmark scripts can automatically download models and store them in a Hugging Face cache directory for reuse in future tests. Alternatively, you can pre-download the model to the cache or another directory on your system.

Many HuggingFace models, including Llama-3.1, have gated access.  You will need to set up an account at [Hugging Face](https://huggingface.co), search for the model of interest, and request access if necessary. You will also need to create a token for accessing these models from vLLM by going to your [user profile](https://huggingface.co/settings/profile), selecting **Access Tokens**, clicking **+ Create New Token**, and creating a new **Read** token.

### System optimization

Before running performance tests you should ensure the system is optimized according to the [ROCm documentation](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html).  In particular, it is important to ensure that NUMA auto-balancing is disabled.

> **Note:** Check that NUMA balancing is properly set by inspecting the output of the command below, which should have a value of 0, with, `cat /proc/sys/kernel/numa_balancing`*

### Launch AMD vLLM Docker

Download and launch the docker.  The HF_TOKEN is required to be set (either here or after launching the container) if you want to allow vLLM to download gated models automatically; use your HuggingFace token in place of `<token>` in the command below:

```bash
docker run -it --rm --ipc=host --network=host --group-add render \
    --privileged --security-opt seccomp=unconfined \
    --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    -e HF_HOME=/data \
    -e HF_TOKEN=<token> \
    -v /data:/data \
    rocm/vllm:latest
```

> **Note:** This document uses `/data` to store models.  If you choose a different directory, you will also need to update the host volume mount when launching the Docker container.  For example, `-v /home/username/models:/data` instead of `-v /data:/data` will store models in `/home/username/models` on the host. Some models can be large, so ensure you have sufficient disk space before downloading. Because downloads may take a long time, consider using `tmux` or `screen` to prevent disconnection.

### Downloading models with Hugging Face CLI

To download models directly (instead of letting vLLM download them automatically), you can use the `huggingface-cli` inside the running Docker container. Log in using the token you created earlier. It is not necessary to save the token as a git credential.

```bash
huggingface-cli login
```

You can download a model to the `huggingface-cache` directory using a command similar to the following (replace with the name of the model you want to download):

```bash
sudo mkdir -p /data/huggingface-cache
sudo chmod -R a+w /data/huggingface-cache
HF_HOME=/data/huggingface-cache huggingface-cli download meta-llama/Llama-3.1-405B-Instruct --exclude "original/*"
```

Alternatively, you can download the model to a specific directory, for example, if you want to quantize it using Quark:

```bash
sudo mkdir -p /data/llama-3.1
sudo chmod -R a+w /data/llama-3.1
huggingface-cli download meta-llama/Llama-3.1-405B-Instruct --exclude "original/*" --local-dir /data/llama-3.1/Llama-3.1-405B-Instruct
```

In the benchmark commands provided later in this document, replace the model name, for example, `amd/Llama-3.1-405B-Instruct-FP8-KV` with the path to the model, for example, `/data/llama-3.1/Llama-3.1-405B-Instruct`.

### Use pre-quantized models

AMD provides [FP8-quantized versions](https://huggingface.co/collections/amd/quark-quantized-ocp-fp8-models) of several models to make them easier to run, including:

- <https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV>
- <https://huggingface.co/amd/Llama-3.1-70B-Instruct-FP8-KV>
- <https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV>

Some models may be private and accessible only to members of <https://huggingface.co/amd>.

These FP8 quantized checkpoints were generated with AMD’s Quark Quantizer. For more information about Quark, see <https://quark.docs.amd.com/latest/quark_example_torch_llm_gen.html>

### Quantize your own models

This step is optional if you want to quantize your own model instead of using AMD's pre-quantized models. These instructions use Llama-3.1-405B as an example, but the commands are similar for other models.

1. Download the model from <https://huggingface.co/meta-llama/Llama-3.1-405B> to the `/data/llama-3.1` directory, as described above.

2. [Download and install Quark](https://quark.docs.amd.com/latest/install.html).

3. Run the quantization script in the example folder using the following command:

    ```bash
    # path to quark quantization script
    export QUARK_DIR=/data/quark-0.6.0+dba9ca364/examples/torch/language_modeling/llm_ptq/quantize_quark.py
    # path to Model 
    export MODEL_DIR=/data/llama-3.1/Llama-3.1-405B-Instruct
    python3 $QUARK_DIR \
    --model_dir $MODEL_DIR \
    --output_dir Llama-3.1-405B-Instruct-FP8-KV \
    --kv_cache_dtype fp8 \
    --quant_scheme w_fp8_a_fp8 \
    --num_calib_data 128 \
    --model_export quark_safetensors \
    --no_weight_matrix_merge \
    --multi_gpu
    ```

    > **Note:** The `--multi_gpu` parameter can be omitted for small models that fit on a single GPU.

## Performance testing with AMD vLLM Docker

### Performance environment variables

Some environment variables can improve the performance of the vLLM kernels on the MI300X and MI325X GPUs. See the [AMD Instinct MI300X workload optimization guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) for more information.

### vLLM engine performance settings

vLLM provides a number of engine options which can be changed to improve performance. For a complete list of vLLM engine options, see the [vLLM Engine Args](https://docs.vllm.ai/en/stable/usage/engine_args.html) documentation.

Below are some key vLLM engine arguments than can affect performance. These options can be passed to the vLLM benchmark scripts:

- **--max-model-len** : Maximum context length supported by the model instance. Setting it lower than the model configuration value can improve performance and GPU memory utilization.
- **--max-num-batched-tokens** : Maximum prefill size, for example, how many prompt tokens can be packed into a single prefill. Setting a higher value can improve prefill performance but uses more GPU memory. A value of `131072` works well for LLama models.
- **--max-num-seqs** : Maximum decode batch size (default `1024`). Larger values allow more prompts to be processed concurrently, increasing throughput but possibly raising latency. If set too high, GPU memory may be insufficient for the KV cache, causing request preemption. The optimal value depends on GPU memory, model size, and maximum context length.
- **--gpu-memory-utilization** : Ratio of GPU memory reserved for the vLLM instance. Default value is `0.9`. Increasing thiw value (up to `0.99`) provides more memory for KV cache. When running in graph mode (for example, without `--enforce-eager`), a slightly smaller value (`0.92` to `0.95`) may be needed to ensure adequate memory for the HIP graph.

### Latency benchmark

The `vllm bench latency` tool measures end-to-end latency for a specified model, input/output length, and batch size.

You can run latency tests for FP8 models with:

```bash
MODEL=amd/Llama-3.1-405B-Instruct-FP8-KV
BS=1
IN=128
OUT=2048
TP=8

vllm bench latency \
    --distributed-executor-backend mp \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --batch-size $BS \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-iters-warmup 3 \
    --num-iters 5

```

When measuring models with long context lengths, performance may improve by setting `--max-model-len` to a smaller value. However, ensure that `--max-model-len` is at least as large as the sum of the input and output token counts.

To estimate Time To First Token (TTFT) with the `vllm bench latency` tool, set the output length (`OUT`) to `1` token.  It is also recommended to use `--enforce-eager` for a more accurate measurement of the actual time to generate the first token. For a more comprehensive TTFT measurement, use the Online Serving Benchmark.

For more information about available parameters, run:

```bash
vllm bench latency -h
```

### Throughput benchmark

The `vllm bench throughput` tool measures offline throughput. It can use either an input dataset or random prompts with fixed input and output lengths.

You can run throughput tests for FP8 models with:

```bash
MODEL=amd/Llama-3.1-405B-Instruct-FP8-KV
IN=128
OUT=2048
TP=8
PROMPTS=1500
MAX_NUM_SEQS=1500

vllm bench throughput \
    --distributed-executor-backend mp \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --disable-detokenize \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --max-model-len 8192 \
    --max-num-batched-tokens 131072 \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-prompts $PROMPTS \
    --max-num-seqs $MAX_NUM_SEQS
```

For FP16/BF16 models, remove `--kv-cache-dtype fp8`.

When measuring models with long context lengths, performance may improve by setting `--max-model-len` to a smaller value (`8192` in this example). However, ensure that `--max-model-len` is at least as large as the total number of input and output tokens.

It is also important to tune vLLM’s `--max-num-seqs` parameter based on the model and input/output lengths. Larger values allow vLLM to use more GPU memory for the KV cache and process more prompts concurrently. However, if the value is too large, the KV cache might reach its capacity, causing vLLM to cancel and reprocess some prompts. Suggested values for various models and configurations are listed below.

For models that fit on a single GPU, it is usually best to run with `--tensor-parallel-size 1`. Requests can then be distributed across multiple copies of vLLM running on different GPUs. This is more efficient than running a single copy of the model with `--tensor-parallel-size 8`.

For optimal performance, the `PROMPTS` value should be a multiple of the `MAX_NUM_SEQS` value. For example, if `MAX_NUM_SEQS=1500`, then the `PROMPTS` value could be `1500`, `3000`, and so on.  If `PROMPTS` is smaller than `MAX_NUM_SEQS`, there won’t be enough prompts for vLLM to maximize concurrency.

For more information about available parameters, run:

```bash
vllm bench throughput -h
```

### Online serving benchmark

The following example benchmarks Llama-3.1-70B using 4096 input tokens, 512 output tokens, and a tensor parallelism value of 8:

```bash
vllm serve amd/Llama-3.1-70B-Instruct-FP8-KV \
    --swap-space 16 \
    --disable-log-requests \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --max-model-len 8192 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.99 \
    --num_scheduler-steps 10
```

For FP16 models, remove `--kv-cache-dtype fp8`. Change port, for example, `--port 8005`, if port 8000 is in use by another process.

Run the client in a separate terminal. Use `port_id` from the previous step or use `port-id=8000`.

```bash
vllm bench serve \
    --port 8000 \
    --model amd/Llama-3.1-70B-Instruct-FP8-KV \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 512 \
    --request-rate 1 \
    --ignore-eos \
    --num-prompts 500 \
    --percentile-metrics ttft,tpot,itl,e2el
```

Once all prompts are processed, terminate the server gracefully using Ctrl+C.

### AITER

The `rocm/vllm:latest` image comes with [AITER](https://github.com/ROCm/aiter) preinstalled, which can provide siginficant performance improvements for certain model, input/output, and batch size configurations. To disable this feature and use vLLM's Triton attention, set `VLLM_ROCM_USE_AITER=0` (the default value is `1`). For more information, see <https://docs.vllm.ai/en/latest/getting_started/quickstart.html#on-attention-backends>

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP #or INT8, INT6, INT4
vllm bench latency --model amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV -tp 8 --batch-size 256 --input-len 128 --output-len 2048
```

## Building vLLM Docker image for ROCm

To build a vLLM image correpsonding to the current `rocm/vllm:latest`, first clone the vLLM repository:

```bash
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
```

Then use the following command to build the image directly from the specified commit:

```bash
     docker build -f docker/Dockerfile.rocm \
    --build-arg REMOTE_VLLM=1 \
    --build-arg VLLM_REPO=https://github.com/ROCm/vllm \
    --build-arg VLLM_BRANCH="38f225c2abeadc04c2cc398814c2f53ea02c3c72" \
    -t vllm-rocm .
```

For further instructions on how to build an upstream vLLM docker image, see <https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-image-from-source>

## Changelog

rocm7.0.0_vllm_0.11.1_20251103:
- Support for Llama4 FP4 & Granite4 model
- vLLM version 0.11.1 RC
- Default AITER enabled

rocm7.0.0_vllm_0.10.2_20251002:
- Support for FP4 models
- GPT-OSS support
- Support for MI35x

rocm6.4.1_vllm_0.10.1_20250909:
- vLLM version 0.10.1
- Flag enabled by default in the docker `-VLLM_V1_USE_PREFILL_DECODE_ATTENTION`

20250715_aiter:
- No need to specify `--compilation-config`, options enabled by default
- Fixed llama3.1 405b CAR issue (no longer requires `--disable-custom-all-reduce`)
- Fixed `+rms_norm` custom kernel issue
- Added quick reduce (set `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP` to enable; supported modes: FP, INT8, INT6, INT4)
- Mitigated the commandr model causing GPU crash through a workaround until the driver issue is fixed

20250620_aiter:
- V1 enabled by default (use `VLLM_USE_V1=0` to override)
- Fixed detokenizers issue
- Fixed AITER MoE issues
- vLLM v0.9.1
  
20250605_aiter:
- Updated to ROCm 6.4.1 and vLLM v0.9.0.1
- AITER MHA
- IBM 3d kernel for unified attention
- Full graph capture for split attention

20250521_aiter:
- AITER V1 engine performance improvements

20250513_aiter:
- Fixed out-of-memory bug
- PyTorch fixes
- Tunable ops fixes

20250410_aiter:
- 2-stage MoE
- MLA from AITER

20250325_aiter:
- Improved DeepSeek-V3/R1 performance
- Initial Gemma-3 enablement
- Detokenizer disablement
- Torch.compile support

20250305_aiter:
- AITER improvements
- Support for FP8 skinny GEMM

20250207_aiter:
- Improved AITER performance
- Bug fixes

20250205_aiter:
- [AITER](https://github.com/ROCm/aiter) support
- Performance improvement for custom paged attention
- Fixed reduced memory overhead bug

20250124:
- Fixed accuracy issues with 405B FP8 Triton FA
- Fixed accuracy issue with TP8

20250117:
- [Experimental DeepSeek-V3 and DeepSeek-R1 support](#running-deepseek-v3-and-deepseek-r1)
