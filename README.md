# vLLM development fork for AMD/gfx11

This is the development staging fork of [vLLM](https://github.com/vllm-project/vllm). The [gfx11](https://github.com/ROCm/vllm/tree/gfx11) branch contains optimizations for AMD's gfx11XX architectures and should be considered under development. For stable releases, use the [official repo](https://github.com/vllm-project/vllm).

Everything not covered below - usage, supported models, docs - is unchanged from upstream; see the [upstream README](https://github.com/vllm-project/vllm/blob/main/README.md).

## Relationship to upstream

This fork is a staging area, not a long-term alternative to vLLM. We upstream our gfx11XX optimizations periodically, so they may take a while to reach an official stable release - but that is where they are headed. In the meantime we regularly merge upstream changes back into this branch, so you get new vLLM features alongside the gfx11XX work rather than having to choose between them.

---

## Getting started

### Install

We recommend the nightly wheels built by this branch's CI:

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install \
    vllm \
    "torch[device-gfx1151]" \
    "torchvision[device-gfx1151]" \
    torchaudio \
    triton \
    "rocm[device-gfx1151]" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://nightly.repo.amd.com/rocm/whl-next/ \
    --extra-index-url https://rocm.frameworks-nightlies.amd.com/whl-multi-arch-staging/vllm-gfx11-dev/ \
    --index-strategy unsafe-first-match \
    --prerelease allow
```

For benchmarking, pin the versions so runs stay reproducible. The following example installs specific versions (that were available at the time this was written):

```bash
uv pip install \
    "vllm==0.26.1rc1.dev1701+gc84454ff4.rocm101.rdna3.prototype" \
    "torch[device-gfx1151]==2.13.0+rocm10.1.0a20260903" \
    "torchvision[device-gfx1151]==0.28.0+rocm10.1.0a20260903" \
    "torchaudio==2.11.0+rocm10.1.0a20260903" \
    "triton==3.8.0+git4cff872c.rocm10.1.0a20260903" \
    "rocm[device-gfx1151]==10.1.0a20260903" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://nightly.repo.amd.com/rocm/whl-next/ \
    --extra-index-url https://rocm.frameworks-nightlies.amd.com/whl-multi-arch-staging/vllm-gfx11-dev/ \
    --index-strategy unsafe-first-match \
    --prerelease allow
```

### Point PYTHONPATH at `amdsmi`

vLLM detects the GPU through `amdsmi`, which the `rocm` wheel ships outside of `site-packages`. Without it on `PYTHONPATH`, every `vllm` command fails with `RuntimeError: Failed to infer device type`. Export this in each shell that runs vLLM:

```bash
export PYTHONPATH="$(python -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "share", "amd_smi"))')${PYTHONPATH:+:$PYTHONPATH}"
```

### Serve a model

To sanity check the install, serve a small model:

```bash
vllm serve Qwen/Qwen3-0.6B
```

Once it reports `Application startup complete`, query it from another shell:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Give me a short introduction to large language models."}],
        "max_tokens": 60
    }'
```

The first start is slow - it downloads weights and compiles kernels - so allow a couple of minutes before assuming something is wrong.

From here, usage is standard vLLM; see the [upstream quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html).

### Other installation options

- Stable vLLM on Strix Halo: [AMD documentation](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html?fam=ryzen&gpu=max-plus-pro-495&rocm-ver=10.0.0&vllm-ver=0.27&i=pip&w=compute&gfx=gfx1151).
- Development: [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

## Contributing

Contributions are welcome. Follow the upstream [contributing guide](https://github.com/vllm-project/vllm/blob/main/CONTRIBUTING.md); target the [gfx11](https://github.com/ROCm/vllm/tree/gfx11) branch for gfx11XX-specific work.
