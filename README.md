This is the development staging fork of [vLLM](https://github.com/vllm-project/vllm). The [gfx11](https://github.com/ROCm/vllm/tree/gfx11) branch contains optimizations for AMD's gfx11XX architectures and should be considered under development. For stable releases, use the official repo.

Everything not covered below - usage, supported models, docs - is unchanged from upstream; see the [upstream README](https://github.com/vllm-project/vllm/blob/main/README.md).

---

## Getting started

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
    --index-url https://rocm.frameworks-nightlies.amd.com/whl-multi-arch-staging/vllm-gfx11-dev/ \
    --extra-index-url https://nightly.repo.amd.com/rocm/whl-next/ \
    --extra-index-url https://pypi.org/simple/ \
    --index-strategy unsafe-first-match \
    --prerelease allow
```

`unsafe-first-match` resolves each package from the first index that provides it, so keep the AMD indexes ahead of PyPI — otherwise you get the generic CUDA builds of `vllm` and `torch`.

For benchmarking, pin the versions so runs stay reproducible. As an example you can pin versions by running:

```bash
uv pip install \
    "vllm==0.26.1rc1.dev1701+gc84454ff4.rocm101.rdna3.prototype" \
    "torch[device-gfx1151]==2.13.0+rocm10.1.0a20260903" \
    "torchvision[device-gfx1151]==0.28.0+rocm10.1.0a20260903" \
    "torchaudio==2.11.0+rocm10.1.0a20260903" \
    "triton==3.8.0+git4cff872c.rocm10.1.0a20260903" \
    "rocm[device-gfx1151]==10.1.0a20260903" \
    --index-url https://rocm.frameworks-nightlies.amd.com/whl-multi-arch-staging/vllm-gfx11-dev/ \
    --extra-index-url https://nightly.repo.amd.com/rocm/whl-next/ \
    --extra-index-url https://pypi.org/simple/ \
    --index-strategy unsafe-first-match \
    --prerelease allow
```

Alternatives:

- Stable vLLM on Strix Halo: [AMD documentation](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html?fam=ryzen&gpu=max-plus-pro-495&rocm-ver=10.0.0&vllm-ver=0.27&i=pip&w=compute&gfx=gfx1151).
- Development: [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).



## Contributing

Contributions are welcome. Follow the upstream [contributing guide](https://github.com/vllm-project/vllm/blob/main/CONTRIBUTING.md); target the `gfx11` branch for gfx11XX-specific work.