# Verification scripts for the ROCM_FLASHINFER backend

Run these inside a ROCm container with `amd-flashinfer` and vLLM installed
(see `../flashinfer-as-vllm-backend.md` for how that image is built). All were
developed against MI300X / gfx942.

| script | what it establishes |
|---|---|
| `patch_use_rocm.py` | **Run first on torch ≥2.11.** Adds `-DUSE_ROCM` to flashinfer's hipcc flags; without it every JIT compile fails. Idempotent. |
| `probe.py` | Full prefill/decode × fp16/bf16 × {auto, fa2, aiter} matrix against a torch SDPA reference. This is what found the fa2 prefill bug. |
| `probe2.py` | Localizes that bug: paged vs ragged vs single entry points, page sizes 1/16/32/64, exact vs partial last page. |
| `probe3.py` | Tests and refutes the "causal mask ignored" explanation for it. |
| `gate_phase1.py` | Backend registration: enum path, class attrs, KV shape/stride, `validate_configuration` rejections, env var, platform priority wiring. 30 checks. |
| `gate_phase2.py` | End-to-end generation, and greedy-token comparison between two backends. Reports the top-2 logprob gap at any divergence so a tie can be told apart from a defect. |
| `gate_phase3.py` | Drives vLLM's own `_test_backend_correctness` (SDPA reference) directly, because the packaged test parametrizes over a gated model. |

## Typical run

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32g \
  -v "$PWD":/vllm -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /vllm -e HF_HUB_OFFLINE=1 --entrypoint bash <image> -c '
    python3 plan/scripts/gate_phase3.py TinyLlama/TinyLlama-1.1B-Chat-v1.0'
```

`gate_phase2.py` and `gate_phase3.py` take an optional model argument and
default to TinyLlama, which is small enough to run beside other workloads.
`gate_phase2.py` also takes a list of backends to compare, e.g.
`... gate_phase2.py <model> ROCM_FLASHINFER ROCM_AITER_FA`.

**Re-run `probe.py` against any new amd-flashinfer build before trusting
prefill** — the backend pins `backend="aiter"` specifically because the default
route is wrong, and a version bump could change routing silently.
