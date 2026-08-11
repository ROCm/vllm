"""Phase 3 layer 1: drive vLLM's SDPA-reference attention harness directly.

The packaged test parametrizes over gated meta-llama/Meta-Llama-3-8B; this box
is offline, so drive the same `_test_backend_correctness` helper with a locally
cached model instead. Same reference, same paged-cache construction, same
tolerances -- just a different config and no pytest parametrization.

Usage: python3 gate_phase3.py [MODEL]
"""

import os
import pathlib
import sys
import traceback

os.environ.setdefault("VLLM_ROCM_USE_FLASHINFER", "1")

# Derive the repo root from this file's location rather than hardcoding a
# mount point: the checkout lives at /vllm when bind-mounted for development
# and at /vllm-workspace inside the image built by
# docker/Dockerfile.rocm_flashinfer.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))


from tests.v1.attention.test_attention_backends import (  # noqa: E402
    BATCH_SPECS,
    _test_backend_correctness,
)
from vllm.config import VllmConfig  # noqa: E402
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SPECS = [
    "single_decode",
    "small_decode",
    "medium_decode",
    "large_decode",
    "single_prefill",
    "small_prefill",
    "medium_prefill",
    # "large_prefill" needs max_model_len 4096; TinyLlama caps at 2048.
    "mixed_small",
    "mixed_medium",
]

BACKENDS = [
    AttentionBackendEnum.ROCM_FLASHINFER,
    AttentionBackendEnum.TRITON_ATTN,
]


def causal_mask_mod(b, h, q_idx, kv_idx, *, context_len):
    return (q_idx + context_len) >= kv_idx


def main():
    # _test_backend_correctness expects an active VllmConfig context, which the
    # packaged test gets from the `default_vllm_config` fixture.
    from vllm.config import set_current_vllm_config

    results = {}
    for spec in SPECS:
        for backend in BACKENDS:
            label = f"{spec:<16} {backend.name}"
            try:
                with set_current_vllm_config(VllmConfig()):
                    _test_backend_correctness(
                        BATCH_SPECS[spec],
                        MODEL,
                        [backend],
                        causal_mask_mod,
                    )
                results[label] = "PASS"
            except AssertionError as e:
                first = str(e).strip().splitlines()[0][:150]
                results[label] = f"FAIL  {first}"
            except Exception as e:
                results[label] = f"ERROR {type(e).__name__}: {str(e)[:150]}"
                if os.environ.get("VERBOSE"):
                    traceback.print_exc()
            print(f"  {label:<40} {results[label]}", flush=True)

    print("\n================ SUMMARY ================")
    npass = sum(1 for v in results.values() if v == "PASS")
    for k, v in results.items():
        print(f"  {k:<40} {v}")
    print(f"\n  {npass}/{len(results)} passed")


if __name__ == "__main__":
    main()
