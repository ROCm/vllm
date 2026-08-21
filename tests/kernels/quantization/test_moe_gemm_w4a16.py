# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the gfx11 W4A16 MoE prefill WMMA GEMM.

Validates the compiled op ``torch.ops._rocm_C.moe_gemm_w4a16`` (gated behind
``VLLM_MOE_HIP`` in production, called directly here) against the Triton
reference ``invoke_fused_moe_kernel_hybrid_triton`` on identical inputs, using
the vLLM MoE weight layout (ExLlama shuffle-packed INT4).

Run ``pytest tests/kernels/quantization/test_moe_gemm_w4a16.py``.
"""

import pytest
import torch

# The op is compiled into _rocm_C on all ROCm builds but has a real body only on
# gfx11 (stub elsewhere). Skip cleanly off ROCm, off gfx11, or if the op is
# absent (older build). Probe op presence before importing rocm.py so CUDA skips
# without touching ROCm-only platform code.
try:
    import vllm._rocm_C  # noqa: F401

    _have_op = hasattr(torch.ops._rocm_C, "moe_gemm_w4a16")
except Exception:
    _have_op = False
if not _have_op:
    pytest.skip("_rocm_C.moe_gemm_w4a16 not available", allow_module_level=True)

from vllm.platforms.rocm import on_gfx11

if not on_gfx11():
    pytest.skip("requires gfx11 (RDNA3 WMMA)", allow_module_level=True)

pytest.importorskip("triton")

from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    pack_int4_exllama_shuffle,
)
from vllm.model_executor.layers.fused_moe import moe_hip_w4a16  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_kernel_hybrid_triton,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl

DEV = "cuda"


@pytest.mark.parametrize("block_m", [32])
@pytest.mark.parametrize(
    "K, N, TOPK",
    [
        (2048, 1024, 8),  # gemm1 family, tuned Qwen3.6 shape
        (1024, 512, 8),  # gemm1 family, different K/N -> exercises runtime K,N
        (512, 2048, 1),  # gemm2 family, tuned Qwen3.6 shape
    ],
)
def test_moe_gemm_w4a16_matches_triton(K, N, TOPK, block_m, monkeypatch):
    # The rdna_moe_gemm kernel is default-on under gfx11, so force it off here -- the
    # reference below dispatches through invoke_fused_moe_kernel_hybrid_triton
    # and must stay on the Triton path to be a real reference.
    monkeypatch.setenv("VLLM_MOE_HIP", "0")

    E, G = 256, 128
    T = 994
    dt = torch.bfloat16
    torch.manual_seed(0)

    A = torch.randn(T, K, dtype=dt, device=DEV) * 0.1
    w_uint4 = torch.randint(0, 16, (E, N, K), dtype=torch.int32, device=DEV)
    w_packed = torch.stack([pack_int4_exllama_shuffle(w_uint4[e]) for e in range(E)])
    w_scale = torch.randn(E, N, K // G, dtype=dt, device=DEV).abs() * 0.01

    logits = torch.randn(T, E, device=DEV)
    _, topk_ids = torch.topk(logits.softmax(-1), TOPK, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, E, ignore_invalid_experts=True
    )
    num_slots = sorted_token_ids.size(0)

    # Triton reference. VLLM_MOE_HIP=0 (set above) makes the in-kernel dispatch
    # predicate False so this stays on the Triton path.
    assert not moe_hip_w4a16.is_enabled(), (
        "VLLM_MOE_HIP must be forced off so the reference uses Triton"
    )
    c_ref = torch.zeros(num_slots, N, dtype=dt, device=DEV)
    invoke_fused_moe_kernel_hybrid_triton(
        A=A,
        B=w_packed,
        C=c_ref,
        B_scale=w_scale,
        topk_weights=None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=TOPK,
        config=dict(
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=4,
            num_warps=2,
            num_stages=1,
        ),
        compute_type=tl.bfloat16,
        group_size=G,
        align_block_size_m=block_m,
    )

    # rdna_moe_gemm compiled op (called directly; expert_ids.numel() == num blocks).
    # The op returns nothing and mutates c in place; an unsupported shape would
    # TORCH_CHECK rather than return a status.
    c = torch.zeros(num_slots, N, dtype=dt, device=DEV)
    torch.ops._rocm_C.moe_gemm_w4a16(
        A,
        w_packed,
        w_scale,
        sorted_token_ids,
        expert_ids,
        c,
        T * TOPK,
        TOPK,
        block_m,
        expert_ids.numel(),
    )
    torch.accelerator.synchronize()

    rel = (
        (c.float() - c_ref.float()).abs().sum() / (c_ref.float().abs().sum() + 1e-9)
    ).item()
    assert rel < 0.01, f"rel={rel:.4f} exceeds tolerance vs Triton reference"
