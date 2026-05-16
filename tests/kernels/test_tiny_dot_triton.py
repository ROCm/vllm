# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the tiny-dot Triton fast path.

The shared_expert_gate = Linear(hidden, 1) in Qwen2/3/3.5 MoE flows
through `rocm_unquantized_gemm_impl` with m == 1, n in {1, ..., 8}.
When the preconditions (K<=4096, bias is None, bf16/fp16 input) match,
the implementation routes through a small Triton kernel
(_tiny_dot_triton in vllm/model_executor/layers/utils.py) instead of
the eager `(x*w).sum(dtype=x.dtype)` chain.

  M=1: legacy decode shared_expert_gate (single token).
  M>1: MTP-verify path (num_speculative_tokens+1 rows per call).

When followed by a sigmoid (the actual Qwen Phase-2 path), the kernel
folds it via APPLY_SIGMOID -- tested here via the public helper
`tiny_sigmoid_dot` and via `apply_sigmoid=True` directly.
"""
from __future__ import annotations

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Triton kernel requires CUDA/ROCm")
@pytest.mark.parametrize("M", [1, 3, 8])
@pytest.mark.parametrize("K", [32, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("apply_sigmoid", [False, True])
def test_tiny_dot_matches_eager(M: int, K: int, dtype: torch.dtype, apply_sigmoid: bool):
    """_tiny_dot_triton(x_flat, w_flat, apply_sigmoid=?, M=?) matches eager.

    For M==1 the return is 0-D; for M>1 it is 1-D [M].  Reference is the
    same per-row reduction the dispatch's eager fallback uses.
    """
    from vllm.model_executor.layers.utils import _tiny_dot_triton
    if _tiny_dot_triton is None:
        pytest.skip("Triton not available")

    torch.manual_seed(0)
    x_2d = (torch.randn(M, K, dtype=dtype, device="cuda") * 0.05).contiguous()
    w = (torch.randn(K, dtype=dtype, device="cuda") * 0.05).contiguous()

    # Eager reference: per-row dot.
    ref = (x_2d * w).sum(dim=-1, dtype=x_2d.dtype)
    if apply_sigmoid:
        ref = torch.sigmoid(ref)
    # M==1: kernel returns 0-D scalar; squeeze the reference to match.
    if M == 1:
        ref = ref.squeeze(0)

    got = _tiny_dot_triton(x_2d.reshape(-1), w, apply_sigmoid=apply_sigmoid, M=M)

    assert got.shape == ref.shape, f"shape mismatch: got {got.shape}, ref {ref.shape}"
    assert torch.allclose(got, ref, atol=5e-3, rtol=1e-2), (
        f"M={M} K={K} {dtype} sigmoid={apply_sigmoid}: "
        f"max abs diff {(got.float() - ref.float()).abs().max().item():.2e}"
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Triton kernel requires CUDA/ROCm")
@pytest.mark.parametrize("K", [1024, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tiny_sigmoid_dot_helper(K: int, dtype: torch.dtype):
    """Public helper tiny_sigmoid_dot(x, weight) matches the eager
    3-launch chain (mul + sum + sigmoid) used at the shared_expert_gate
    call site in qwen2_moe.py."""
    from vllm.model_executor.layers.utils import tiny_sigmoid_dot

    torch.manual_seed(0)
    # Mirror the call site: x is 1-D hidden, weight is [1, hidden].
    x = (torch.randn(K, dtype=dtype, device="cuda") * 0.05).contiguous()
    weight = (torch.randn(1, K, dtype=dtype, device="cuda") * 0.05).contiguous()

    ref = torch.sigmoid((x * weight.reshape(-1)).sum(dtype=x.dtype))
    got = tiny_sigmoid_dot(x, weight)

    assert torch.allclose(got, ref, atol=5e-3, rtol=1e-2), (
        f"K={K} {dtype}: got {got.item():.4e}, ref {ref.item():.4e}, "
        f"abs diff {(got.float() - ref.float()).abs().item():.2e}"
    )
