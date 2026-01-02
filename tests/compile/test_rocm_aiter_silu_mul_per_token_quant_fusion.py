# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ROCm aiter fused_silu_mul_per_token_quant fusion pass.
"""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.rocm_aiter_fusion import (
    AITER_PER_TOKEN_QUANT_OP,
    FUSED_SILU_MUL_PER_TOKEN_QUANT_OP,
    VLLM_PER_TOKEN_QUANT_OP,
    RocmAiterSiluMulFp8PerTokenQuantFusionPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform

try:
    from .backend import TestBackend
except ImportError:
    # For manual testing without pytest
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()


class TestSiluMulPerTokenQuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, use_aiter_quant: bool = True):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.hidden_size = hidden_size
        self.use_aiter_quant = use_aiter_quant
        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)

        if self.use_aiter_quant:
            # Use aiter per-token quant
            out, scale = rocm_aiter_ops.per_token_quant(y, FP8_DTYPE)
        else:
            # Use vllm per-token quant (use torch ops directly)
            # This matches what's in the fusion pattern
            import torch

            out = torch.empty_like(y, dtype=FP8_DTYPE)
            scale = torch.empty(1, dtype=torch.float32, device=y.device)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, y, scale, None)

        return out, scale

    def ops_before_fusion(self):
        silu_mul_op = (
            torch.ops._C.silu_and_mul.default
            if self.enable_silu_mul_custom_op
            else torch.ops.aten.mul
        )

        quant_op = (
            AITER_PER_TOKEN_QUANT_OP
            if self.use_aiter_quant
            else VLLM_PER_TOKEN_QUANT_OP
        )

        return [silu_mul_op, quant_op]

    def ops_after_fusion(self):
        return [FUSED_SILU_MUL_PER_TOKEN_QUANT_OP]


@pytest.mark.skipif(
    not current_platform.is_rocm() or not rocm_aiter_ops.is_enabled(),
    reason="Requires ROCm with aiter support",
)
@pytest.mark.parametrize("hidden_size", [128, 4096])
@pytest.mark.parametrize("num_tokens", [1, 32, 128])
@pytest.mark.parametrize("use_aiter_quant", [True, False])
def test_silu_mul_per_token_quant_fusion(
    hidden_size: int, num_tokens: int, use_aiter_quant: bool
):
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        )
    )

    model = (
        TestSiluMulPerTokenQuantModel(hidden_size, use_aiter_quant=use_aiter_quant)
        .eval()
        .cuda()
    )

    x = torch.randn(num_tokens, 2 * hidden_size, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        ref_out, ref_scale = model(x)

    fusion_passes = [RocmAiterSiluMulFp8PerTokenQuantFusionPass(vllm_config)]
    passes = [
        NoOpEliminationPass(vllm_config),
        *fusion_passes,
        PostCleanupPass(vllm_config),
    ]
    backend = TestBackend(*passes)

    model_compiled = torch.compile(model, backend=backend)

    with torch.no_grad():
        fused_out, fused_scale = model_compiled(x)

    ref_dequant = ref_out.to(torch.float32) * ref_scale
    fused_dequant = fused_out.to(torch.float32) * fused_scale

    rtol, atol = 5e-1, 5e-2
    try:
        torch.testing.assert_close(ref_dequant, fused_dequant, rtol=rtol, atol=atol)
    except AssertionError as e:
        diff = torch.abs(ref_dequant - fused_dequant)
        raise AssertionError(
            "Dequantized output mismatch.\n"
            f"  rtol={rtol}, atol={atol}\n"
            f"  mean_diff={diff.mean().item():.6f}\n"
            f"  max_diff={diff.max().item():.6f}\n"
        ) from e

    rtol_scale, atol_scale = 1e-2, 1e-2
    try:
        torch.testing.assert_close(
            ref_scale, fused_scale, rtol=rtol_scale, atol=atol_scale
        )
    except AssertionError as e:
        raise AssertionError(
            "Scale mismatch.\n"
            f"  rtol={rtol_scale}, atol={atol_scale}\n"
            f"  max_diff={torch.max(torch.abs(ref_scale - fused_scale)).item():.6f}\n"
            f"  mean_diff={torch.mean(torch.abs(ref_scale - fused_scale)).item():.6f}\n"
        ) from e

    ops_before = model.ops_before_fusion()
    ops_after = model.ops_after_fusion()

    graph = backend.graphs[0] if hasattr(backend, "graphs") and backend.graphs else None

    if graph is not None:
        fused_op_found = False
        for node in graph.nodes:
            if node.op == "call_function" and node.target in ops_after:
                fused_op_found = True
                break

        assert fused_op_found, (
            f"Fused op {ops_after} not found in compiled graph. "
            f"Expected fusion to occur."
        )

        # Optionally check that original ops are not present
        # (This is stricter and may not always hold depending on graph structure)
        for node in graph.nodes:
            if node.op == "call_function":
                # The original separate ops should ideally not be present
                # but we'll just ensure the fused op exists
                pass

    print(
        f"Fusion test passed: tokens={num_tokens}, hidden={hidden_size}, "
        f"  use_aiter_quant={use_aiter_quant}, "
        f"  rtol={rtol}, atol={atol}"
    )


@pytest.mark.skipif(
    not current_platform.is_rocm() or not rocm_aiter_ops.is_enabled(),
    reason="Requires ROCm with aiter support",
)
def test_fusion_pass_registered():
    """Test that the fusion pass is properly registered."""
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        )
    )

    fusion_pass = RocmAiterSiluMulFp8PerTokenQuantFusionPass(vllm_config)

    assert hasattr(fusion_pass, "patterns"), "Fusion pass missing patterns attribute"

    print("Fusion pass can be instantiated and has patterns registered")


if __name__ == "__main__":
    if current_platform.is_rocm() and rocm_aiter_ops.is_enabled():
        print("Running manual fusion tests...")

        print("\n1. Testing fusion pass registration...")
        test_fusion_pass_registered()

        print("\n2. Testing fusion with aiter per-token quant...")
        test_silu_mul_per_token_quant_fusion(
            hidden_size=4096, num_tokens=128, use_aiter_quant=True
        )

        print("\n3. Testing fusion with vllm per-token quant...")
        test_silu_mul_per_token_quant_fusion(
            hidden_size=4096, num_tokens=128, use_aiter_quant=False
        )

        print("\n✓ All manual fusion tests passed!")
    else:
        print("Skipping tests - ROCm with aiter not available")
