# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import envs
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.compilation.activation_quant_fusion import (
    SILU_MUL_OP,
    ActivationQuantFusionPass,
)
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.rocm_aiter_fusion import (
    RocmAiterSiluMulFp8PerTokenQuantFusionPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    dispatch_w8a8_scaled_mm,
    maybe_create_device_identity,
)
from vllm.platforms import current_platform

from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()

AITER_PER_TOKEN_QUANT_OP = torch.ops.vllm.rocm_aiter_per_token_quant.default
VLLM_PER_TOKEN_QUANT_OP = torch.ops._C.dynamic_per_token_scaled_fp8_quant.default
FUSED_SILU_MUL_PER_TOKEN_QUANT_OP = (
    torch.ops.vllm.rocm_aiter_fused_silu_mul_per_token_quant.default
)


class TestSiluMulPerTokenQuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, use_aiter_quant: bool = True, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.hidden_size = hidden_size
        self.use_aiter_quant = use_aiter_quant
        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()

        weight_bf16 = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16)
        weight_absmax = torch.max(torch.abs(weight_bf16), dim=0, keepdim=True)[
            0
        ]  # [1, hidden_size]
        fp8_max = torch.finfo(FP8_DTYPE).max
        self.weight_scale = (
            (weight_absmax / fp8_max).clamp(min=1e-12).to(torch.float32).t()
        )
        self.weight = (weight_bf16 / weight_absmax).to(FP8_DTYPE).t()

    def forward(self, x):
        y = self.silu_and_mul(x)

        if self.use_aiter_quant:
            out, scale_a = rocm_aiter_ops.per_token_quant(y, FP8_DTYPE)
        else:
            from vllm._custom_ops import scaled_fp8_quant

            out, scale_a = scaled_fp8_quant(y, use_per_token_if_dynamic=True)

        # Use _scaled_mm to skip shuffling for testing
        w8a8_scaled_mm = dispatch_w8a8_scaled_mm(
            preferred_backend="torch",
            per_tensor_weights=False,
            per_tensor_activations=False,
        )
        num_tokens = x.shape[0]
        result = w8a8_scaled_mm(
            qinput=out,
            weight=self.weight,
            scale_a=scale_a,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
            bias=None,
            output_shape=[num_tokens, self.hidden_size],
        )

        return result

    def ops_in_model_before(self):
        silu_mul_op = (
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul
        )

        quant_op = (
            AITER_PER_TOKEN_QUANT_OP
            if self.use_aiter_quant
            else VLLM_PER_TOKEN_QUANT_OP
        )

        return [silu_mul_op, quant_op]

    def ops_in_model_after(self):
        return [FUSED_SILU_MUL_PER_TOKEN_QUANT_OP]


@pytest.mark.parametrize("num_tokens", [32, 128, 1024])
@pytest.mark.parametrize(
    "hidden_size", [256, 4096]
)  # Minimum 256 required for aiter fused kernel (vec_size >= 4)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("enable_silu_mul_custom_op", [True, False])
@pytest.mark.parametrize("use_aiter_quant", [True, False])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["rocm"] or not IS_AITER_FOUND,
    reason="Only test on ROCm with aiter support",
)
def test_fusion_silu_and_mul_per_token_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    enable_silu_mul_custom_op: bool,
    use_aiter_quant: bool,
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    maybe_create_device_identity()

    x = torch.rand(num_tokens, hidden_size * 2)

    custom_ops = []
    if enable_silu_mul_custom_op:
        custom_ops.append("+silu_and_mul")

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        ),
    )

    with set_current_vllm_config(config):
        fusion_passes = [ActivationQuantFusionPass(config)]
        if IS_AITER_FOUND:
            fusion_passes += [RocmAiterSiluMulFp8PerTokenQuantFusionPass(config)]

        passes = [NoOpEliminationPass(config), *fusion_passes, PostCleanupPass(config)]
        backend = TestBackend(*passes)

        model = TestSiluMulPerTokenQuantModel(
            hidden_size=hidden_size, use_aiter_quant=use_aiter_quant
        )

        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)
        result2 = model2(x)

        atol, rtol = 2e-2, 5e-2
        torch.testing.assert_close(result, result2, atol=atol, rtol=rtol)

        assert sum([p.matched_count for p in fusion_passes]) == 1

        backend.check_before_ops(model.ops_in_model_before())

        backend.check_after_ops(model.ops_in_model_after())
