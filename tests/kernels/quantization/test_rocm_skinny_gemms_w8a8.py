# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the W8A8 INT8 skinny GEMM kernel (wvSplitK_w8a8).

This kernel handles int8 weights x int8 activations with:
- Per-channel weight scale (fp16/bf16)
- Per-tensor activation scale (float32)
- Optional bias
"""

import math

import pytest
import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units

DTYPES = [torch.bfloat16, torch.float16]
BIAS_MODES = [0, 1, 2]  # 0=no bias, 1=per-output [M], 2=per-batch [N,M]
SEEDS = [0]

# (N, K, M) test shapes: N=batch, K=inner dim, M=output features
# K must be divisible by 16, M must be divisible by YTILE (1 or 4)
NKM_FACTORS = [
    # Basic shapes
    (1, 32, 16),
    (1, 64, 64),
    (1, 128, 256),
    (1, 256, 512),
    (1, 512, 1024),
    # Typical LLM decode shapes
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    # Multiple batch sizes
    (2, 256, 256),
    (2, 4096, 4096),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (5, 2048, 2048),
    # Extended K values
    (1, 9216, 512),
    (2, 10240, 1024),
    # Larger K (tests LDS capacity, int8 allows 2x vs fp16)
    (1, 16384, 1024),
    (2, 16384, 1024),
    (1, 32768, 1024),
]


def ref_w8a8_gemm(
    w_int8: torch.Tensor,
    a_int8: torch.Tensor,
    w_scale: torch.Tensor,
    a_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference implementation: dequantize and matmul.

    Args:
        w_int8: [M, K] int8 weights
        a_int8: [N, K] int8 activations
        w_scale: [M] per-channel weight scale (fp16/bf16)
        a_scale: scalar per-tensor activation scale (float32)
        bias: optional bias
        out_dtype: output dtype (fp16 or bf16)

    Returns:
        [N, M] output in out_dtype
    """
    # Dequantize to float32 for reference accuracy
    w_f32 = w_int8.float() * w_scale.float().unsqueeze(1)  # [M, K]
    a_f32 = a_int8.float() * a_scale.float()  # [N, K]

    # Matmul: [N, K] x [K, M] -> [N, M]
    out = torch.mm(a_f32, w_f32.t())

    if bias is not None:
        out = out + bias.float()

    return out.to(out_dtype)


def quantize_symmetric(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel int8 quantization.

    Args:
        tensor: [rows, cols] float tensor

    Returns:
        quantized: [rows, cols] int8
        scale: [rows] float32 per-channel scale
    """
    amax = tensor.abs().amax(dim=1)
    scale = amax / 127.0
    scale = scale.clamp(min=1e-10)
    quantized = (tensor / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def quantize_per_tensor(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor int8 quantization.

    Args:
        tensor: float tensor

    Returns:
        quantized: int8 tensor
        scale: scalar float32 scale
    """
    amax = tensor.abs().max()
    scale = amax / 127.0
    scale = scale.clamp(min=1e-10)
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.reshape(1)


@pytest.mark.parametrize("xnorm", [False, True])
@pytest.mark.parametrize("n,k,m", NKM_FACTORS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_wvsplitk_w8a8_kernel(xnorm, n, k, m, dtype, seed, bias_mode):
    torch.manual_seed(seed)
    cu_count = num_compute_units()

    xavier = math.sqrt(2 / k) if xnorm else 1

    # Generate random fp16 data, then quantize to int8
    W_fp = (torch.rand(m, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier
    A_fp = (torch.rand(n, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier

    # Quantize weights per-channel, activations per-tensor
    W_int8, w_scale = quantize_symmetric(W_fp)
    A_int8, a_scale = quantize_per_tensor(A_fp)

    # Convert weight scale to output dtype
    w_scale_typed = w_scale.to(dtype)

    BIAS = None
    if bias_mode == 1:
        BIAS = (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1) * xavier
    elif bias_mode == 2:
        BIAS = (torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1) * xavier

    # Reference: dequantize and matmul in float32
    ref_out = ref_w8a8_gemm(W_int8, A_int8, w_scale_typed, a_scale, BIAS, dtype)

    # Kernel under test
    out = ops.wvSplitK_w8a8(W_int8, A_int8, w_scale_typed, a_scale, cu_count, BIAS)

    if xnorm:
        atol = max(1e-3, torch.finfo(dtype).eps * math.sqrt(k))
        torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-2)
    else:
        # Accumulation error scales with sqrt(K) for fp16
        atol = torch.finfo(dtype).eps * math.sqrt(k)
        torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-2)
