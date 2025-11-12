# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test for torch_channelwise_w8a8_scaled_mm narrow bug fix."""

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    maybe_create_device_identity,
    torch_channelwise_w8a8_scaled_mm,
)
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()


def _reference_impl(
    qinput: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference implementation using explicit dequantization."""
    input_fp32 = qinput.to(torch.float32) * scale_a
    # weight is (hidden_size, out_size), scale_b is (out_size, 1)
    # We need to transpose scale_b to (1, out_size) for proper broadcasting
    weight_fp32 = weight.to(torch.float32) * scale_b.t()
    output = torch.matmul(input_fp32, weight_fp32)
    if bias is not None:
        output = output + bias
    return output.to(out_dtype)


@pytest.mark.parametrize("batch_size", [1, 16, 72])
@pytest.mark.parametrize("use_per_token_scale", [False, True])
@pytest.mark.parametrize("hidden_size,out_size", [(128, 256), (256, 512)])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
    reason="FP8 requires CUDA or ROCm",
)
def test_torch_channelwise_narrow_fix(
    batch_size: int,
    use_per_token_scale: bool,
    hidden_size: int,
    out_size: int,
):
    """Test the narrow operation fix for per-tensor vs per-token scales."""
    torch.manual_seed(42 + batch_size)
    current_platform.seed_everything(42 + batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    maybe_create_device_identity()

    qinput = torch.randn(batch_size, hidden_size, device=device).to(FP8_DTYPE)
    weight = torch.randn(out_size, hidden_size, device=device).to(FP8_DTYPE).t()

    if use_per_token_scale:
        scale_a = torch.rand(batch_size, 1, dtype=torch.float32, device=device) + 0.5
    else:
        scale_a = torch.rand(1, dtype=torch.float32, device=device) + 0.5

    scale_b = torch.rand(out_size, 1, dtype=torch.float32, device=device) + 0.5

    output = torch_channelwise_w8a8_scaled_mm(
        qinput=qinput,
        weight=weight,
        out_dtype=torch.float16,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=None,  # type: ignore
        output_shape=[batch_size, out_size],
    )

    assert output.shape == (batch_size, out_size)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert output.abs().max() > 0


@pytest.mark.parametrize("batch_size", [8, 64])
@pytest.mark.parametrize("use_per_token_scale", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
    reason="FP8 requires CUDA or ROCm",
)
def test_numerical_correctness(
    batch_size: int,
    use_per_token_scale: bool,
    use_bias: bool,
):
    """Test numerical correctness against reference implementation."""
    torch.manual_seed(42)
    current_platform.seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    maybe_create_device_identity()

    hidden_size = 128
    out_size = 256

    qinput = torch.randn(batch_size, hidden_size, device=device).to(FP8_DTYPE)
    weight = torch.randn(out_size, hidden_size, device=device).to(FP8_DTYPE).t()

    if use_per_token_scale:
        scale_a = torch.rand(batch_size, 1, dtype=torch.float32, device=device) + 0.5
    else:
        scale_a = torch.rand(1, dtype=torch.float32, device=device) + 0.5

    scale_b = torch.rand(out_size, 1, dtype=torch.float32, device=device) + 0.5
    bias = (
        torch.randn(out_size, dtype=torch.float16, device=device) if use_bias else None
    )

    output = torch_channelwise_w8a8_scaled_mm(
        qinput=qinput,
        weight=weight,
        out_dtype=torch.float16,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,  # type: ignore
        output_shape=[batch_size, out_size],
    )

    expected = _reference_impl(qinput, weight, scale_a, scale_b, bias, torch.float16)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 72])
@pytest.mark.parametrize("use_per_token_scale", [False, True])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
    reason="FP8 requires CUDA or ROCm",
)
def test_with_bias(batch_size: int, use_per_token_scale: bool):
    """Test that bias is correctly applied."""
    torch.manual_seed(42)
    current_platform.seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    maybe_create_device_identity()

    hidden_size = 128
    out_size = 256

    qinput = torch.randn(batch_size, hidden_size, device=device).to(FP8_DTYPE)
    weight = torch.randn(out_size, hidden_size, device=device).to(FP8_DTYPE).t()

    if use_per_token_scale:
        scale_a = torch.rand(batch_size, 1, dtype=torch.float32, device=device) + 0.5
    else:
        scale_a = torch.rand(1, dtype=torch.float32, device=device) + 0.5

    scale_b = torch.rand(out_size, 1, dtype=torch.float32, device=device) + 0.5
    bias = torch.randn(out_size, dtype=torch.float16, device=device) * 2.0

    output_with_bias = torch_channelwise_w8a8_scaled_mm(
        qinput=qinput,
        weight=weight,
        out_dtype=torch.float16,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        output_shape=[batch_size, out_size],
    )

    output_no_bias = torch_channelwise_w8a8_scaled_mm(
        qinput=qinput,
        weight=weight,
        out_dtype=torch.float16,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=None,  # type: ignore
        output_shape=[batch_size, out_size],
    )

    assert not torch.allclose(output_with_bias, output_no_bias, rtol=1e-3, atol=1e-3)

    diff = output_with_bias - output_no_bias
    mean_diff = diff.mean(dim=0)
    correlation = torch.corrcoef(torch.stack([mean_diff, bias]))[0, 1]
    assert correlation > 0.9


@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
    reason="FP8 requires CUDA or ROCm",
)
def test_scale_consistency():
    """Test that per-tensor and per-token scales produce consistent results."""
    torch.manual_seed(42)
    current_platform.seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    maybe_create_device_identity()

    batch_size = 16
    hidden_size = 128
    out_size = 256

    qinput = torch.randn(batch_size, hidden_size, device=device).to(FP8_DTYPE)
    weight = torch.randn(out_size, hidden_size, device=device).to(FP8_DTYPE).t()

    scale_value = 0.75
    scale_a_tensor = torch.tensor([scale_value], dtype=torch.float32, device=device)
    scale_a_token = torch.full(
        (batch_size, 1), scale_value, dtype=torch.float32, device=device
    )
    scale_b = torch.rand(out_size, 1, dtype=torch.float32, device=device) + 0.5

    output_tensor = torch_channelwise_w8a8_scaled_mm(
        qinput=qinput,
        weight=weight,
        out_dtype=torch.float16,
        scale_a=scale_a_tensor,
        scale_b=scale_b,
        bias=None,  # type: ignore
        output_shape=[batch_size, out_size],
    )

    output_token = torch_channelwise_w8a8_scaled_mm(
        qinput=qinput,
        weight=weight,
        out_dtype=torch.float16,
        scale_a=scale_a_token,
        scale_b=scale_b,
        bias=None,  # type: ignore
        output_shape=[batch_size, out_size],
    )

    assert torch.allclose(output_tensor, output_token, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [16, 64])
@pytest.mark.parametrize("use_per_token_scale", [False, True])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
    reason="FP8 requires CUDA or ROCm",
)
def test_dynamic_fp8_quantization(batch_size: int, use_per_token_scale: bool):
    """Test dynamic FP8 quantization correctness."""
    torch.manual_seed(42 + batch_size)
    current_platform.seed_everything(42 + batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    maybe_create_device_identity()

    hidden_size = 256
    out_size = 512

    # Create FP16 tensors
    input_fp16 = torch.randn(
        batch_size, hidden_size, device=device, dtype=torch.float16
    )
    weight_fp16 = torch.randn(out_size, hidden_size, device=device, dtype=torch.float16)

    fp8_max = torch.finfo(FP8_DTYPE).max
    if use_per_token_scale:
        scale_a = input_fp16.abs().amax(dim=1, keepdim=True) / fp8_max
        scale_a = torch.clamp(scale_a, min=1e-12)
    else:
        scale_a = input_fp16.abs().amax() / fp8_max
        scale_a = torch.clamp(scale_a, min=1e-12).reshape(1)

    scale_b = weight_fp16.abs().amax(dim=1, keepdim=True) / fp8_max
    scale_b = torch.clamp(scale_b, min=1e-12)
    qinput = (input_fp16 / scale_a).to(FP8_DTYPE)
    qweight = (weight_fp16.t() / scale_b.t()).to(FP8_DTYPE)

    output_fp8 = torch_channelwise_w8a8_scaled_mm(
        qinput=qinput,
        weight=qweight,
        out_dtype=torch.float16,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=None,  # type: ignore
        output_shape=[batch_size, out_size],
    )

    output_fp16 = torch.matmul(input_fp16, weight_fp16.t())
    input_dequant = qinput.to(torch.float32) * scale_a
    weight_dequant = qweight.to(torch.float32) * scale_b.t()
    output_manual = torch.matmul(input_dequant, weight_dequant).to(torch.float16)

    assert torch.isfinite(output_fp8).all()
    assert output_fp8.abs().max() > 0
    assert output_fp8.shape == (batch_size, out_size)
    # NOTE: This is the true test of correctness
    assert torch.allclose(output_fp8, output_manual, rtol=0.05, atol=0.1)
    # NOTE: The following are for future reference
    #       If impl breaks these may help detect numerical issues early
    max_diff = (output_fp8 - output_fp16).abs().max()
    mean_diff = (output_fp8 - output_fp16).abs().mean()
    assert max_diff < 3.0  # NOTE: magic number - but this is for future reference
    assert mean_diff < 1.0  # NOTE: magic number - but this is for future reference
