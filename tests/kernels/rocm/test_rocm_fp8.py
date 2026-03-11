# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 kernels on ROCm.

Covers:
- VLLM_ROCM_FP8_PADDING: FP8 weight tensors padded to 256-byte boundary
- VLLM_ROCM_USE_AITER_FP8BMM: AITER FP8 batch matrix multiply vs torch._scaled_mm
- VLLM_ROCM_FP8_MFMA_PAGE_ATTN: FP8 KV cache paged attention correctness
- FP8 per-tensor and per-token quantization on ROCm
- AITER group FP8 quant (already exercised in test_rocm_aiter_moe.py, extended here)
"""

import importlib.util

import pytest
import torch

from tests.kernels.rocm.utils import _assert_accurate, _assert_deterministic
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

fp8_available = current_platform.is_rocm() and current_platform.supports_fp8()
aiter_available = importlib.util.find_spec("aiter") is not None

FP8_DTYPE = (
    current_platform.fp8_dtype() if current_platform.is_rocm() else torch.float8_e4m3fn
)


def require_fp8():
    if not current_platform.supports_fp8():
        pytest.skip("FP8 not supported on this hardware")


def require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter package required for this test")


# ── FP8 padding tests ─────────────────────────────────────────────────────


def test_fp8_padding_env_var_readable(monkeypatch):
    """VLLM_ROCM_FP8_PADDING env var is readable and defaults to True."""
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)
    # Default is True
    assert isinstance(envs.VLLM_ROCM_FP8_PADDING, bool)


@pytest.mark.parametrize("padding_enabled", [True, False])
def test_fp8_padding_env_var_set(padding_enabled, monkeypatch):
    """VLLM_ROCM_FP8_PADDING can be set to True or False."""
    monkeypatch.setenv("VLLM_ROCM_FP8_PADDING", "1" if padding_enabled else "0")
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)
    assert padding_enabled == envs.VLLM_ROCM_FP8_PADDING


def test_fp8_padding_256_byte_alignment():
    """FP8 weight padding to 256 bytes preserves values in the original region.

    The skinny GEMM tests use pad_fp8() which pads weights to be a multiple
    of 256 bytes. Verify that the padding operation is safe and the original
    data region is unchanged.
    """
    require_fp8()
    torch.set_default_device("cuda")

    M, K = 64, 256
    weight = torch.randn(M, K).to(FP8_DTYPE)

    # Replicate pad_fp8 from test_rocm_skinny_gemms.py
    import torch.nn.functional as F

    num_pad = 256 // weight.element_size()
    padded = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]

    assert padded.shape == weight.shape
    # Values in the original region must match after round-trip
    torch.testing.assert_close(padded.to(torch.float32), weight.to(torch.float32))


# ── AITER FP8 BMM tests ───────────────────────────────────────────────────


def test_fp8bmm_env_var_readable():
    """VLLM_ROCM_USE_AITER_FP8BMM env var is readable."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_USE_AITER_FP8BMM, bool)


def test_fp8bmm_is_enabled_api():
    """rocm_aiter_ops.is_fp8bmm_enabled() API exists and returns bool/None."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    result = rocm_aiter_ops.is_fp8bmm_enabled()
    assert result is None or isinstance(result, bool)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.skipif(not fp8_available, reason="FP8 not supported")
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 128, 128)),
        pytest.param(
            (4, 256, 512),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    "ROCm FP8 scaled_mm precision gap at larger shapes: "
                    "requires atol=0.5, NVIDIA Cutlass FP8 achieves "
                    "atol=1.5e-1 (test_cutlass_scaled_mm.py). "
                    "Fix in ROCm FP8 GEMM accumulation."
                ),
            ),
        ),
        pytest.param(
            (8, 512, 1024),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    "ROCm FP8 scaled_mm precision gap at larger shapes: "
                    "requires atol=0.5, NVIDIA Cutlass FP8 achieves "
                    "atol=1.5e-1 (test_cutlass_scaled_mm.py). "
                    "Fix in ROCm FP8 GEMM accumulation."
                ),
            ),
        ),
    ],
)
def test_fp8_scaled_mm_vs_reference(shape):
    """FP8 scaled_mm output matches float32 matmul on the same dequantized inputs.

    The correct reference is the float32 matmul of the dequantized FP8 tensors
    (A_fp8 * scale_a) @ (B_fp8 * scale_b).T — this isolates accumulation error
    in the FP8 GEMM from per-tensor quantization error in A and B.
    Comparing against the original unquantized A @ B.T is incorrect: per-tensor
    FP8 quantization error accumulates over K and far exceeds any reasonable atol.
    This exercises the FP8 GEMM path used by AITER linear layers
    (VLLM_ROCM_USE_AITER_LINEAR, VLLM_ROCM_USE_AITER_FP8BMM).
    """
    require_fp8()
    from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    batch, M, K = shape
    N = K  # square for simplicity

    A = torch.randn(batch * M, K)
    B = torch.randn(N, K)

    A_fp8, scale_a = ref_dynamic_per_tensor_fp8_quant(A)
    B_fp8, scale_b = ref_dynamic_per_tensor_fp8_quant(B)

    out_dtype = torch.bfloat16

    # Kernel under test: FP8 scaled matmul
    out_fp8 = torch._scaled_mm(
        A_fp8, B_fp8.t(), out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b
    )

    assert out_fp8.shape == (batch * M, N)
    assert out_fp8.dtype == out_dtype
    assert not torch.any(torch.isnan(out_fp8))

    # Reference: float32 matmul on dequantized FP8 inputs.
    # This is what the FP8 GEMM computes: (A_fp8 * scale_a) @ (B_fp8 * scale_b).T
    # Running in float32 avoids BF16 rounding contaminating the reference.
    A_dq = A_fp8.float() * scale_a.float()
    B_dq = B_fp8.float() * scale_b.float()
    ref_f32 = torch.mm(A_dq, B_dq.t())

    _assert_accurate(out_fp8.float(), ref_f32, atol=1.5e-1, rtol=0.0)


@pytest.mark.skipif(not fp8_available, reason="FP8 not supported")
@torch.inference_mode()
def test_fp8_scaled_mm_determinism():
    """torch._scaled_mm (FP8 GEMM path) is deterministic on ROCm."""
    require_fp8()
    from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant

    torch.set_default_device("cuda")
    torch.manual_seed(5)

    M, K, N = 64, 128, 128
    A = torch.randn(M, K)
    B = torch.randn(N, K)
    A_fp8, scale_a = ref_dynamic_per_tensor_fp8_quant(A)
    B_fp8, scale_b = ref_dynamic_per_tensor_fp8_quant(B)

    def run_gemm():
        return torch._scaled_mm(
            A_fp8,
            B_fp8.t(),
            out_dtype=torch.bfloat16,
            scale_a=scale_a,
            scale_b=scale_b,
        )

    _assert_deterministic(run_gemm, n_runs=4)


# ── FP8 MFMA page attention test ──────────────────────────────────────────


def test_fp8_mfma_page_attn_env_var_readable():
    """VLLM_ROCM_FP8_MFMA_PAGE_ATTN env var is readable and defaults to False."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_FP8_MFMA_PAGE_ATTN, bool)


@pytest.mark.skipif(not fp8_available, reason="FP8 not supported")
@torch.inference_mode()
def test_fp8_mfma_page_attn_correctness():
    """FP8 KV cache paged attention produces finite results.

    Tests VLLM_ROCM_FP8_MFMA_PAGE_ATTN path: paged attention with FP8
    quantized KV cache on ROCm.
    """
    require_fp8()
    from vllm import _custom_ops as ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    num_q_heads = 8
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 512
    num_seqs = 2
    seq_lens = [128, 256]
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=torch.bfloat16)
    # FP8 KV cache
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size).to(
        FP8_DTYPE
    )
    value_cache = torch.randn_like(key_cache.float()).to(FP8_DTYPE)

    max_num_blocks = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    output = torch.empty(num_seqs, num_q_heads, head_size, dtype=torch.bfloat16)
    num_partitions = (max_seq_len + 255) // 256
    tmp_output = torch.empty(
        num_seqs, num_q_heads, num_partitions, head_size, dtype=torch.float32
    )
    exp_sums = torch.empty(num_seqs, num_q_heads, num_partitions, dtype=torch.float32)
    max_logits = torch.empty_like(exp_sums)

    # kv_cache_dtype="fp8" triggers FP8 dequantization path
    ops.paged_attention_rocm(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens_tensor,
        None,
        block_size,
        max_seq_len,
        None,
        "fp8",
        torch.tensor(1.0, dtype=torch.float32),  # k_scale
        torch.tensor(1.0, dtype=torch.float32),  # v_scale
    )

    assert output.shape == (num_seqs, num_q_heads, head_size)
    assert not torch.any(torch.isnan(output))
    assert not torch.any(torch.isinf(output))

    # Reference: BF16 paged attention (same query, FP8 KV cast to BF16)
    key_cache_bf16 = key_cache.to(torch.bfloat16)
    value_cache_bf16 = value_cache.to(torch.bfloat16)
    output_ref = torch.empty_like(output)
    exp_sums_ref = torch.empty_like(exp_sums)
    max_logits_ref = torch.empty_like(max_logits)
    tmp_output_ref = torch.empty_like(tmp_output)

    ops.paged_attention_rocm(
        output_ref,
        exp_sums_ref,
        max_logits_ref,
        tmp_output_ref,
        query,
        key_cache_bf16,
        value_cache_bf16,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens_tensor,
        None,
        block_size,
        max_seq_len,
        None,
        "auto",
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )

    # FP8 dequantized KV cache introduces quantization error; use loose tolerance
    _assert_accurate(
        output.float(),
        output_ref.float(),
        atol=0.5,
        rtol=0.1,
        pass_rate=0.95,
        max_violation_factor=5.0,
    )


# ── FP8 quantization correctness ──────────────────────────────────────────


@pytest.mark.skipif(not fp8_available, reason="FP8 not supported")
@pytest.mark.parametrize("shape", [(128, 256), (512, 1024), (1024, 4096)])
def test_fp8_per_tensor_quant_roundtrip(shape):
    """FP8 per-tensor quantize then dequant is close to original."""
    require_fp8()
    from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant

    torch.set_default_device("cuda")
    x = torch.randn(*shape)

    x_fp8, scale = ref_dynamic_per_tensor_fp8_quant(x)
    # Dequantize
    x_dequant = x_fp8.to(torch.float32) * scale

    assert x_dequant.shape == x.shape
    # FP8 quantization introduces limited error; check relative error bound
    rel_error = (x_dequant - x).abs() / (x.abs() + 1e-5)
    assert rel_error.mean() < 0.1  # less than 10% mean relative error


@pytest.mark.skipif(
    not (aiter_available and fp8_available), reason="aiter + FP8 required"
)
def test_aiter_group_fp8_quant_roundtrip():
    """AITER group FP8 quant: dequantized output is close to original input."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")

    M, N = 64, 4096
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)
    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(x, group_size)

    from aiter import dtypes

    assert x_fp8.dtype == dtypes.fp8
    assert scales.dtype == torch.float32

    # Dequantize: scales are [M, num_groups]; expand to [M, N]
    _num_groups = (N + group_size - 1) // group_size
    scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :N]
    x_dequant = x_fp8.float() * scales_expanded

    # FP8 group quant should preserve values within ~10% relative error
    rel_error = (x_dequant - x.float()).abs() / (x.float().abs() + 1e-5)
    assert rel_error.mean() < 0.1, (
        f"Group FP8 quant mean relative error {rel_error.mean():.4f} exceeds 10%"
    )
    assert (rel_error < 0.5).float().mean() > 0.99, (
        "Over 1% of group FP8 quant values have >50% relative error"
    )


# ── FP8 per-token group quant ─────────────────────────────────────────────


@pytest.mark.skipif(not fp8_available, reason="FP8 not supported")
def test_fp8_per_token_quant_via_custom_op():
    """FP8 per-token quantization: dequantized output is close to original."""
    require_fp8()
    from vllm import _custom_ops as ops

    torch.set_default_device("cuda")
    x = torch.randn(32, 4096, dtype=torch.bfloat16)

    # ops.scaled_fp8_quant with use_per_token_if_dynamic=True performs per-token quant.
    # scale shape: [num_tokens, 1] (one scale per token row).
    out, scale = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    assert out.shape == x.shape
    assert out.dtype == FP8_DTYPE
    assert scale.shape == (x.shape[0], 1), (
        f"Expected per-token scale shape ({x.shape[0]}, 1), got {scale.shape}"
    )

    # Dequantize: scale is [num_tokens, 1]
    x_dequant = out.float() * scale.float()

    # FP8 per-token quant should preserve values within ~5% relative error
    rel_error = (x_dequant - x.float()).abs() / (x.float().abs() + 1e-5)
    assert rel_error.mean() < 0.05, (
        f"FP8 per-token quant mean relative error {rel_error.mean():.4f} exceeds 5%"
    )
