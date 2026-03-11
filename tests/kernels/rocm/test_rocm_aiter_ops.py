# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AITER custom ops on ROCm that are not covered by other test files.

Covers:
- rocm_aiter_rms_norm: shape/dtype correctness + accuracy for float16 AND bfloat16
- rocm_aiter_rmsnorm2d_fwd_with_add: residual-add variant shape + accuracy
- rocm_aiter_triton_rotary_embedding: in-place NeoX-style RoPE accuracy
- rocm_aiter_per_token_quant: per-token FP8 quantization shape + roundtrip
- rocm_aiter_per_tensor_quant: per-tensor FP8 quantization shape + roundtrip
- rocm_aiter_act_mul_and_fp8_group_quant: SiLU-gate + FP8 group quant
- rocm_aiter_group_fp8_quant: group-wise FP8 quantization shape + roundtrip
- Fused RMSNorm + quantization ops (accuracy vs sequential composition):
  * rocm_aiter_rmsnorm_fused_dynamic_quant: RMSNorm + per-token FP8
  * rocm_aiter_rmsnorm_fused_add_dynamic_quant: residual-add + RMSNorm + per-token FP8
  * rocm_aiter_rmsnorm_fp8_group_quant: RMSNorm + FP8 group quant
  * rocm_aiter_rmsnorm_with_add_fp8_group_quant:
    residual-add + RMSNorm + FP8 group quant
- End-to-end inference chains: RMSNorm → per-token/group FP8 quant (bf16 and fp16)
- Determinism: bitwise-identical results across N runs
- AITER env var reads: VLLM_ROCM_USE_AITER_RMSNORM, VLLM_ROCM_USE_AITER_TRITON_ROPE,
  VLLM_ROCM_USE_AITER_TRITON_GEMM, VLLM_ROCM_USE_AITER_LINEAR
"""

import importlib

import pytest
import torch

from tests.kernels.rocm.utils import _assert_accurate, _assert_deterministic
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


def require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter package required for this test")


def require_fp8():
    if not current_platform.supports_fp8():
        pytest.skip("FP8 not supported on this hardware")


# ── Env var readable tests ────────────────────────────────────────────────


def test_use_aiter_rmsnorm_env_var_readable():
    """VLLM_ROCM_USE_AITER_RMSNORM is readable and is a bool."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_USE_AITER_RMSNORM, bool)


def test_use_aiter_triton_rope_env_var_readable():
    """VLLM_ROCM_USE_AITER_TRITON_ROPE is readable and is a bool."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_USE_AITER_TRITON_ROPE, bool)


def test_use_aiter_triton_gemm_env_var_readable():
    """VLLM_ROCM_USE_AITER_TRITON_GEMM is readable and is a bool."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_USE_AITER_TRITON_GEMM, bool)


def test_use_aiter_linear_env_var_readable():
    """VLLM_ROCM_USE_AITER_LINEAR is readable and is a bool."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_USE_AITER_LINEAR, bool)


@pytest.mark.parametrize("enabled", [True, False])
def test_aiter_rmsnorm_env_var_set(enabled, monkeypatch):
    """VLLM_ROCM_USE_AITER_RMSNORM can be set via env."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_RMSNORM", "1" if enabled else "0")
    import vllm.envs as envs

    importlib.reload(envs)
    assert enabled == envs.VLLM_ROCM_USE_AITER_RMSNORM


@pytest.mark.parametrize("enabled", [True, False])
def test_aiter_triton_rope_env_var_set(enabled, monkeypatch):
    """VLLM_ROCM_USE_AITER_TRITON_ROPE can be set via env."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "1" if enabled else "0")
    import vllm.envs as envs

    importlib.reload(envs)
    assert enabled == envs.VLLM_ROCM_USE_AITER_TRITON_ROPE


# ── Op registration tests ─────────────────────────────────────────────────


def test_rocm_aiter_rms_norm_registered():
    """rocm_aiter_rms_norm custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rms_norm")
    assert callable(torch.ops.vllm.rocm_aiter_rms_norm)


def test_rocm_aiter_rmsnorm2d_fwd_with_add_registered():
    """rocm_aiter_rmsnorm2d_fwd_with_add custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm2d_fwd_with_add")


def test_rocm_aiter_triton_rotary_embedding_registered():
    """rocm_aiter_triton_rotary_embedding custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_triton_rotary_embedding")


def test_rocm_aiter_per_token_quant_registered():
    """rocm_aiter_per_token_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_per_token_quant")


def test_rocm_aiter_per_tensor_quant_registered():
    """rocm_aiter_per_tensor_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_per_tensor_quant")


def test_rocm_aiter_act_mul_and_fp8_group_quant_registered():
    """rocm_aiter_act_mul_and_fp8_group_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_act_mul_and_fp8_group_quant")


def test_rocm_aiter_group_fp8_quant_registered():
    """rocm_aiter_group_fp8_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_group_fp8_quant")


# ── rocm_aiter_rms_norm correctness tests ─────────────────────────────────


def test_rocm_aiter_rms_norm_output_shape():
    """rocm_aiter_rms_norm returns tensor of same shape as input."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    out = rocm_aiter_ops.rms_norm(x, weight, eps)
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert not torch.any(torch.isnan(out))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_aiter_rms_norm_dtype(dtype):
    """rocm_aiter_rms_norm preserves input dtype."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    x = torch.randn(16, 256, dtype=dtype)
    weight = torch.ones(256, dtype=dtype)
    out = rocm_aiter_ops.rms_norm(x, weight, 1e-5)
    assert out.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_aiter_rms_norm_vs_torch(dtype):
    """rocm_aiter_rms_norm matches PyTorch manual RMSNorm for float16 and bfloat16."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    M, N = 8, 128
    x = torch.randn(M, N, dtype=dtype)
    weight = torch.ones(N, dtype=dtype)
    eps = 1e-5

    # Reference: float32 RMSNorm for precision
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref = (x.float() / rms * weight.float()).to(dtype)

    out = rocm_aiter_ops.rms_norm(x, weight, eps)
    _assert_accurate(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


# ── rocm_aiter_rmsnorm2d_fwd_with_add tests ───────────────────────────────


def test_rocm_aiter_rmsnorm_with_add_output_shapes():
    """rocm_aiter_rmsnorm2d_fwd_with_add returns (normed, residual)
    with correct shapes."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    M, N = 16, 256
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    out, res_out = rocm_aiter_ops.rms_norm2d_with_add(x, residual, weight, eps)
    assert out.shape == (M, N)
    assert res_out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert not torch.any(torch.isnan(out))


# ── rocm_aiter_per_token_quant tests ──────────────────────────────────────


def test_rocm_aiter_per_token_quant_output_shapes():
    """rocm_aiter_per_token_quant returns (quantized, scale) with correct shapes."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    fp8_dtype = current_platform.fp8_dtype()

    x_quant, scale = rocm_aiter_ops.per_token_quant(x, fp8_dtype)
    assert x_quant.shape == (M, N)
    assert x_quant.dtype == fp8_dtype
    assert scale.shape[0] == M  # one scale per token
    assert not torch.any(torch.isnan(scale))


# ── rocm_aiter_per_tensor_quant tests ─────────────────────────────────────


def test_rocm_aiter_per_tensor_quant_output_shapes():
    """rocm_aiter_per_tensor_quant returns (quantized, scale) with correct shapes."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    fp8_dtype = current_platform.fp8_dtype()

    x_quant, scale = rocm_aiter_ops.per_tensor_quant(x, fp8_dtype)
    assert x_quant.shape == (M, N)
    assert x_quant.dtype == fp8_dtype
    # Scale is a scalar or single-element tensor
    assert scale.numel() == 1


# ── rocm_aiter_act_mul_and_fp8_group_quant tests ──────────────────────────


def test_rocm_aiter_act_mul_and_fp8_group_quant_output_shapes():
    """act_mul_and_fp8_group_quant halves the last dim (gate+up) and returns FP8."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401 — ensure op is registered

    torch.set_default_device("cuda")
    M, N = 32, 512  # N must be even (gate + up halves)
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)

    x_quant, scale = torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant(
        x, group_size
    )
    N_half = N // 2
    fp8_dtype = current_platform.fp8_dtype()
    assert x_quant.shape == (M, N_half)
    assert x_quant.dtype == fp8_dtype
    expected_scale_cols = (N_half + group_size - 1) // group_size
    assert scale.shape == (M, expected_scale_cols)


# ── rocm_aiter_group_fp8_quant tests ──────────────────────────────────────


def test_rocm_aiter_group_fp8_quant_output_shapes():
    """rocm_aiter_group_fp8_quant returns (quantized, scales) with correct shapes."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    M, N = 32, 512
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)

    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(x, group_size)
    fp8_dtype = current_platform.fp8_dtype()
    assert x_fp8.shape == (M, N)
    assert x_fp8.dtype == fp8_dtype
    expected_scale_cols = (N + group_size - 1) // group_size
    assert scales.shape == (M, expected_scale_cols)
    assert scales.dtype == torch.float32


# ── rocm_aiter_ops API availability ───────────────────────────────────────


def test_rocm_aiter_ops_rmsnorm_attr():
    """rocm_aiter_ops has rms_norm attribute when aiter is available."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    assert hasattr(rocm_aiter_ops, "rms_norm") or hasattr(
        torch.ops.vllm, "rocm_aiter_rms_norm"
    )


def test_rocm_aiter_ops_fused_rmsnorm_quant_registered():
    """rocm_aiter_rmsnorm_fused_dynamic_quant op is registered when aiter available."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm_fused_dynamic_quant") or hasattr(
        torch.ops.vllm, "rocm_aiter_rmsnorm_fp8_group_quant"
    )


# ── Additional env var set test ────────────────────────────────────────────


@pytest.mark.parametrize("enabled", [True, False])
def test_use_aiter_triton_gemm_set(enabled, monkeypatch):
    """VLLM_ROCM_USE_AITER_TRITON_GEMM can be set to True or False."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_GEMM", "1" if enabled else "0")
    import vllm.envs as envs

    importlib.reload(envs)
    assert enabled == envs.VLLM_ROCM_USE_AITER_TRITON_GEMM


# ── Numerical accuracy tests for AITER custom ops ─────────────────────────


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "AITER rms_norm2d_with_add: max element error 0.03125 (2^-5 BF16 ULP) "
        "exceeds 3x atol=0.03 max-violation check. Target atol=1e-3 (stricter than "
        "NVIDIA RMSNorm baseline atol=1e-2 from test_layernorm.py). "
        "Kernel must reduce max absolute error below 0.003."
    ),
)
def test_rocm_aiter_rmsnorm_with_add_vs_torch():
    """rocm_aiter_rmsnorm2d_fwd_with_add matches manual residual+RMSNorm reference."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    M, N = 16, 256
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    # Reference: add residual, then RMSNorm
    h = x.float() + residual.float()
    rms = h.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref_normed = (h / rms * weight.float()).to(torch.bfloat16)
    ref_residual = (x.float() + residual.float()).to(torch.bfloat16)

    out, res_out = rocm_aiter_ops.rms_norm2d_with_add(x, residual, weight, eps)

    _assert_accurate(out.float(), ref_normed.float(), atol=1e-2, rtol=1e-2)
    _assert_accurate(res_out.float(), ref_residual.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "AITER Triton RoPE precision gap: max element error ~0.03125 (2^-5 BF16 ULP) "
        "requires atol~=2e-2; NVIDIA CUDA RoPE achieves atol=1e-3 for bf16 "
        "(allclose_default.py). Fix in upstream aiter rope kernel."
    ),
)
def test_rocm_aiter_triton_rotary_embedding_vs_torch():
    """rocm_aiter_triton_rotary_embedding matches manual NeoX-style RoPE reference.

    The AITER kernel calls rope_cached_thd_positions_offsets_2c_fwd_inplace with
    reuse_freqs_front_part=True. This means it reads only the first head_size//2
    entries of cos and sin from the cache and applies them to both halves of the
    head (pairwise NeoX rotation). The cos_sin_cache must be built with the second
    half mirroring the first so the reference and kernel agree.
    """
    require_aiter()
    import vllm._aiter_ops  # noqa: F401 — ensure op is registered

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    num_tokens = 8
    num_heads = 4
    head_size = 64
    half_dim = head_size // 2
    max_pos = 32

    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.bfloat16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.bfloat16)

    # Build cos/sin cache with second half mirroring first half.
    # The kernel uses reuse_freqs_front_part=True: it reads only cos/sin[:, :half_dim]
    # and applies those same frequencies to both the first and second
    # halves of the head.
    cos_half = torch.randn(max_pos, half_dim, dtype=torch.bfloat16)
    sin_half = torch.randn(max_pos, half_dim, dtype=torch.bfloat16)
    cos_cache = torch.cat([cos_half, cos_half], dim=-1)  # [max_pos, head_size]
    sin_cache = torch.cat([sin_half, sin_half], dim=-1)  # [max_pos, head_size]
    cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1)  # [max_pos, 2*head_size]

    # Reference: NeoX-style pairwise rotation with front-half frequencies only.
    # rotate_style=0 (NeoX): [x1*c - x2*s, x2*c + x1*s]
    cos_pos = cos_half[positions]  # [num_tokens, half_dim]
    sin_pos = sin_half[positions]  # [num_tokens, half_dim]

    def apply_rope_ref(t: torch.Tensor) -> torch.Tensor:
        t_r = t.float().view(num_tokens, num_heads, head_size)
        c = cos_pos.float().unsqueeze(1)  # [num_tokens, 1, half_dim]
        s = sin_pos.float().unsqueeze(1)
        x1, x2 = t_r[..., :half_dim], t_r[..., half_dim:]
        rotated = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
        return rotated.to(t.dtype).view(num_tokens, num_heads * head_size)

    ref_q = apply_rope_ref(query)
    ref_k = apply_rope_ref(key)

    # AITER in-place RoPE (modifies query/key in-place)
    q_aiter = query.clone()
    k_aiter = key.clone()
    torch.ops.vllm.rocm_aiter_triton_rotary_embedding(
        positions,
        q_aiter,
        k_aiter,
        head_size,
        cos_sin_cache,
        True,  # is_neox style → rotate_style=0
    )

    _assert_accurate(q_aiter.float(), ref_q.float(), atol=1e-3, rtol=1.6e-2)
    _assert_accurate(k_aiter.float(), ref_k.float(), atol=1e-3, rtol=1.6e-2)


def test_rocm_aiter_per_token_quant_roundtrip():
    """rocm_aiter_per_token_quant: dequantized output is close to original."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(1)

    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    fp8_dtype = current_platform.fp8_dtype()

    x_quant, scale = rocm_aiter_ops.per_token_quant(x, fp8_dtype)

    # Dequantize: scale is [M] or [M, 1]
    scale_exp = scale.view(M, 1).float()
    x_dequant = x_quant.float() * scale_exp

    rel_error = (x_dequant - x.float()).abs() / (x.float().abs() + 1e-5)
    assert rel_error.mean() < 0.05, (
        f"per_token_quant mean rel error {rel_error.mean():.4f} > 5%"
    )


def test_rocm_aiter_per_tensor_quant_roundtrip():
    """rocm_aiter_per_tensor_quant: dequantized output is close to original."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(2)

    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    fp8_dtype = current_platform.fp8_dtype()

    x_quant, scale = rocm_aiter_ops.per_tensor_quant(x, fp8_dtype)

    # Dequantize: scale is scalar
    x_dequant = x_quant.float() * scale.float()

    rel_error = (x_dequant - x.float()).abs() / (x.float().abs() + 1e-5)
    assert rel_error.mean() < 0.05, (
        f"per_tensor_quant mean rel error {rel_error.mean():.4f} > 5%"
    )


def test_rocm_aiter_act_mul_fp8_group_quant_roundtrip():
    """act_mul_and_fp8_group_quant: dequantized output matches SiLU gate reference."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401 — ensure op is registered

    torch.set_default_device("cuda")
    torch.manual_seed(3)

    M, N = 32, 512  # N even: N//2 gate, N//2 up
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)

    x_quant, scale = torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant(
        x, group_size
    )

    N_half = N // 2
    # Reference: SiLU(gate) * up
    gate = x.float()[:, :N_half]
    up = x.float()[:, N_half:]
    ref = torch.sigmoid(gate) * gate * up  # SiGLU

    # Dequantize: scale is [M, num_groups]
    _num_groups = (N_half + group_size - 1) // group_size
    scale_exp = scale.repeat_interleave(group_size, dim=1)[:, :N_half]
    x_dequant = x_quant.float() * scale_exp

    rel_error = (x_dequant - ref).abs() / (ref.abs() + 1e-5)
    assert rel_error.mean() < 0.1, (
        f"act_mul_fp8_group_quant mean rel error {rel_error.mean():.4f} > 10%"
    )


def test_rocm_aiter_group_fp8_quant_roundtrip():
    """rocm_aiter_group_fp8_quant: dequantized output is close to original."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(4)

    M, N = 32, 512
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)

    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(x, group_size)

    # Dequantize
    scales_exp = scales.repeat_interleave(group_size, dim=1)[:, :N]
    x_dequant = x_fp8.float() * scales_exp

    rel_error = (x_dequant - x.float()).abs() / (x.float().abs() + 1e-5)
    assert rel_error.mean() < 0.05, (
        f"group_fp8_quant mean rel error {rel_error.mean():.4f} > 5%"
    )


def test_rocm_aiter_rms_norm_determinism():
    """rocm_aiter_rms_norm produces bitwise-identical results across N runs."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(5)

    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    _assert_deterministic(rocm_aiter_ops.rms_norm, x, weight, eps, n_runs=4)


# ── Op registration tests for fused RMSNorm+quant ops ─────────────────────


def test_rocm_aiter_rmsnorm_fused_dynamic_quant_registered():
    """rocm_aiter_rmsnorm_fused_dynamic_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm_fused_dynamic_quant")


def test_rocm_aiter_rmsnorm_fused_add_dynamic_quant_registered():
    """rocm_aiter_rmsnorm_fused_add_dynamic_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm_fused_add_dynamic_quant")


def test_rocm_aiter_rmsnorm_fp8_group_quant_registered():
    """rocm_aiter_rmsnorm_fp8_group_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm_fp8_group_quant")


def test_rocm_aiter_rmsnorm_with_add_fp8_group_quant_registered():
    """rocm_aiter_rmsnorm_with_add_fp8_group_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm_with_add_fp8_group_quant")


# ── Fused RMSNorm + quantization accuracy tests ───────────────────────────


def test_rocm_aiter_rmsnorm_fused_dynamic_quant_vs_sequential():
    """Fused RMSNorm+per-token-FP8-quant matches sequential rms_norm→per_token_quant.

    Tests that the fused kernel produces the same result as the two-step
    sequential composition: rms_norm(x) followed by per_token_quant.
    The fused path is used in production for inference throughput.
    """
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()

    # Sequential reference
    normed = rocm_aiter_ops.rms_norm(x, weight, eps)
    ref_q, ref_scale = rocm_aiter_ops.per_token_quant(normed, fp8_dtype)
    ref_dequant = ref_q.float() * ref_scale.float()

    # Fused op
    fused_q, fused_scale = torch.ops.vllm.rocm_aiter_rmsnorm_fused_dynamic_quant(
        x, weight, eps, fp8_dtype
    )
    fused_dequant = fused_q.float() * fused_scale.float()

    assert fused_q.shape == (M, N)
    assert fused_q.dtype == fp8_dtype
    assert fused_scale.shape == (M, 1)

    # Fused vs sequential: both should recover RMSNorm output within FP8 error
    rel_error = (fused_dequant - ref_dequant).abs() / (ref_dequant.abs() + 1e-5)
    assert rel_error.mean() < 0.05, (
        f"fused_dynamic_quant vs sequential mean rel error {rel_error.mean():.4f} > 5%"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "AITER rmsnorm_fused_add_dynamic_quant: residual output max element error "
        "0.03125 (2^-5 BF16 ULP) exceeds 3x atol=0.003 max-violation check. "
        "Target atol=1e-3 (stricter than NVIDIA RMSNorm baseline atol=1e-2 from "
        "test_layernorm.py). Kernel must reduce max absolute error below 0.003."
    ),
)
def test_rocm_aiter_rmsnorm_fused_add_dynamic_quant_vs_reference():
    """Fused (residual-add + RMSNorm + per-token-FP8-quant)
    matches sequential reference.

    Production path: input + residual → RMSNorm → FP8 quantize, returning
    both the quantized norm output and the residual sum for the next layer.
    """
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(1)

    M, N = 16, 256
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()

    # Sequential reference: add residual → rms_norm → per_token_quant
    h = (x.float() + residual.float()).to(torch.bfloat16)
    normed = rocm_aiter_ops.rms_norm(h, weight, eps)
    ref_q, ref_scale = rocm_aiter_ops.per_token_quant(normed, fp8_dtype)
    ref_residual_out = h

    # Fused op: returns (x_quant, residual_out, scale)
    fused_q, fused_res_out, fused_scale = (
        torch.ops.vllm.rocm_aiter_rmsnorm_fused_add_dynamic_quant(
            x, residual, weight, eps, fp8_dtype
        )
    )

    assert fused_q.shape == (M, N)
    assert fused_q.dtype == fp8_dtype
    assert fused_scale.shape == (M, 1)
    assert fused_res_out.shape == (M, N)

    # Residual output matches x + residual
    _assert_accurate(
        fused_res_out.float(), ref_residual_out.float(), atol=1e-3, rtol=1e-3
    )
    # Dequantized output matches sequential path
    fused_dequant = fused_q.float() * fused_scale.float()
    ref_dequant = ref_q.float() * ref_scale.float()
    rel_error = (fused_dequant - ref_dequant).abs() / (ref_dequant.abs() + 1e-5)
    assert rel_error.mean() < 0.05, (
        "fused_add_dynamic_quant vs sequential mean "
        f"rel error {rel_error.mean():.4f} > 5%"
    )


def test_rocm_aiter_rmsnorm_fp8_group_quant_vs_sequential():
    """Fused RMSNorm+FP8-group-quant matches sequential rms_norm→group_fp8_quant.

    Tests both output shapes and dequantized accuracy against the two-step
    sequential composition.
    """
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(2)

    M, N = 32, 512
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()
    expected_groups = (N + group_size - 1) // group_size

    # Fused op: (x_quant, scales)
    fused_q, fused_scales = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant(
        x, weight, eps, group_size
    )
    assert fused_q.shape == (M, N)
    assert fused_q.dtype == fp8_dtype
    assert fused_scales.shape == (M, expected_groups)

    # Dequantize and compare to reference: rms_norm → group quant → dequant
    normed = rocm_aiter_ops.rms_norm(x, weight, eps)
    ref_q, ref_scales = rocm_aiter_ops.group_fp8_quant(normed, group_size)
    scales_exp = ref_scales.repeat_interleave(group_size, dim=1)[:, :N]
    ref_dequant = ref_q.float() * scales_exp
    fused_scales_exp = fused_scales.repeat_interleave(group_size, dim=1)[:, :N]
    fused_dequant = fused_q.float() * fused_scales_exp

    rel_error = (fused_dequant - ref_dequant).abs() / (ref_dequant.abs() + 1e-5)
    assert rel_error.mean() < 0.05, (
        "rmsnorm_fp8_group_quant vs sequential mean "
        f"rel error {rel_error.mean():.4f} > 5%"
    )


def test_rocm_aiter_rmsnorm_with_add_fp8_group_quant_shapes():
    """Fused (residual-add + RMSNorm + FP8-group-quant) returns correct shapes."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401

    torch.set_default_device("cuda")
    torch.manual_seed(3)

    M, N = 32, 512
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()
    expected_groups = (N + group_size - 1) // group_size

    # Returns (x_quant, residual_out, scales)
    fused_q, fused_res, fused_scales = (
        torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant(
            x, residual, weight, eps, group_size
        )
    )

    assert fused_q.shape == (M, N)
    assert fused_q.dtype == fp8_dtype
    assert fused_res.shape == (M, N)
    assert fused_res.dtype == torch.bfloat16
    assert fused_scales.shape == (M, expected_groups)
    assert not torch.any(torch.isnan(fused_scales))


def test_rocm_aiter_rmsnorm_with_add_fp8_group_quant_residual_accuracy():
    """Fused rmsnorm_with_add_fp8_group_quant residual output matches x + residual."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(4)

    M, N = 16, 256
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    fused_q, fused_res, fused_scales = (
        torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant(
            x, residual, weight, eps, group_size
        )
    )

    # Residual output must equal x + residual
    ref_residual = (x.float() + residual.float()).to(torch.bfloat16)
    _assert_accurate(fused_res.float(), ref_residual.float(), atol=1e-2, rtol=1e-2)

    # Dequantized quant output must match rms_norm(x + residual)
    h = ref_residual
    rms = h.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref_normed = (h.float() / rms * weight.float()).to(torch.bfloat16)
    ref_q, ref_scales = rocm_aiter_ops.group_fp8_quant(ref_normed, group_size)
    ref_scales_exp = ref_scales.repeat_interleave(group_size, dim=1)[:, :N]
    ref_dequant = ref_q.float() * ref_scales_exp
    fused_scales_exp = fused_scales.repeat_interleave(group_size, dim=1)[:, :N]
    fused_dequant = fused_q.float() * fused_scales_exp
    rel_error = (fused_dequant - ref_dequant).abs() / (ref_dequant.abs() + 1e-5)
    assert rel_error.mean() < 0.05, (
        f"rmsnorm_with_add_fp8_group_quant mean rel error {rel_error.mean():.4f} > 5%"
    )


# ── End-to-end inference chain test ───────────────────────────────────────


def test_rocm_aiter_rms_norm_then_per_token_quant_e2e():
    """End-to-end: BF16 RMSNorm → per-token FP8 quantization → dequantize.

    Simulates the inference path through a transformer layer norm before a
    linear projection: verifies the full chain produces accurate output
    compared to a float32 reference.
    """
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    # Llama-style hidden dim
    M, N = 32, 4096
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.randn(N, dtype=torch.bfloat16)  # learned scale
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()

    # Float32 reference for the full chain
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref_normed_f32 = x.float() / rms * weight.float()

    # AITER chain: RMSNorm → per-token FP8 quant → dequant
    normed = rocm_aiter_ops.rms_norm(x, weight, eps)
    x_q, scale = rocm_aiter_ops.per_token_quant(normed, fp8_dtype)
    x_dequant = x_q.float() * scale.float()  # scale: [M, 1]

    # Dequantized result should match the float32 reference within FP8 quant error
    rel_error = (x_dequant - ref_normed_f32).abs() / (ref_normed_f32.abs() + 1e-5)
    assert rel_error.mean() < 0.05, (
        f"RMSNorm→per_token_quant e2e mean rel error {rel_error.mean():.4f} > 5%"
    )
    # Shape and dtype checks
    assert x_q.shape == (M, N)
    assert x_q.dtype == fp8_dtype
    assert scale.shape == (M, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_aiter_rms_norm_then_group_fp8_quant_e2e(dtype):
    """End-to-end: RMSNorm (fp16/bf16) → FP8 group quantization → dequantize.

    Covers both float16 and bfloat16 inputs to verify dtype-agnostic
    behavior of the RMSNorm+group-quant pipeline.
    """
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    M, N = 16, 512
    group_size = 128
    x = torch.randn(M, N, dtype=dtype)
    weight = torch.randn(N, dtype=dtype)
    eps = 1e-5

    # Float32 reference
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref_normed_f32 = x.float() / rms * weight.float()

    # AITER chain: RMSNorm → group FP8 quant → dequant
    normed = rocm_aiter_ops.rms_norm(x, weight, eps)
    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(normed.bfloat16(), group_size)
    scales_exp = scales.repeat_interleave(group_size, dim=1)[:, :N]
    x_dequant = x_fp8.float() * scales_exp

    rel_error = (x_dequant - ref_normed_f32).abs() / (ref_normed_f32.abs() + 1e-5)
    assert rel_error.mean() < 0.1, (
        f"RMSNorm({dtype})->group_fp8_quant e2e mean "
        f"rel error {rel_error.mean():.4f} > 10%"
    )
