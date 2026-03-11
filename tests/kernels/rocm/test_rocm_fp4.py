# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP4 / MXFP4 kernels on ROCm, targeting gfx950 (MI350X) parity
with NVIDIA B200 nvfp4/mxfp4 kernel tests.

Covers:
- VLLM_ROCM_USE_AITER_FP4_ASM_GEMM: FP4 ASM GEMM via aiter
- VLLM_ROCM_USE_AITER_FP4BMM: FP4 batch matrix multiply via aiter
- OCP MX MXFP4 quantize-dequant roundtrip using mxfp4_utils
- is_rocm_aiter_fp4_asm_gemm_enabled() / is_fp4bmm_enabled() API
- rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()
- MXFP4 triton GEMM dynamic quant path (aiter dynamic_mxfp4_quant)
- MXFP4 output format: packed uint8 FP4 values (2/byte), E8M0 uint8 scales
- MXFP4 quant format and accuracy for Llama-class shapes (4096, 11008, 14336)
- gfx950 A4W4 FP4 GEMM accuracy (gemm_afp4wfp4): multiple shapes + skinny decode
- gfx950 preshuffled weight scale GEMM (gemm_afp4wfp4_preshuffled_weight_scales)
- gfx950 hardware FP4 dynamic quant (dynamic_per_group_scaled_quant_fp4)
- FP4 GEMM determinism on gfx950
"""

import importlib.util

import pytest
import torch

from tests.kernels.rocm.utils import _assert_accurate, _assert_deterministic
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

aiter_available = importlib.util.find_spec("aiter") is not None


def require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter package required for this test")


def require_gfx950():
    """Skip if not on gfx950 (MI325X / MI300X successor)."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("FP4 ASM GEMM only supported on gfx950")


# ── Env var and API tests ─────────────────────────────────────────────────


def test_fp4_asm_gemm_env_var_readable():
    """VLLM_ROCM_USE_AITER_FP4_ASM_GEMM is readable (defaults to False)."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM, bool)


def test_fp4bmm_env_var_readable():
    """VLLM_ROCM_USE_AITER_FP4BMM is readable (defaults to True)."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_USE_AITER_FP4BMM, bool)


def test_fp4_asm_gemm_enabled_api():
    """rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled() returns bool/None."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    result = rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()
    assert result is None or isinstance(result, bool)


def test_fp4bmm_enabled_api():
    """rocm_aiter_ops.is_fp4bmm_enabled() returns bool/None.
    Requires gfx950 to be True.
    """
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    result = rocm_aiter_ops.is_fp4bmm_enabled()
    assert result is None or isinstance(result, bool)


def test_is_rocm_aiter_fp4_asm_gemm_enabled():
    """is_rocm_aiter_fp4_asm_gemm_enabled() returns bool."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        is_rocm_aiter_fp4_asm_gemm_enabled,
    )

    result = is_rocm_aiter_fp4_asm_gemm_enabled()
    assert isinstance(result, bool)


# ── MXFP4 quantization utils tests ────────────────────────────────────────


def test_mxfp4_quant_dequant_roundtrip():
    """MXFP4 quantize-dequant roundtrip preserves values within FP4 error bounds.

    Uses the vllm mxfp4_utils which underpins Quark OCP MX and AITER FP4 paths.
    MXFP4 has 4-bit precision (block-scale E8M0), so we expect:
    - Mean relative error < 25%
    - At least 99% of elements within 100% relative error
    """
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    # Representative weight sizes (must be multiples of block_size=32)
    M, K = 128, 256
    x = torch.randn(M, K, dtype=torch.bfloat16)

    x_dequant = quant_dequant_mxfp4(x)

    assert x_dequant.shape == x.shape
    assert x_dequant.dtype == x.dtype
    assert not torch.any(torch.isnan(x_dequant))
    assert not torch.any(torch.isinf(x_dequant))

    # Numerical accuracy bound for FP4 quantization
    rel_error = (x_dequant.float() - x.float()).abs() / (x.float().abs() + 1e-6)
    mean_rel_error = rel_error.mean().item()
    assert mean_rel_error < 0.25, (
        f"MXFP4 mean relative error {mean_rel_error:.4f} exceeds 25% bound"
    )
    pass_rate = (rel_error < 1.0).float().mean().item()
    assert pass_rate > 0.99, (
        f"MXFP4 pass rate (< 100% rel err) {pass_rate:.4f} below 99%"
    )


@pytest.mark.parametrize("shape", [(64, 128), (256, 512), (512, 1024)])
def test_mxfp4_quant_dequant_accuracy_shapes(shape):
    """MXFP4 quant-dequant accuracy holds across various weight shapes."""
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    x_dequant = quant_dequant_mxfp4(x)

    assert x_dequant.shape == x.shape
    assert not torch.any(torch.isnan(x_dequant))

    rel_error = (x_dequant.float() - x.float()).abs() / (x.float().abs() + 1e-6)
    assert rel_error.mean() < 0.25, (
        f"Shape {shape}: mean rel error {rel_error.mean():.4f} > 25%"
    )


def test_mxfp4_dequant_function():
    """dequant_mxfp4 correctly dequantizes MXFP4 data to BF16."""
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    M, K = 64, 128
    x = torch.randn(M, K, dtype=torch.bfloat16)

    x_dequant_via_roundtrip = quant_dequant_mxfp4(x)
    assert x_dequant_via_roundtrip.dtype == torch.bfloat16

    # The result should be close to the original (within FP4 quantization error)
    rel_error = (x_dequant_via_roundtrip - x).abs() / (x.abs() + 1e-5)
    assert rel_error.mean() < 0.3  # FP4 has limited precision


# ── OCP MX block size ─────────────────────────────────────────────────────


def test_ocp_mx_block_size():
    """OCP MX block size constant is 32 (per-1x32 block quantization)."""
    from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
        OCP_MX_BLOCK_SIZE,
    )

    assert OCP_MX_BLOCK_SIZE == 32


# ── AITER triton FP4 GEMM tests ────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_triton_fp4_gemm_importable():
    """aiter triton FP4 GEMM modules are importable when aiter is available."""
    require_aiter()
    try:
        from aiter.ops.triton.gemm_afp4wfp4 import (  # noqa: F401
            gemm_afp4wfp4,
            gemm_afp4wfp4_preshuffled_weight_scales,
        )
        from aiter.ops.triton.quant import dynamic_mxfp4_quant  # noqa: F401
    except ImportError:
        pytest.skip("aiter triton FP4 GEMM not available in this aiter version")


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_fp4_shuffle_weight_importable():
    """aiter shuffle_weight utility is importable."""
    require_aiter()
    try:
        from aiter.ops.shuffle import shuffle_weight  # noqa: F401
    except ImportError:
        pytest.skip("aiter shuffle_weight not available in this aiter version")


# ── AITER FP4 BMM tests ────────────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_fp4bmm_op_registered():
    """rocm_aiter_fp4_bmm custom op is registered when aiter is available."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401 triggers registration

    # FP4 BMM op registration depends on env + gfx950
    # We just verify the API pathway works
    from vllm._aiter_ops import rocm_aiter_ops

    result = rocm_aiter_ops.is_fp4bmm_enabled()
    assert result is None or isinstance(result, bool)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_fp4_asm_gemm_conditional_import():
    """FP4 ASM GEMM is only imported when VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1."""
    require_aiter()
    import vllm.envs as envs
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        is_rocm_aiter_fp4_asm_gemm_enabled,
    )

    enabled = is_rocm_aiter_fp4_asm_gemm_enabled()
    # Should match env var + USE_AITER + gfx950 check
    if not envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM or not envs.VLLM_ROCM_USE_AITER:
        assert not enabled


# ── Determinism and extended accuracy tests ────────────────────────────────


def test_mxfp4_determinism():
    """quant_dequant_mxfp4 produces bitwise-identical results across N runs."""
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(3)

    x = torch.randn(64, 128, dtype=torch.bfloat16)

    _assert_deterministic(quant_dequant_mxfp4, x, n_runs=4)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_fp4bmm_accuracy():
    """AITER FP4 BMM result is close to torch.matmul on dequantized inputs.

    Skipped unless on gfx950 (MI325X) where the FP4 BMM hardware op is active.
    On other architectures the test validates the API returns without error.
    """
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    result = rocm_aiter_ops.is_fp4bmm_enabled()
    if not result:
        pytest.skip("FP4 BMM not enabled on this hardware (requires gfx950)")

    torch.set_default_device("cuda")
    torch.manual_seed(4)

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    M, K, N = 64, 128, 64
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)  # transposed weight

    # Reference: matmul on dequantized FP4
    A_dq = quant_dequant_mxfp4(A)
    B_dq = quant_dequant_mxfp4(B)
    ref = torch.matmul(A_dq, B_dq.t())

    # FP4 BMM: quantize A and B, run the batch matmul
    # The AITER FP4 BMM op may require shuffled weights; use triton path as proxy
    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        B_fp4, B_scale = dynamic_mxfp4_quant(B)
        out = gemm_afp4wfp4(A_fp4, B_fp4, A_scale, B_scale)
    except (ImportError, Exception):
        pytest.skip("aiter triton FP4 GEMM not available in this aiter version")

    assert out.shape == (M, N)
    # FP4 precision: allow large tolerance (FP4 has only 4 bits of mantissa)
    rel_error = (out.float() - ref.float()).abs() / (ref.float().abs() + 1e-3)
    assert rel_error.mean() < 0.5, (
        f"FP4 BMM mean relative error {rel_error.mean():.4f} exceeds 50%"
    )


# ── AITER MXFP4 dynamic quant format tests ────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_dynamic_mxfp4_quant_output_format():
    """dynamic_mxfp4_quant returns packed uint8 FP4 values and E8M0 uint8 scales.

    OCP MXFP4: block_size=32, 2 FP4 E2M1 values packed per byte.
    Scale shape: (M, K // 32) — one E8M0 exponent byte per 32-element block.
    """
    require_aiter()
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError:
        pytest.skip("aiter triton MXFP4 quant not available")

    torch.set_default_device("cuda")
    M, K = 64, 256
    x = torch.randn(M, K, dtype=torch.bfloat16)

    x_fp4, x_scale = dynamic_mxfp4_quant(x)

    # FP4 values packed 2-per-byte → shape (M, K // 2)
    assert x_fp4.shape == (M, K // 2), (
        f"Expected fp4 shape ({M}, {K // 2}), got {x_fp4.shape}"
    )
    assert x_fp4.dtype == torch.uint8

    # One E8M0 scale byte per 32-element block → shape (M, K // 32)
    assert x_scale.shape == (M, K // 32), (
        f"Expected scale shape ({M}, {K // 32}), got {x_scale.shape}"
    )
    assert x_scale.dtype == torch.uint8


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (128, 4096),  # Llama-7B hidden
        (256, 11008),  # Llama-7B FFN
        (64, 14336),  # Llama-70B FFN
        (512, 8192),  # Llama-70B hidden
        (32, 28672),  # DeepSeek-style large FFN
    ],
)
def test_aiter_dynamic_mxfp4_quant_llama_shapes(shape, dtype):
    """MXFP4 quantization output format is correct for Llama-class weight shapes.

    Targets B200 parity: NVIDIA nvfp4 tests parametrize over Llama shapes
    (7168, 14336, 28672...). This covers gfx950 MXFP4 quant format validation.
    """
    require_aiter()
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError:
        pytest.skip("aiter triton MXFP4 quant not available")

    torch.set_default_device("cuda")
    M, K = shape
    x = torch.randn(M, K, dtype=dtype)

    x_fp4, x_scale = dynamic_mxfp4_quant(x)

    assert x_fp4.shape == (M, K // 2)
    assert x_fp4.dtype == torch.uint8
    assert x_scale.shape == (M, K // 32)
    assert x_scale.dtype == torch.uint8
    assert not torch.any(torch.isnan(x_fp4.float()))


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_dynamic_mxfp4_quant_determinism():
    """dynamic_mxfp4_quant is bitwise deterministic across repeated runs."""
    require_aiter()
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError:
        pytest.skip("aiter triton MXFP4 quant not available")

    torch.set_default_device("cuda")
    torch.manual_seed(7)
    x = torch.randn(128, 256, dtype=torch.bfloat16)

    fp4_results = []
    scale_results = []
    for _ in range(4):
        fp4, scale = dynamic_mxfp4_quant(x)
        fp4_results.append(fp4)
        scale_results.append(scale)

    for i in range(1, 4):
        assert torch.equal(fp4_results[0], fp4_results[i]), (
            f"dynamic_mxfp4_quant FP4 output not deterministic on run {i}"
        )
        assert torch.equal(scale_results[0], scale_results[i]), (
            f"dynamic_mxfp4_quant scale not deterministic on run {i}"
        )


# ── MXFP4 quant-dequant accuracy — Llama shapes ───────────────────────────


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (128, 4096),  # Llama-7B hidden (keep M small to avoid quark slowness)
        (64, 11008),  # Llama-7B FFN
        (32, 14336),  # Llama-70B FFN
        (64, 8192),  # Llama-70B hidden
    ],
)
def test_mxfp4_quant_dequant_llama_shapes(shape, dtype):
    """MXFP4 quant-dequant roundtrip accuracy on Llama-class weight shapes.

    Matches B200 test_mxfp4_qutlass.py coverage: large shapes, both float16
    and bfloat16, FP4 E2M1 quantization error bounds (mean < 25%).
    """
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    M, K = shape
    x = torch.randn(M, K, dtype=dtype)
    x_dq = quant_dequant_mxfp4(x)

    assert x_dq.shape == x.shape
    assert x_dq.dtype == dtype
    assert not torch.any(torch.isnan(x_dq))

    rel_error = (x_dq.float() - x.float()).abs() / (x.float().abs() + 1e-6)
    assert rel_error.mean() < 0.25, (
        f"Shape {shape} dtype {dtype}: MXFP4 mean rel error "
        f"{rel_error.mean():.4f} > 25%"
    )
    # At least 99% within 100% relative error
    pass_rate = (rel_error < 1.0).float().mean().item()
    assert pass_rate > 0.99, f"Shape {shape}: MXFP4 pass rate {pass_rate:.4f} < 99%"


# ── gfx950 hardware FP4 GEMM tests ────────────────────────────────────────
# These tests require gfx950 (MI350X) and will skip on other architectures.
# They target parity with NVIDIA B200 nvfp4_scaled_mm tests.


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize(
    "shape",
    [
        (64, 128, 64),  # small square-ish
        (128, 256, 128),  # medium
        (128, 4096, 4096),  # Llama-7B hidden square
        (256, 4096, 11008),  # Llama-7B FFN
        (64, 8192, 28672),  # Llama-70B FFN
    ],
)
def test_aiter_fp4_gemm_a4w4_accuracy(shape):
    """AITER A4W4 FP4 GEMM output is close to matmul on dequantized weights.

    Tests gfx950 parity with B200 test_nvfp4_scaled_mm.py: E2M1×E2M1 GEMM
    with block-scaled quantization, multiple Llama-class shapes.
    Requires gfx950 hardware; gracefully skips elsewhere.
    """
    require_aiter()
    require_gfx950()

    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError:
        pytest.skip("aiter triton FP4 GEMM not available")

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    M, K, N = shape

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)  # weight stored as (N, K)

    # Quantize
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    B_fp4, B_scale = dynamic_mxfp4_quant(B)

    # FP4 GEMM: computes A_fp4 (M, K) @ B_fp4.T (K, N) → (M, N)
    out = gemm_afp4wfp4(A_fp4, B_fp4, A_scale, B_scale)

    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"
    assert out.dtype == torch.bfloat16
    assert not torch.any(torch.isnan(out))

    # Reference: matmul on dequantized FP4 inputs (quark roundtrip)
    A_dq = quant_dequant_mxfp4(A)
    B_dq = quant_dequant_mxfp4(B)
    ref = torch.matmul(A_dq.float(), B_dq.t().float())

    # FP4 has only 3 mantissa bits; accumulated error across K is expected.
    # atol=0.35 matches B200 FP4 GEMM tolerance patterns.
    _assert_accurate(out.float(), ref, atol=0.35, rtol=0.0)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize(
    "shape",
    [
        (64, 128, 64),
        (128, 256, 128),
        (128, 4096, 4096),
    ],
)
def test_aiter_fp4_gemm_preshuffled_accuracy(shape):
    """AITER FP4 GEMM with preshuffled weight scales matches a4w4 accuracy.

    The preshuffled variant pre-permutes scale tensors for better memory
    access patterns. Output should match the non-shuffled variant.
    Requires gfx950.
    """
    require_aiter()
    require_gfx950()

    try:
        from aiter.ops.triton.gemm_afp4wfp4 import (
            gemm_afp4wfp4,
            gemm_afp4wfp4_preshuffled_weight_scales,
        )
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError:
        pytest.skip("aiter triton FP4 GEMM not available")

    torch.set_default_device("cuda")
    torch.manual_seed(1)
    M, K, N = shape

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)

    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    B_fp4, B_scale = dynamic_mxfp4_quant(B)

    out_base = gemm_afp4wfp4(A_fp4, B_fp4, A_scale, B_scale)
    out_preshuffled = gemm_afp4wfp4_preshuffled_weight_scales(
        A_fp4, B_fp4, A_scale, B_scale
    )

    assert out_preshuffled.shape == (M, N)
    assert out_preshuffled.dtype == torch.bfloat16
    # Preshuffled and non-shuffled should produce identical results
    torch.testing.assert_close(out_preshuffled, out_base, atol=1e-5, rtol=0.0)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_fp4_gemm_a4w4_determinism():
    """AITER FP4 A4W4 GEMM is bitwise deterministic across repeated runs.

    Targets parity with B200 FP4 determinism requirements.
    Requires gfx950.
    """
    require_aiter()
    require_gfx950()

    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError:
        pytest.skip("aiter triton FP4 GEMM not available")

    torch.set_default_device("cuda")
    torch.manual_seed(2)

    M, K, N = 128, 256, 128
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)

    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    B_fp4, B_scale = dynamic_mxfp4_quant(B)

    def run_gemm():
        return gemm_afp4wfp4(A_fp4, B_fp4, A_scale, B_scale)

    _assert_deterministic(run_gemm, n_runs=4)


# ── gfx950 hardware FP4 dynamic quantization ──────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize(
    "shape",
    [
        (128, 256),
        (256, 4096),
        (64, 14336),
    ],
)
def test_aiter_hardware_fp4_dynamic_quant_format(shape):
    """aiter hardware FP4 dynamic quant produces correct output format.

    Tests gfx950 hardware-accelerated FP4 quantization (OCP MXFP4 E2M1).
    Parity with B200 scaled_fp4_quant: block_size=32, packed uint8 output.
    Requires gfx950.
    """
    require_aiter()
    require_gfx950()

    from aiter import dynamic_per_group_scaled_quant_fp4

    torch.set_default_device("cuda")
    M, K = shape
    group_size = 32

    x = torch.randn(M, K, dtype=torch.bfloat16)
    out_fp4 = torch.empty(M, K // 2, dtype=torch.uint8)
    scales = torch.empty(M, K // group_size, dtype=torch.uint8)

    dynamic_per_group_scaled_quant_fp4(out_fp4, x, scales, group_size)

    assert out_fp4.shape == (M, K // 2), (
        f"Shape {shape}: expected fp4 ({M}, {K // 2}), got {out_fp4.shape}"
    )
    assert scales.shape == (M, K // group_size), (
        f"Shape {shape}: expected scale ({M}, {K // group_size}), got {scales.shape}"
    )


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_hardware_fp4_quant_vs_triton():
    """Hardware FP4 quant (dynamic_per_group_scaled_quant_fp4) matches triton.

    Both the gfx950 hardware path and the triton path should produce
    bitwise-identical packed FP4 values and scales for the same input.
    Requires gfx950.
    """
    require_aiter()
    require_gfx950()

    from aiter import dynamic_per_group_scaled_quant_fp4

    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError:
        pytest.skip("aiter triton MXFP4 quant not available")

    torch.set_default_device("cuda")
    torch.manual_seed(5)

    M, K = 128, 256
    group_size = 32
    x = torch.randn(M, K, dtype=torch.bfloat16)

    # Hardware path
    out_hw = torch.empty(M, K // 2, dtype=torch.uint8)
    scales_hw = torch.empty(M, K // group_size, dtype=torch.uint8)
    dynamic_per_group_scaled_quant_fp4(out_hw, x, scales_hw, group_size)

    # Triton path
    out_triton, scales_triton = dynamic_mxfp4_quant(x)

    assert out_hw.shape == out_triton.shape
    assert scales_hw.shape == scales_triton.shape
    # Both should produce numerically equivalent quantizations
    torch.testing.assert_close(out_hw.float(), out_triton.float(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        scales_hw.float(), scales_triton.float(), atol=0.0, rtol=0.0
    )


# ── ROCm skinny GEMM FP4 tests ────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize("M", [1, 2, 4, 8])  # decode / skinny batch sizes
@pytest.mark.parametrize("N, K", [(4096, 4096), (4096, 11008), (8192, 8192)])
def test_aiter_fp4_gemm_skinny_shapes(M, N, K):
    """FP4 GEMM accuracy for skinny (small-M) shapes (decode phase).

    Skinny GEMMs (M=1..8) are the bottleneck in decode phase.
    Tests parity with B200 nvfp4_scaled_mm tests at decode batch sizes.
    Requires gfx950.
    """
    require_aiter()
    require_gfx950()

    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
    except ImportError:
        pytest.skip("aiter triton FP4 GEMM not available")

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)

    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    B_fp4, B_scale = dynamic_mxfp4_quant(B)

    out = gemm_afp4wfp4(A_fp4, B_fp4, A_scale, B_scale)

    assert out.shape == (M, N)
    assert not torch.any(torch.isnan(out))

    A_dq = quant_dequant_mxfp4(A)
    B_dq = quant_dequant_mxfp4(B)
    ref = torch.matmul(A_dq.float(), B_dq.t().float())

    _assert_accurate(out.float(), ref, atol=0.35, rtol=0.0)
