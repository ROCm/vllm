#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the hybrid W4A16 kernel (Triton prefill + HIP skinny decode).

The hybrid kernel stores weights in ExLlama shuffle format [N, K//8] int32.
Tests validate:
  - Triton GEMM (triton_w4a16_skinny_fmt_gemm) for prefill path
  - HIP wvSplitK_int4_g for decode path
  - Full hybrid dispatch (torch.ops.vllm.rdna_hybrid_w4a16_apply) routing

Run `pytest tests/kernels/quantization/test_hybrid_w4a16_triton.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# This test module is ROCm/Triton specific. Avoid import-time failures on
# non-ROCm or environments without Triton by skipping early.
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

device = "cuda"

hybrid_w4a16_module = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16"
)
triton_w4a16_skinny_fmt_gemm = hybrid_w4a16_module.triton_w4a16_skinny_fmt_gemm


pack_int4_exllama_shuffle = hybrid_w4a16_module.pack_int4_exllama_shuffle
padded_row_stride_bytes = hybrid_w4a16_module.padded_row_stride_bytes


def _pack_exllama_shuffle(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack [K, N] int4 values into ExLlama shuffle format [N, K//8] int32."""
    return pack_int4_exllama_shuffle(w_int4_kn.t().contiguous())


def _row_padded_empty(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    """[rows, cols] view whose row stride follows padded_row_stride_bytes."""
    elem = torch.empty((), dtype=dtype).element_size()
    ld = padded_row_stride_bytes(cols * elem) // elem
    return torch.empty((rows, ld), device=device, dtype=dtype)[:, :cols]


@pytest.mark.parametrize(
    "row_bytes,expected",
    [
        (128, 128),
        (129, 384),
        (1024, 1152),  # weights, K = 2048
        (1792, 1920),  # weights, K = 3584
        (2048, 2176),  # weights, K = 4096
        (7168, 7296),  # weights K = 14336 / bf16 activations K = 3584
        (8192, 8320),  # bf16 activations or output, K or N = 4096
        (28672, 28800),  # bf16 output, N = 14336
    ],
)
def test_padded_row_stride_bytes(row_bytes, expected):
    """Smallest stride >= row_bytes with stride % 256 == 128."""
    assert padded_row_stride_bytes(row_bytes) == expected


@pytest.mark.parametrize("K", [1536, 2048, 3584, 4096])
def test_pack_skinny_int4_row_stride_gfx115x(K):
    """gfx115x weight rows are padded to an odd multiple of 128 B, data intact."""
    from vllm.platforms.rocm import on_gfx115x

    if not on_gfx115x():
        pytest.skip("gfx115x only")
    unpacked = torch.randint(0, 16, (64, K), device=device, dtype=torch.int32)

    w_q = hybrid_w4a16_module.pack_skinny_int4(unpacked)

    assert w_q.stride(0) == padded_row_stride_bytes(K // 2)
    torch.testing.assert_close(
        w_q.view(torch.int32), pack_int4_exllama_shuffle(unpacked)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,K,N,G", [(17, 256, 512, 32), (64, 1024, 256, 128)])
def test_triton_w4a16_skinny_fmt_gemm_row_padded(dtype, M, K, N, G, random_seed: int):
    """Row-padded activations and output give the same result as dense ones."""
    set_random_seed(random_seed)

    a = _row_padded_empty(M, K, dtype)
    a.copy_(0.25 * torch.randn((M, K), device=device, dtype=torch.float32))
    w_int4 = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )
    out = _row_padded_empty(M, N, dtype)

    triton_w4a16_skinny_fmt_gemm(
        a=a, b_q=_pack_exllama_shuffle(w_int4), scales=scales, group_size=G, out=out
    )
    ref = _w4a16_skinny_reference(a, w_int4, scales, group_size=G, zp_bias=8)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


def _w4a16_skinny_reference(
    a_mk: torch.Tensor,
    w_int4_kn: torch.Tensor,
    scales_nkg: torch.Tensor,
    *,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    """Reference implementation for symmetric W4A16 with skinny layout.

    a_mk: [M, K] fp16/bf16
    w_int4_kn: [K, N] int4 values (unpacked, int32)
    scales_nkg: [N, K//G] scales (skinny layout)
    """
    M, K = a_mk.shape

    # Expand scales from [N, K//G] to [K, N]
    scales_kn = scales_nkg.t().contiguous()  # [K//G, N]
    s_full = scales_kn.repeat_interleave(group_size, dim=0).to(torch.float32)

    w_fp = (w_int4_kn - zp_bias).to(torch.float32) * s_full  # [K, N]
    out = a_mk.to(torch.float32) @ w_fp  # [M, N]
    return out.to(a_mk.dtype)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1, 256, 256, 32),
        (17, 256, 512, 32),
        (32, 512, 256, 64),
        (33, 512, 512, 128),
        (64, 1024, 256, 256),
    ],
)
def test_triton_w4a16_skinny_fmt_gemm_matches_reference(
    dtype, M, K, N, G, random_seed: int
):
    assert K % G == 0 and K % 8 == 0, (
        f"Invalid test shape: K={K} must be divisible by G={G} and 8"
    )

    set_random_seed(random_seed)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4 = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)

    # Pack into ExLlama shuffle format [N, K//8]
    b_packed = _pack_exllama_shuffle(w_int4)

    # Scales in skinny layout [N, K//G]
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    out = triton_w4a16_skinny_fmt_gemm(
        a=a,
        b_q=b_packed,
        scales=scales,
        group_size=G,
    )
    ref = _w4a16_skinny_reference(
        a,
        w_int4,
        scales,
        group_size=G,
        zp_bias=8,
    )

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


def _w4a16_skinny_reference_asymmetric(
    a_mk: torch.Tensor,
    w_int4_kn: torch.Tensor,
    scales_nkg: torch.Tensor,
    zp_raw_nkg: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Reference implementation for asymmetric W4A16 with skinny layout.

    a_mk: [M, K] fp16/bf16
    w_int4_kn: [K, N] int4 values (unpacked, int32)
    scales_nkg: [N, K//G] scales (skinny layout)
    zp_raw_nkg: [N, K//G] raw zero-points in activation dtype
    """
    # Expand scales and raw zp from [N, K//G] to [K, N]
    scales_kn = scales_nkg.t().contiguous()  # [K//G, N]
    s_full = scales_kn.repeat_interleave(group_size, dim=0).to(torch.float32)

    zp_raw_kn = zp_raw_nkg.t().contiguous()  # [K//G, N]
    zp_raw_full = zp_raw_kn.repeat_interleave(group_size, dim=0).to(torch.float32)

    # dequant: (nibble - zp_raw) * scale
    w_fp = (w_int4_kn.to(torch.float32) - zp_raw_full) * s_full  # [K, N]
    out = a_mk.to(torch.float32) @ w_fp  # [M, N]
    return out.to(a_mk.dtype)


def _pack_scale_zp(
    scales_nkg: torch.Tensor, zp_nkg: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Build the asymmetric PackedSb carrier [N, K//G] fp32 that
    triton_w4a16_skinny_fmt_gemm consumes (low 16 bits = scale; high 16 bits =
    fp16 bias_eff = -(8 + (zp-8))*scale, or bf16 integer zp). Mirrors
    RDNAHybridW4A16LinearKernel.process_weights_after_loading.
    """
    scale_u16 = scales_nkg.contiguous().view(torch.uint16).to(torch.int32) & 0xFFFF
    if dtype == torch.float16:
        s32 = scales_nkg.to(torch.float32)
        bias_eff = (-(8.0 * s32 + (zp_nkg.to(torch.float32) - 8.0) * s32)).to(dtype)
        hi_u16 = bias_eff.contiguous().view(torch.uint16).to(torch.int32) & 0xFFFF
    else:
        hi_u16 = zp_nkg.to(torch.int32) & 0xFFFF
    return ((hi_u16 << 16) | scale_u16).view(torch.float32).contiguous()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1, 256, 256, 32),
        (17, 256, 512, 32),
        (32, 512, 256, 64),
        (33, 512, 512, 128),
        (64, 1024, 256, 128),
    ],
)
def test_triton_w4a16_skinny_fmt_gemm_asymmetric(dtype, M, K, N, G, random_seed: int):
    assert K % G == 0 and K % 8 == 0, (
        f"Invalid test shape: K={K} must be divisible by G={G} and 8"
    )

    set_random_seed(random_seed)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4 = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)

    # Pack into ExLlama shuffle format [N, K//8]
    b_packed = _pack_exllama_shuffle(w_int4)

    # Scales in skinny layout [N, K//G]
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    # Raw per-group zero-points [N, K//G] in activation dtype
    zp_raw = torch.randint(0, 16, (N, K // G), device=device, dtype=torch.int32)
    zp = zp_raw.to(dtype)

    out = triton_w4a16_skinny_fmt_gemm(
        a=a,
        b_q=b_packed,
        scales=scales,
        group_size=G,
        packed_scale_zp=_pack_scale_zp(scales, zp, dtype),
    )
    ref = _w4a16_skinny_reference_asymmetric(
        a,
        w_int4,
        scales,
        zp,
        group_size=G,
    )

    # bf16 accumulation at larger shapes needs slightly looser tolerance
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Tests for the HIP wvSplitK_int4_g decode kernel
# ---------------------------------------------------------------------------


def _hip_skinny_reference(
    a_mk: torch.Tensor,
    w_int4_nk: torch.Tensor,
    scales_nkg: torch.Tensor,
    *,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    """Reference for symmetric HIP skinny: C = A @ (W - zp_bias) * S."""
    K = a_mk.shape[1]
    N = w_int4_nk.shape[0]
    num_groups = K // group_size

    w_fp = (w_int4_nk.to(torch.float32) - zp_bias).view(N, num_groups, group_size)
    s = scales_nkg.to(torch.float32).unsqueeze(-1)
    w_dequant = (w_fp * s).view(N, K)

    return (a_mk.to(torch.float32) @ w_dequant.t()).to(a_mk.dtype)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        # M<=5 ensures the HIP skinny path is taken
        (1, 256, 256, 32),
        (1, 256, 256, 64),
        (1, 512, 256, 128),
        (2, 512, 256, 64),
        (3, 256, 512, 64),
        # (1, 4096, *, 128): N=1 at K=4096 -- the small-tile decode rule on
        # gfx1150/gfx1151 (W=16, AC=32, YT=1, UN=2); the K=4096 branch on
        # the other gfx11/gfx12 targets.
        (1, 4096, 256, 128),
    ],
)
def test_hip_skinny_wvSplitK_int4_g(dtype, M, K, N, G, random_seed: int):
    """Test HIP wvSplitK_int4_g kernel directly via _custom_ops."""
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(random_seed)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)

    b_packed_i32 = pack_int4_exllama_shuffle(w_int4_nk)
    b_packed_i8 = b_packed_i32.view(torch.int8)

    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    cu_count = num_compute_units()
    out = ops.wvSplitK_int4_g(b_packed_i8, a, scales, cu_count, G)

    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_bias=8)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Tests for the full hybrid dispatch (HIP decode + Triton prefill)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        # M=1: HIP skinny decode path
        (1, 256, 256, 64),
        (1, 512, 256, 32),
        (1, 512, 256, 128),
        # M>5: Triton prefill path
        (32, 512, 256, 64),
        (64, 1024, 256, 128),
    ],
)
def test_hybrid_w4a16_dispatch(dtype, M, K, N, G, random_seed: int):
    """Test the full hybrid dispatch via the custom op."""
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(random_seed)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)

    b_packed_i32 = pack_int4_exllama_shuffle(w_int4_nk)
    b_packed_i8 = b_packed_i32.view(torch.int8)

    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    cu_count = num_compute_units()
    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        a, b_packed_i8, scales, None, None, cu_count, G
    )

    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_bias=8)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,dequant",
    [(1, False), (32, False), (256, True)],
    ids=["decode", "triton_prefill", "dequant_prefill"],
)
def test_hybrid_w4a16_dispatch_row_padded_output(dtype, M, dequant, random_seed):
    """Every path writes the output with the requested row stride, and the
    fake impl reports the same layout (opcheck)."""
    from tests.kernels.utils import opcheck
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(random_seed)
    K, N, G = 512, 256, 64
    ld = padded_row_stride_bytes(N * 2) // 2

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    b_packed_i32 = pack_int4_exllama_shuffle(w_int4_nk)
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )
    bias = torch.randn(N, device=device, dtype=dtype)
    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_bias=8) + bias
    w_dequant = (
        ((w_int4_nk - 8).float().view(N, K // G, G) * scales.float().unsqueeze(-1))
        .view(N, K)
        .to(dtype)
        if dequant
        else None
    )
    args = (
        a,
        b_packed_i32.view(torch.int8),
        scales,
        None,
        bias,
        num_compute_units(),
        G,
        None,
        w_dequant,
        ld,
    )

    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(*args)

    assert out.shape == (M, N) and out.stride() == (ld, 1)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)
    opcheck(torch.ops.vllm.rdna_hybrid_w4a16_apply, args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 32])
def test_hybrid_w4a16_dispatch_row_padded_activation(dtype, M, random_seed: int):
    """Both the decode and the prefill path accept a row-padded activation."""
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(random_seed)
    K, N, G = 512, 256, 64

    a = _row_padded_empty(M, K, dtype)
    a.copy_(0.25 * torch.randn((M, K), device=device, dtype=torch.float32))
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    b_packed_i32 = pack_int4_exllama_shuffle(w_int4_nk)
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        a,
        b_packed_i32.view(torch.int8),
        scales,
        None,
        None,
        num_compute_units(),
        G,
    )
    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_bias=8)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)
