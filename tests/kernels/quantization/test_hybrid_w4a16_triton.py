#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton path of the hybrid W4A16 kernel.

The hybrid kernel stores weights in ExLlama shuffle format [N, K//8] int32.
This test validates the Triton GEMM (triton_w4a16_skinny_fmt_gemm) that reads
from this layout.

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
    "vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16"
)
triton_w4a16_skinny_fmt_gemm = hybrid_w4a16_module.triton_w4a16_skinny_fmt_gemm


pack_int4_exllama_shuffle = hybrid_w4a16_module.pack_int4_exllama_shuffle


def _pack_exllama_shuffle(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack [K, N] int4 values into ExLlama shuffle format [N, K//8] int32."""
    return pack_int4_exllama_shuffle(w_int4_kn.t().contiguous())


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
        zp_bias=8,
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
        zp=zp,
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
# Performance regression test
# ---------------------------------------------------------------------------

# Reference TFLOPS measured on gfx1151 (Strix Halo, 40 CUs) with the
# tuned kernel (num_stages=1, UNROLL_K=4, BM=64/BN=256/BK=64/w=8 for
# M>1024).
# Key: (M, K, N, group_size, has_zp) -> reference TFLOPS
_PERF_REFERENCE_TFLOPS: dict[tuple[int, int, int, int, bool], float] = {
    # Qwen2.5-7B shapes — symmetric (compressed-tensors w4a16)
    (1606, 3584, 37888, 128, False): 25.0,
    (1606, 3584, 18944, 128, False): 26.0,
    (1606, 3584, 4608, 128, False): 27.0,
    (1606, 3584, 3584, 128, False): 26.0,
    # Qwen2.5-7B shapes — asymmetric (AWQ, zero_point=True)
    (1606, 3584, 37888, 128, True): 24.5,
    (1606, 3584, 18944, 128, True): 24.5,
    (1606, 3584, 4608, 128, True): 25.5,
    (1606, 3584, 3584, 128, True): 24.5,
}

PERF_TOLERANCE = 0.05  # 5% relative tolerance


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("has_zp", [False, True], ids=["symmetric", "asymmetric"])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1606, 3584, 37888, 128),
        (1606, 3584, 18944, 128),
        (1606, 3584, 4608, 128),
        (1606, 3584, 3584, 128),
    ],
)
def test_triton_w4a16_prefill_perf_regression(M, K, N, G, has_zp):
    """Fail if prefill TFLOPS drops more than 5% below reference."""
    triton_testing = pytest.importorskip("triton.testing")

    ref_tflops = _PERF_REFERENCE_TFLOPS[(M, K, N, G, has_zp)]
    num_groups = K // G

    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b_q_i32 = torch.randint(0, 2**31, (N, K // 8), dtype=torch.int32, device=device)
    scales = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.01
    zp = None
    if has_zp:
        zp = torch.randint(0, 16, (N, num_groups), dtype=torch.int32, device=device).to(
            torch.float16
        )

    def run():
        triton_w4a16_skinny_fmt_gemm(a, b_q_i32, scales, G, zp=zp)

    # Warm up to trigger Triton JIT compilation before timing.
    for _ in range(3):
        run()
    torch.accelerator.synchronize()

    ms = triton_testing.do_bench(run, warmup=50, rep=100)
    tflops = (2 * M * N * K) * 1e-12 / (ms * 1e-3)

    mode = "asymmetric" if has_zp else "symmetric"
    min_tflops = ref_tflops * (1 - PERF_TOLERANCE)
    assert tflops >= min_tflops, (
        f"Performance regression ({mode}): {tflops:.2f} TFLOPS < "
        f"{min_tflops:.2f} TFLOPS (reference {ref_tflops:.1f}, "
        f"tolerance {PERF_TOLERANCE * 100:.0f}%) for "
        f"M={M} K={K} N={N} G={G} ({ms:.3f} ms)"
    )
