#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the hipBLASLt W4A16 path against the RDNAHybrid reference.

Opt-in: needs a built rocm-libraries checkout whose hipBLASLt exposes the w4a16
API, and the process must resolve that libhipblaslt (see hipblaslt_w4a16.py):

    VLLM_HIPBLASLT_W4A16_ROOT=<root> \\
    LD_LIBRARY_PATH=<root>/build/projects/hipblaslt/library \\
    pytest tests/kernels/quantization/test_hipblaslt_w4a16.py
"""

import importlib

import pytest
import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

if not envs.VLLM_HIPBLASLT_W4A16_ROOT:
    pytest.skip("VLLM_HIPBLASLT_W4A16_ROOT is unset", allow_module_level=True)

# Same directory, and pytest's default import mode puts it on sys.path.
from test_rdna_hybrid_w4a16 import (  # noqa: E402
    _pack_zp_rows_for_kernel,
    _rdna_hybrid_w4a16_reference,
)

hipblaslt_module = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.hipblaslt_w4a16"
)
hybrid_module = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16"
)

device = "cuda"


def _build_inputs(M, N, K, group_size, dtype, has_zp):
    """Weights, scales and zero-points exactly as the layer hands them over."""
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    # hipBLASLt dropped the ExLlama encoding; its weights are plain K order.
    w_q = hybrid_module.pack_int4_plain(w_int4_nk).contiguous()
    scales_nkg = (
        0.05 * torch.rand((N, K // group_size), device=device, dtype=torch.float32)
    ).to(dtype)

    zp_nkg = None
    w_zp = None
    if has_zp:
        zp_nkg = torch.randint(
            0, 16, (N, K // group_size), device=device, dtype=torch.int32
        )
        w_zp = _pack_zp_rows_for_kernel(zp_nkg)

    scale_buf, _ = hipblaslt_module.build_scale_buffer(scales_nkg, w_zp, group_size)
    return x_mk, w_int4_nk, w_q.view(torch.int8), scales_nkg, zp_nkg, scale_buf


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", hybrid_module.SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize("M", [1, 5, 64, 256], ids=["M=1", "M=5", "M=64", "M=256"])
def test_hipblaslt_w4a16_matches_reference(dtype, group_size, has_zp, M):
    """The hipBLASLt GEMM reproduces (nibble - zp) * scale over the same buffers.

    hipBLASLt does not ship a kernel for every (group size, encoding, dtype)
    combination yet. A missing one must surface as a hard error naming the
    shape, not as a silent fallback, so that is asserted rather than skipped.
    """
    set_random_seed(0)
    K, N = 1024, 256

    x_mk, w_int4_nk, w_q, scales_nkg, zp_nkg, scale_buf = _build_inputs(
        M, N, K, group_size, dtype, has_zp
    )

    try:
        out = hipblaslt_module.hipblaslt_w4a16_gemm(
            x_mk, w_q, scale_buf, group_size, has_zp
        )
    except RuntimeError as e:
        if "no solution for" not in str(e):
            raise
        pytest.xfail(f"hipBLASLt has no kernel for this combination: {e}")

    ref = _rdna_hybrid_w4a16_reference(
        x_mk, w_int4_nk, scales_nkg, zp_nkg, group_size, bias=None
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("group_size", [32, 128])
def test_build_scale_buffer_zero_point_layout(group_size):
    """The zero-point region is vLLM's packing transposed, nibbles untouched.

    Checked against hipBLASLt's own addressing (w4a16_datagen.hpp): the
    zero-point for (n, g) is at byte (n//2)*ceil(K/G) + g of the region, in the
    low nibble for even n.
    """
    K, N = 512, 32
    num_groups = K // group_size
    scales = torch.zeros((N, num_groups), dtype=torch.float16, device=device)
    zp_nkg = torch.randint(0, 16, (N, num_groups), device=device, dtype=torch.int32)

    buf, view = hipblaslt_module.build_scale_buffer(
        scales, _pack_zp_rows_for_kernel(zp_nkg), group_size
    )
    assert view.data_ptr() == buf.data_ptr()

    scale_bytes = N * num_groups * scales.element_size()
    align = hipblaslt_module.ZERO_POINT_ALIGNMENT
    zp_region = buf[-(-scale_bytes // align) * align :].cpu()

    n = torch.arange(N).repeat_interleave(num_groups)
    g = torch.arange(num_groups).repeat(N)
    byte = zp_region[(n // 2) * num_groups + g].to(torch.int32)
    got = torch.where(n % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
    torch.testing.assert_close(got, zp_nkg.cpu().reshape(-1))
