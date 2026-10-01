#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm Hybrid W4A16 kernel (HIP skinny + Triton prefill).

Run `pytest tests/kernels/quantization/test_rdna_hybrid_w4a16.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

from vllm.platforms.rocm import on_gfx1x  # noqa: E402

device = "cuda"

hybrid_module = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16"
)
RDNAHybridW4A16LinearKernel = hybrid_module.RDNAHybridW4A16LinearKernel
pack_int4_exllama_shuffle = hybrid_module.pack_int4_exllama_shuffle
SUPPORTED_GROUP_SIZES = hybrid_module.SUPPORTED_GROUP_SIZES
MAX_SKINNY_BATCH_SIZE = hybrid_module.MAX_SKINNY_BATCH_SIZE

# Per-channel (-1) is symmetric-only (see can_implement's zero_points guard),
# so it is only added to the group-size axis of tests that stay on the
# symmetric path. test_rdna_hybrid_w4a16_process_weights_asymmetric_repack
# must NOT use this -- per-channel asymmetric must be rejected, not tested
# as if it worked.
SYM_GROUP_SIZES = SUPPORTED_GROUP_SIZES + [-1]


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _pack_zp_rows_for_kernel(zp_nkg: torch.Tensor) -> torch.Tensor:
    """Pack raw uint4 zero points along N: [N, G] int32 -> [N//8, G] int32.

    Mirrors the layout the HIP skinny kernel reads (and that
    ``process_weights_after_loading`` produces): row n's nibble lives in
    word[n//8] at bits 4*(n%8).
    """
    assert zp_nkg.dtype == torch.int32
    N, G = zp_nkg.shape
    assert N % 8 == 0
    shifts = (torch.arange(8, device=zp_nkg.device, dtype=torch.int32) * 4)[:, None]
    return torch.sum(
        (zp_nkg.view(N // 8, 8, G) & 0xF) << shifts, dim=1, dtype=torch.int32
    ).contiguous()


def _build_packed_scale_zp(
    scales_nkg: torch.Tensor, zp_nkg: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Build the Triton scale/zp carrier exactly as the layer does.

    fp16 packs scale | bias_eff, bf16 packs scale | zp_int; see the kernel
    docstring for how each is consumed.
    """
    scale_u16 = scales_nkg.view(torch.uint16).to(torch.int32) & 0xFFFF
    if dtype == torch.float16:
        w_s_f32 = scales_nkg.to(torch.float32)
        scaled_zp_f32 = (zp_nkg.to(torch.float32) - 8.0) * w_s_f32
        bias_eff = (-(8.0 * w_s_f32 + scaled_zp_f32)).to(dtype)
        hi_u16 = bias_eff.contiguous().view(torch.uint16).to(torch.int32) & 0xFFFF
    else:
        hi_u16 = zp_nkg.to(torch.int32) & 0xFFFF
    return ((hi_u16 << 16) | scale_u16).view(torch.float32).contiguous()


def _rdna_hybrid_w4a16_reference(
    x_mk: torch.Tensor,
    w_int4_nk: torch.Tensor,
    scales_nkg: torch.Tensor,
    zp_nkg: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Reference for the Hybrid W4A16 op.

    x_mk: [M, K] fp16/bf16
    w_int4_nk: [N, K] int32 with raw uint4 values in [0, 15]
    scales_nkg: [N, K//G] fp16/bf16
    zp_nkg: [N, K//G] int32 raw zero points in [0, 15],
            or None for symmetric (uint4b8, dequant subtracts 8)
    group_size: -1 (per-channel, resolved to a single group of size K) or one
            of SUPPORTED_GROUP_SIZES.
    """
    N, K = w_int4_nk.shape
    G = K if group_size == -1 else group_size
    assert K % G == 0
    s_full = scales_nkg.repeat_interleave(G, dim=1).to(torch.float32)  # [N, K]
    if zp_nkg is None:
        z_full = torch.full((N, K), 8.0, device=x_mk.device, dtype=torch.float32)
    else:
        z_full = zp_nkg.repeat_interleave(G, dim=1).to(torch.float32)
    w_fp = (w_int4_nk.to(torch.float32) - z_full) * s_full  # [N, K]
    out = x_mk.to(torch.float32) @ w_fp.t()  # [M, N]
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(x_mk.dtype)


# ---------------------------------------------------------------------------
# Forward correctness: decode (HIP skinny) + prefill (Triton)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", SYM_GROUP_SIZES)
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize(
    "M",
    [1, MAX_SKINNY_BATCH_SIZE, MAX_SKINNY_BATCH_SIZE + 1, 64],
    ids=["M=1_decode", "M=5_decode", "M=6_prefill", "M=64_prefill"],
)
def test_rdna_hybrid_w4a16_apply_matches_reference(dtype, group_size, has_zp, M):
    """Smoke test the registered custom op for both decode and prefill batches.

    Verifies the dispatch logic in `_rdna_hybrid_w4a16_apply_impl`:
      - M <= MAX_SKINNY_BATCH_SIZE: HIP wvSplitK_int4_g
      - M > MAX_SKINNY_BATCH_SIZE: Triton prefill kernel

    group_size=-1 (per-channel) is symmetric-only (can_implement rejects it
    with zero_points=True), so the has_zp=True combination is skipped for it
    rather than exercised.
    """
    if has_zp and group_size == -1:
        pytest.skip("per-channel (group_size=-1) W4A16 is symmetric-only")

    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    set_random_seed(0)

    K, N = 1024, 256
    G = K if group_size == -1 else group_size
    assert K % G == 0 and K % 8 == 0 and N % 8 == 0

    # Activations.
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)

    # Weights as raw uint4 in [N, K], packed to ExLlama shuffle [N, K//8].
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    w_q_i32 = pack_int4_exllama_shuffle(w_int4_nk).contiguous()  # [N, K//8] int32
    w_q = w_q_i32.view(torch.int8)  # same bytes viewed as int8 [N, K//2]

    # Scales [N, K//G] in act dtype.
    scales_nkg = (
        0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)
    ).to(dtype)

    # Optional raw zero points [N, K//G] as int32 nibbles. The kernels consume
    # them packed 8x along N (skinny) and folded into the carrier (Triton).
    if has_zp:
        zp_nkg = torch.randint(0, 16, (N, K // G), device=device, dtype=torch.int32)
        w_zp = _pack_zp_rows_for_kernel(zp_nkg)
        packed_scale_zp = _build_packed_scale_zp(scales_nkg, zp_nkg, dtype)
    else:
        zp_nkg = None
        w_zp = None
        packed_scale_zp = None

    from vllm.utils.platform_utils import num_compute_units

    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        x_mk,
        w_q,
        scales_nkg,
        w_q_i32,
        w_zp,
        None,  # bias
        num_compute_units(),
        group_size,
        packed_scale_zp,
    )

    ref = _rdna_hybrid_w4a16_reference(
        x_mk, w_int4_nk, scales_nkg, zp_nkg, group_size, bias=None
    )

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, MAX_SKINNY_BATCH_SIZE + 1])
def test_rdna_hybrid_w4a16_apply_with_bias(dtype, M):
    """Bias is added correctly on both decode and prefill paths."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    set_random_seed(0)
    K, N, G = 1024, 128, 128

    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    w_q_i32 = pack_int4_exllama_shuffle(w_int4_nk).contiguous()
    w_q = w_q_i32.view(torch.int8)
    scales_nkg = (
        0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)
    ).to(dtype)
    bias = torch.randn(N, device=device, dtype=dtype) * 0.1

    from vllm.utils.platform_utils import num_compute_units

    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        x_mk,
        w_q,
        scales_nkg,
        w_q_i32,
        None,
        bias,
        num_compute_units(),
        G,
    )
    ref = _rdna_hybrid_w4a16_reference(x_mk, w_int4_nk, scales_nkg, None, G, bias=bias)

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# pack_int4_exllama_shuffle round-trips correctly
# ---------------------------------------------------------------------------


def test_pack_int4_exllama_shuffle_layout():
    """Pack 8 K-values per int32 in interleave [0,2,4,6,1,3,5,7] order."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    set_random_seed(0)
    N, K = 4, 16
    w = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    packed = pack_int4_exllama_shuffle(w)
    assert packed.shape == (N, K // 8) and packed.dtype == torch.int32

    # Manual unshuffle using ExLlama shifts [0,16,4,20,8,24,12,28].
    shifts = torch.tensor(
        [0, 16, 4, 20, 8, 24, 12, 28], device=device, dtype=torch.int32
    )
    unshuffled = (packed.unsqueeze(-1) >> shifts) & 0xF
    unshuffled = unshuffled.reshape(N, K)
    torch.testing.assert_close(unshuffled, w)


# ---------------------------------------------------------------------------
# process_weights_after_loading: layout repack and zp normalization
# ---------------------------------------------------------------------------


def _pack_int4_along_k_to_ckpt(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack int4 values along K into CT checkpoint layout: [K,N] -> [N, K//8]."""
    assert w_int4_kn.dtype == torch.int32
    K, N = w_int4_kn.shape
    assert K % 8 == 0
    out = torch.zeros((N, K // 8), dtype=torch.int32, device=w_int4_kn.device)
    for i in range(8):
        out |= (w_int4_kn[i::8, :].t() & 0xF) << (i * 4)
    return out.contiguous()


def _pack_int4_along_n_for_zp(zp_int4_gn: torch.Tensor) -> torch.Tensor:
    """Pack int4 zero points along N: [G, N] -> [G, N//8] int32 (CT layout)."""
    assert zp_int4_gn.dtype == torch.int32
    G, N = zp_int4_gn.shape
    assert N % 8 == 0
    shifts = torch.arange(8, device=zp_int4_gn.device, dtype=torch.int32) * 4
    return torch.sum(
        (zp_int4_gn.view(G, N // 8, 8) & 0xF) << shifts, dim=2, dtype=torch.int32
    ).contiguous()


def _build_dummy_layer(
    w_ckpt_nk8: torch.Tensor,
    scales_ckpt_nkg: torch.Tensor,
    zeros_ckpt: torch.Tensor | None,
):
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedColumnParameter,
        PackedvLLMParameter,
    )

    weight_loader = lambda *args, **kwargs: None

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "weight_packed",
        PackedvLLMParameter(
            data=w_ckpt_nk8,
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
            packed_factor=8,
            packed_dim=1,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        GroupQuantScaleParameter(
            data=scales_ckpt_nkg,
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
        ),
    )
    if zeros_ckpt is not None:
        layer.register_parameter(
            "weight_zero_point",
            PackedColumnParameter(
                data=zeros_ckpt,
                weight_loader=weight_loader,
                output_dim=0,
                packed_factor=8,
                packed_dim=0,
            ),
        )
    return layer


@pytest.mark.parametrize("group_size", SYM_GROUP_SIZES)
def test_rdna_hybrid_w4a16_process_weights_symmetric_repack(group_size, dist_init):
    """uint4b8 (symmetric): w_q -> [N, K//8] int8 ExLlama shuffle, no zp param.

    group_size=-1 (per-channel) needs no special case here: process_weights_
    after_loading never divides by group_size -- the group count comes from
    the loaded scale tensor's shape -- so a per-channel scale ([N, 1]) flows
    through the same repack path as a grouped one.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    set_random_seed(0)

    K, N = 256, 128
    G = K if group_size == -1 else group_size
    assert K % G == 0

    # Reference unpacked weights, then pack into CT checkpoint layout [N, K//8].
    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    w_ckpt_nk8 = _pack_int4_along_k_to_ckpt(w_int4_kn)
    scales_ckpt_nkg = 0.05 * torch.rand((N, K // G), device=device, dtype=torch.float16)

    layer = _build_dummy_layer(w_ckpt_nk8, scales_ckpt_nkg, zeros_ckpt=None)

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
    )
    kernel = RDNAHybridW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name=None,
        w_gidx_param_name=None,
    )
    kernel.process_weights_after_loading(layer)

    # Skinny weight is stored once as int8 [N, K//2]; the Triton path
    # reinterprets it as int32 [N, K//8] via a view (no separate parameter).
    assert layer.weight_packed.dtype == torch.int8
    assert tuple(layer.weight_packed.shape) == (N, K // 2)
    w_q_i32 = layer.weight_packed.view(torch.int32)
    assert tuple(w_q_i32.shape) == (N, K // 8)

    expected_packed = pack_int4_exllama_shuffle(w_int4_kn.t().contiguous())
    torch.testing.assert_close(w_q_i32, expected_packed)

    # Scales: [N, K//G] (skinny layout, no transpose since CT already had it).
    assert tuple(layer.weight_scale.shape) == (N, K // G)
    torch.testing.assert_close(layer.weight_scale, scales_ckpt_nkg)


@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
def test_rdna_hybrid_w4a16_process_weights_asymmetric_repack(group_size, dist_init):
    """uint4 (asymmetric): zero points unpacked to raw values in act dtype."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    set_random_seed(0)

    K, N = 256, 128
    G = group_size
    assert K % G == 0 and N % 8 == 0

    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    w_ckpt_nk8 = _pack_int4_along_k_to_ckpt(w_int4_kn)
    scales_ckpt_nkg = 0.05 * torch.rand((N, K // G), device=device, dtype=torch.float16)

    # CT zero-point layout is N-packed: [N//8, K//G] int32.
    zeros_int4_gn = torch.randint(0, 16, (K // G, N), device=device, dtype=torch.int32)
    zeros_packed_gn8 = _pack_int4_along_n_for_zp(zeros_int4_gn)  # [K//G, N//8]
    zeros_ckpt_n8kg = zeros_packed_gn8.t().contiguous()  # [N//8, K//G]

    layer = _build_dummy_layer(w_ckpt_nk8, scales_ckpt_nkg, zeros_ckpt=zeros_ckpt_n8kg)

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=G,
        zero_points=True,
        has_g_idx=False,
    )
    kernel = RDNAHybridW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
        w_gidx_param_name=None,
    )
    kernel.process_weights_after_loading(layer)

    # Zero-points stay in their PACKED 4-bit form -- [N//8, K//G] int32, row n's
    # nibble at word[n//8] bits 4*(n%8) -- which is what the HIP skinny kernel
    # reads. Expanding them to the activation dtype would cost 4x the DRAM
    # traffic on a memory-bound kernel (and the C++ rejects that dtype).
    assert layer.weight_zero_point.dtype == torch.int32
    assert tuple(layer.weight_zero_point.shape) == (N // 8, K // G)
    torch.testing.assert_close(layer.weight_zero_point, zeros_ckpt_n8kg)

    # The Triton prefill path reads the scale/zp carrier instead of two loads;
    # asymmetric layers must have it built at load time.
    assert hasattr(layer, "_hybrid_w_packed_scale_zp")
    assert tuple(layer._hybrid_w_packed_scale_zp.shape) == (N, K // G)

    # Quantized weights match symmetric path's layout regardless of zp.
    w_q_i32 = layer.weight_packed.view(torch.int32)
    assert tuple(w_q_i32.shape) == (N, K // 8)
    expected_packed = pack_int4_exllama_shuffle(w_int4_kn.t().contiguous())
    torch.testing.assert_close(w_q_i32, expected_packed)


# ---------------------------------------------------------------------------
# can_implement enforces the supported-group-size policy
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize(
    "group_size,expected_ok",
    [(32, True), (64, True), (128, True), (256, False), (-1, True)],
)
def test_hybrid_can_implement_group_size(group_size, expected_ok):
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    K, N = 1024, 256
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
    )
    ok, _ = RDNAHybridW4A16LinearKernel.can_implement(config)
    assert ok is expected_ok


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
def test_hybrid_can_implement_per_channel_rejects_asymmetric():
    """Per-channel (group_size=-1) is symmetric-only: asymmetric zero points
    must be rejected rather than silently accepted."""
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    K, N = 1024, 256
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=-1,
        zero_points=True,
        has_g_idx=False,
    )
    ok, reason = RDNAHybridW4A16LinearKernel.can_implement(config)
    assert ok is False
    assert "symmetric-only" in reason


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
    """Reference for symmetric HIP skinny: C = A @ (W - zp_bias) * S.

    group_size=-1 (per-channel) resolves to a single group of size K, i.e.
    scales_nkg is [N, 1] and every column of w_int4_nk shares one scale.
    """
    K = a_mk.shape[1]
    N = w_int4_nk.shape[0]
    G = K if group_size == -1 else group_size
    num_groups = K // G

    w_fp = (w_int4_nk.to(torch.float32) - zp_bias).view(N, num_groups, G)
    s = scales_nkg.to(torch.float32).unsqueeze(-1)
    w_dequant = (w_fp * s).view(N, K)

    return (a_mk.to(torch.float32) @ w_dequant.t()).to(a_mk.dtype)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1, 256, 256, 32),
        (1, 256, 256, 64),
        (1, 512, 256, 128),
        (2, 512, 256, 64),
        (3, 256, 512, 64),
        (1, 256, 256, -1),
        (3, 512, 256, -1),
        # Deep K: exercises the kernel's separate chunked (non-"sml") compute
        # body, which carries its own GROUP_SIZE==0 instantiation distinct
        # from the shallow rows above.
        (1, 8192, 256, -1),
    ],
)
def test_hip_skinny_wvSplitK_int4_g(dtype, M, K, N, G):
    """Test HIP wvSplitK_int4_g kernel directly via _custom_ops."""
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)

    b_packed_i32 = pack_int4_exllama_shuffle(w_int4_nk)
    b_packed_i8 = b_packed_i32.view(torch.int8)

    G_resolved = K if G == -1 else G
    scales = (
        0.05 * torch.rand((N, K // G_resolved), device=device, dtype=torch.float32)
    ).to(dtype)

    cu_count = num_compute_units()
    out = ops.wvSplitK_int4_g(b_packed_i8, a, scales, cu_count, G)

    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_bias=8)

    # rtol=1e-2, atol=5e-2 holds with margin: per-channel vs group_size=128 at
    # matched deep K (both dtypes, several seeds) shows no assert_close
    # failures; worst observed per-element margin is ~+0.0077 (bf16, K=18944).
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Tests for the full hybrid dispatch (HIP decode + Triton prefill)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1, 256, 256, 64),
        (1, 512, 256, 32),
        (1, 512, 256, 128),
        (32, 512, 256, 64),
        (64, 1024, 256, 128),
        (1, 256, 256, -1),
        (3, 512, 256, -1),
        # Deep K: both M<=5 here route to the HIP skinny kernel's
        # GROUP_SIZE==0 arm, exercising its separate chunked compute body --
        # not redundant with the shallow -1 rows above.
        (1, 8192, 256, -1),
    ],
)
def test_rdna_hybrid_w4a16_dispatch(dtype, M, K, N, G):
    """Test the full hybrid dispatch via the custom op."""
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)

    b_packed_i32 = pack_int4_exllama_shuffle(w_int4_nk)
    b_packed_i8 = b_packed_i32.view(torch.int8)

    G_resolved = K if G == -1 else G
    scales = (
        0.05 * torch.rand((N, K // G_resolved), device=device, dtype=torch.float32)
    ).to(dtype)

    cu_count = num_compute_units()
    out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
        a, b_packed_i8, scales, b_packed_i32, None, None, cu_count, G
    )

    ref = _hip_skinny_reference(a, w_int4_nk, scales, group_size=G, zp_bias=8)

    # See test_hip_skinny_wvSplitK_int4_g above for the measured tolerance
    # headroom on the per-channel cases, including deep K.
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Negative tests: the macro rewrite's hard TORCH_CHECK(false, ...) defaults
# replaced a silent fall-through to group_size=128. Prove that's really gone.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("bad_group_size", [0, -2])
def test_hip_skinny_wvSplitK_int4_g_rejects_bad_group_size(bad_group_size):
    """group_size=0 or -2 must raise, not silently fall through to 128."""
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)
    K, N = 256, 256
    a = (0.25 * torch.randn((1, K), device=device, dtype=torch.float16)).to(
        torch.float16
    )
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    b_packed_i8 = pack_int4_exllama_shuffle(w_int4_nk).view(torch.int8)
    scales = (0.05 * torch.rand((N, 1), device=device, dtype=torch.float32)).to(
        torch.float16
    )

    with pytest.raises(RuntimeError, match="group_size must be"):
        ops.wvSplitK_int4_g(b_packed_i8, a, scales, num_compute_units(), bad_group_size)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
def test_hip_skinny_wvSplitK_int4_g_rejects_per_channel_with_zero_points():
    """Per-channel (group_size=-1) is symmetric-only at the host boundary too:
    passing zero_points must raise rather than dequantize incorrectly."""
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)
    K, N = 256, 256
    a = (0.25 * torch.randn((8, K), device=device, dtype=torch.float16)).to(
        torch.float16
    )
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    b_packed_i8 = pack_int4_exllama_shuffle(w_int4_nk).view(torch.int8)
    scales = (0.05 * torch.rand((N, 1), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    zp = torch.randint(0, 16, (N // 8, 1), device=device, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="symmetric-only"):
        ops.wvSplitK_int4_g(
            b_packed_i8, a, scales, num_compute_units(), -1, zero_points=zp
        )
