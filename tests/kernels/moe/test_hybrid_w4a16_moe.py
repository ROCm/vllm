# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HybridW4A16MoEExperts (Triton prefill + HIP decode).

Validates the hybrid MoE kernel by:
1. Creating random fp16 MoE weights
2. Quantizing them to symmetric (or asymmetric) 4-bit, group_size=32 or 128
3. Packing into ExLlama shuffle format [E, N, K//8] int32
4. Running HybridW4A16MoEExperts via FusedMoEModularKernel
5. Comparing against torch_experts reference using dequantized weights

Tests exercise both paths:
- Decode (M<=5): HIP wvSplitK_int4 kernel
- Prefill (M>5): Triton fused_moe kernel with use_shuffle_w4a16
"""

import dataclasses
from collections.abc import Callable

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    pack_int4_exllama_shuffle,
)
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.hybrid_w4a16_moe import (
    HybridW4A16MoEExperts,
    hybrid_w4a16_moe_enabled,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.moe_wna16 import (
    awq_moe_qzeros_to_hybrid,
    awq_moe_weight_to_hybrid,
    convert_awq_tensor,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import awq_pack
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

NUM_BITS = 4
PACK_FACTOR = 32 // NUM_BITS  # 8 nibbles per int32


def _quantize_4bit_skinny(
    w: torch.Tensor,
    group_size: int,
    asym: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """4-bit quantization → skinny ExLlama format.

    Input:  w [K, N]

    Returns:
      q_skinny: [N, K//8] int32 (ExLlama shuffle packed)
      scales:   [N, K//G] (skinny layout)
      w_ref:    [K, N] (dequantized reference)
      zp:       [N//8, K//G] int32 packed zero points (asym), else None

    """
    K, N = w.shape
    assert K % group_size == 0
    num_groups = K // group_size

    w_grouped = w.reshape(num_groups, group_size, N)
    if asym:
        w_min = w_grouped.amin(dim=1, keepdim=True)
        w_max = w_grouped.amax(dim=1, keepdim=True)
        scales = ((w_max - w_min) / 15.0).clamp(min=1e-5)
        zp = torch.round(-w_min / scales).clamp(0, 15).int()
        w_q = (torch.round(w_grouped / scales).int() + zp).clamp(0, 15)
    else:
        abs_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        scales = abs_max / 7.0
        zp = torch.full_like(scales, 8, dtype=torch.int32)
        w_q = torch.round(w_grouped / scales).clamp(-7, 7).int() + 8

    # Dequantized reference
    w_ref = ((w_q - zp).float() * scales).reshape(K, N).to(w.dtype)
    w_q = w_q.reshape(K, N)

    zp_packed = None
    if asym:
        # [G, 1, N] -> [N//8, 8, G]; row n at nibble n % 8.
        zp_n = zp.squeeze(1).t().reshape(N // 8, 8, num_groups)
        shifts = (4 * torch.arange(8, device=w.device, dtype=torch.int32))[:, None]
        zp_packed = (zp_n << shifts).sum(dim=1, dtype=torch.int32)

    # Pack into ExLlama shuffle: transpose to [N, K], pack to [N, K//8]
    w_q_uint4 = w_q.to(torch.uint8)  # values in [0, 15]
    w_q_t = w_q_uint4.t().contiguous()  # [N, K]
    q_skinny = pack_int4_exllama_shuffle(w_q_t)  # [N, K//8] int32

    # Scales: [num_groups, N] → [N, num_groups] (skinny layout)
    scales_skinny = scales.squeeze(1).t().contiguous()  # [N, K//G]

    return q_skinny, scales_skinny, w_ref, zp_packed


def _make_hybrid_moe_weights(
    E: int,
    K: int,
    N: int,
    group_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    asym: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Create fake skinny-packed MoE weights for E experts.

    Returns (w_skinny, scales, w_ref, zp) where:
    - w_skinny: [E, N, K//8] int32 (ExLlama shuffle packed)
    - scales:   [E, N, K//G] (skinny layout)
    - w_ref:    [E, N, K] (torch_experts convention)
    - zp:       [E, N//8, K//G] int32 (asym), else None
    """
    experts = []
    for _ in range(E):
        w_fp = torch.randn(K, N, device=device, dtype=dtype) / 10.0
        experts.append(_quantize_4bit_skinny(w_fp, group_size, asym))

    w_skinny = torch.stack([x[0] for x in experts])
    w_scales = torch.stack([x[1] for x in experts])
    w_ref = torch.stack([x[2].t() for x in experts])  # [N, K] for torch_experts
    w_zp = torch.stack([x[3] for x in experts]) if asym else None

    return w_skinny, w_scales, w_ref, w_zp


def _make_awq_moe_weights(
    E: int,
    K: int,
    N: int,
    group_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    asym: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """AutoAWQ checkpoint tensors through the MoeWNA16 loader conversion and
    the hybrid repack. Same return contract as _make_hybrid_moe_weights.
    """
    G = K // group_size
    q = torch.randint(0, 16, (E, K, N), dtype=torch.int32)
    zp = torch.randint(0, 16, (E, G, N), dtype=torch.int32)
    scales = torch.rand(E, G, N, dtype=dtype) / 50 + 1e-3
    w_ref = (q - zp.repeat_interleave(group_size, 1)).to(dtype) * (
        scales.repeat_interleave(group_size, 1)
    )
    # Checkpoint: qweight [K, N//8], qzeros [G, N//8], scales [G, N] per expert.
    # Loader: qweight -> [N, K//2] uint8, qzeros -> [N//2, G] uint8, scales.T.
    qweight = torch.stack(
        [convert_awq_tensor(awq_pack(q[e], 4, K, N), "qweight") for e in range(E)]
    )
    qzeros = torch.stack(
        [convert_awq_tensor(awq_pack(zp[e], 4, G, N), "qzeros") for e in range(E)]
    )
    return (
        awq_moe_weight_to_hybrid(qweight.to(device)),
        scales.transpose(1, 2).contiguous().to(device),
        w_ref.transpose(1, 2).contiguous().to(device),
        awq_moe_qzeros_to_hybrid(qzeros.to(device)),
    )


def _run_hybrid_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
    force_triton: bool = False,
    force_hip: bool = False,
    dtype: torch.dtype = torch.float16,
    asym: bool = False,
    make_weights: Callable = _make_hybrid_moe_weights,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build weights, run HybridW4A16MoEExperts and torch_experts reference.

    Args:
        force_triton: Force the Triton prefill path for all batch sizes.
        force_hip: Force the HIP wvSplitK path for all batch sizes.

    Returns (hybrid_output, reference_output).

    """
    set_random_seed(1)
    device = torch.device("cuda")

    assert k % group_size == 0

    # w1: gate+up projection [E, 2*N, K//8], ref [E, 2*N, K]
    w1_skinny, w1_scales, w1_ref, w1_zp = make_weights(
        e, k, 2 * n, group_size, device, dtype, asym
    )
    # w2: down projection [E, K, N//8], ref [E, K, N]
    w2_skinny, w2_scales, w2_ref, w2_zp = make_weights(
        e, n, k, group_size, device, dtype, asym
    )

    hidden = torch.randn(m, k, device=device, dtype=dtype) / 10
    scores = torch.randn(m, e, device=device, dtype=dtype)

    topk_weights, topk_ids, _ = fused_topk(hidden, scores, topk, False)

    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        block_shape=[0, group_size],
    )

    moe_config = make_dummy_moe_config(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size=n,
        in_dtype=dtype,
    )

    experts = HybridW4A16MoEExperts(
        moe_config=moe_config,
        quant_config=quant_config,
    )

    orig_threshold = HybridW4A16MoEExperts.MAX_SKINNY_BATCH_SIZE
    if force_triton:
        HybridW4A16MoEExperts.MAX_SKINNY_BATCH_SIZE = 0
    elif force_hip:
        HybridW4A16MoEExperts.MAX_SKINNY_BATCH_SIZE = 10000

    try:
        mk = FusedMoEKernelModularImpl(
            fused_experts=experts,
            prepare_finalize=MoEPrepareAndFinalizeNoDPEPModular(),
        )

        init_workspace_manager(device)
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config):
            torch_output = torch_experts(
                hidden,
                w1_ref,
                w2_ref,
                topk_weight=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=e,
            )

            hybrid_out = mk.apply(
                hidden_states=hidden,
                w1=w1_skinny,
                w2=w2_skinny,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=e,
                expert_map=None,
                activation=MoEActivation.SILU,
                apply_router_weight_on_input=False,
            )
    finally:
        HybridW4A16MoEExperts.MAX_SKINNY_BATCH_SIZE = orig_threshold

    return hybrid_out, torch_output


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
@pytest.mark.parametrize("m", [1, 4, 16, 64])
@pytest.mark.parametrize(
    "n,k",
    [
        (256, 256),
        (512, 256),
        # Qwen3.5-A3B MoE shapes -- exercise the gfx11 K=2048 (gate_up) and
        # K=512 (down) dispatch branches added by this PR.
        (1024, 2048),
        (2048, 512),
    ],
)
@pytest.mark.parametrize("e,topk", [(8, 2), (16, 4)])
@pytest.mark.parametrize("group_size", [32, 128])
def test_hybrid_w4a16_moe(m: int, n: int, k: int, e: int, topk: int, group_size: int):
    """Test natural dispatch: HIP for decode (m<=5), Triton for prefill (m>5)."""
    hybrid_out, torch_output = _run_hybrid_moe(m, n, k, e, topk, group_size)
    torch.testing.assert_close(hybrid_out, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
def test_hybrid_w4a16_moe_small_gs_large_alignment():
    """Small group_size at >= one full alignment block of routed rows per
    expert, so prefill takes the large-alignment tiles."""
    m, n, k, e, topk, group_size = 512, 768, 2048, 8, 2, 32
    assert m * topk / e >= HybridW4A16MoEExperts.TRITON_BLOCK_SIZE_M_SMALL_GS_LARGE
    hybrid_out, torch_output = _run_hybrid_moe(m, n, k, e, topk, group_size)
    torch.testing.assert_close(hybrid_out, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
@pytest.mark.parametrize("m", [1, 4, 16])
@pytest.mark.parametrize("n,k", [(256, 256)])
@pytest.mark.parametrize("e,topk", [(8, 2)])
@pytest.mark.parametrize("group_size", [32])
def test_hybrid_w4a16_moe_force_triton(
    m: int, n: int, k: int, e: int, topk: int, group_size: int
):
    """Force the Triton path for all batch sizes (including m=1)."""
    hybrid_out, torch_output = _run_hybrid_moe(
        m, n, k, e, topk, group_size, force_triton=True
    )
    torch.testing.assert_close(hybrid_out, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
@pytest.mark.parametrize("m", [1, 16, 64])
@pytest.mark.parametrize("n,k", [(256, 256)])
@pytest.mark.parametrize("e,topk", [(8, 2)])
@pytest.mark.parametrize("group_size", [32])
def test_hybrid_w4a16_moe_force_hip(
    m: int, n: int, k: int, e: int, topk: int, group_size: int
):
    """Force the HIP wvSplitK path for all batch sizes (including m=64)."""
    hybrid_out, torch_output = _run_hybrid_moe(
        m, n, k, e, topk, group_size, force_hip=True
    )
    torch.testing.assert_close(hybrid_out, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
# m=64 / m=512 reach both small-group-size prefill alignments (64 and 128).
@pytest.mark.parametrize("m", [1, 4, 64, 512])
@pytest.mark.parametrize("n,k", [(256, 256), (1024, 2048), (2048, 512)])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hybrid_w4a16_moe_asym(
    m: int, n: int, k: int, group_size: int, dtype: torch.dtype
):
    """Per-group zero points (AWQ) on both the HIP decode and Triton paths."""
    hybrid_out, torch_output = _run_hybrid_moe(
        m, n, k, 16, 4, group_size, dtype=dtype, asym=True
    )
    torch.testing.assert_close(hybrid_out, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
@pytest.mark.parametrize("m", [1, 64])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hybrid_w4a16_moe_awq_checkpoint(m: int, group_size: int, dtype: torch.dtype):
    """AutoAWQ-packed qweight/qzeros through the MoeWNA16 loader conversion and
    the hybrid repack match the dequantized reference (layout regressions)."""
    hybrid_out, torch_output = _run_hybrid_moe(
        m, 256, 512, 8, 2, group_size, dtype=dtype, make_weights=_make_awq_moe_weights
    )
    # Reference uses the same quantized values; a swapped nibble gives >2.6e-2.
    torch.testing.assert_close(hybrid_out, torch_output, atol=5e-3, rtol=0)


def test_hybrid_w4a16_moe_gate(monkeypatch: pytest.MonkeyPatch):
    """Enabled by default on gfx11/gfx12; an explicit --moe-backend or
    VLLM_MOE_HYBRID_W4A16=0 turns it off."""
    from vllm.platforms.rocm import on_gfx1x

    moe = make_dummy_moe_config()
    if current_platform.is_rocm() and on_gfx1x():
        assert hybrid_w4a16_moe_enabled(moe)
    assert not hybrid_w4a16_moe_enabled(dataclasses.replace(moe, moe_backend="triton"))
    monkeypatch.setenv("VLLM_MOE_HYBRID_W4A16", "0")
    assert not hybrid_w4a16_moe_enabled(moe)


def _effective_block_k_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the BLOCK_SIZE_K that invoke_fused_moe_kernel_hybrid_triton
    actually launches fused_moe_kernel_gptq_awq with."""
    from vllm.model_executor.layers.fused_moe import fused_moe as fm

    real = fm.fused_moe_kernel_gptq_awq
    seen: list[int] = []

    class _Spy:
        def __getitem__(self, grid):
            launch = real[grid]

            def run(*args, **kwargs):
                seen.append(kwargs["BLOCK_SIZE_K"])
                return launch(*args, **kwargs)

            return run

    monkeypatch.setattr(fm, "fused_moe_kernel_gptq_awq", _Spy())
    return seen


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
@pytest.mark.parametrize(
    "group_size,block_k,n,k,expected_bk",
    [
        # BLOCK_K spanning 2 / 4 groups: gemm1 (k=2048) and gemm2 (k=768).
        (32, 32, 1536, 2048, 32),
        (32, 64, 1536, 2048, 64),
        (32, 128, 1536, 2048, 128),
        (32, 64, 2048, 768, 64),
        (32, 128, 2048, 768, 128),
        (64, 64, 1536, 2048, 64),
        (64, 128, 1536, 2048, 128),
        (64, 128, 2048, 768, 128),
        # gs128 at BLOCK_K == group_size: no-regression check.
        (128, 128, 1536, 2048, 128),
        # K % BLOCK_K != 0 (k=192 = 6 groups of 32): falls back to BK=G.
        (32, 128, 256, 192, 32),
    ],
)
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hybrid_w4a16_triton_block_k_multigroup(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
    block_k: int,
    n: int,
    k: int,
    expected_bk: int,
    asym: bool,
    dtype: torch.dtype,
):
    """Shuffle-path Triton kernel with BLOCK_SIZE_K above group_size (several
    scales / zero points per K tile) matches the dequantized reference."""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_kernel_hybrid_triton,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.triton_utils import tl

    m, e, topk, align = 128, 128, 8, 64
    set_random_seed(1)
    device = torch.device("cuda")
    seen = _effective_block_k_spy(monkeypatch)

    w, w_scales, w_ref, w_zp = _make_hybrid_moe_weights(
        e, k, n, group_size, device, dtype, asym
    )
    hidden = torch.randn(m, k, device=device, dtype=dtype) / 10
    scores = torch.randn(m, e, device=device, dtype=dtype)
    _, topk_ids, _ = fused_topk(hidden, scores, topk, False)
    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, align, e, None, ignore_invalid_experts=True
    )
    out = torch.zeros(sorted_ids.size(0), n, device=device, dtype=dtype)
    cfg = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    invoke_fused_moe_kernel_hybrid_triton(
        A=hidden,
        B=w,
        C=out,
        B_scale=w_scales,
        topk_weights=None,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=npp,
        mul_routed_weight=False,
        top_k=topk,
        config=cfg,
        compute_type=tl.float16 if dtype == torch.float16 else tl.bfloat16,
        group_size=group_size,
        align_block_size_m=align,
        B_zp=w_zp,
    )
    assert seen == [expected_bk]

    # Row t * topk + j of the slot-space output is token t through expert
    # topk_ids[t, j].
    ref = torch.zeros(m * topk, n, device=device, dtype=torch.float32)
    for ex in topk_ids.unique().tolist():
        tok, j = (topk_ids == ex).nonzero(as_tuple=True)
        ref[tok * topk + j] = hidden[tok].float() @ w_ref[ex].float().T
    torch.testing.assert_close(out[: m * topk].float(), ref, atol=2e-2, rtol=2e-2)
