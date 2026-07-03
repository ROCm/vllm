# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch for the rdna_moe_gemm W4A16 MoE prefill GEMM (gfx11 / RDNA3).

Enabled by default on gfx11 builds (where the kernel is compiled); set
``VLLM_MOE_HIP=0`` to force the Triton path or ``VLLM_MOE_HIP=1`` to force-enable.
When active and the routed problem matches one of the tuned Qwen3.6-35B-A3B
prefill MoE shapes, the producer/consumer WMMA kernel replaces the Triton
``fused_moe_kernel_gptq_awq`` for that GEMM.  Any other shape, arch, or an
explicit disable falls through to Triton unchanged, so the path is A/B-able.

The kernel itself is compiled into ``_rocm_C``
(``csrc/rocm/moe_gemm_w4a16_wmma.cu``; the WMMA body is gfx11-only, a stub
elsewhere) and exposed as ``torch.ops._rocm_C.moe_gemm_w4a16``.  This module just
makes the dispatch decision host-side (env + shape predicate) and wraps the call
as a registered vLLM custom op with a no-op fake, so the path is graph-safe under
torch.compile.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.platforms.rocm import on_gfx11
from vllm.utils.torch_utils import direct_register_custom_op

# The kernel takes K, N and the weight row stride at runtime, so it handles any
# shape meeting the tile constraints and any weight N-row padding. What stays
# fixed (compile-time instantiations) is the tile family keyed by top_k ->
# required N divisor (BN), the group size G, and block_m. Keep in sync with the
# host dispatch in csrc/rocm/moe_gemm_w4a16_wmma.cu.
_TILE_N_DIVISOR_BY_TOPK = {8: 256, 1: 512}  # top_k -> BN
_SUPPORTED_G = 128


def is_enabled() -> bool:
    # Default-on on gfx11 (the only arch with a real kernel body; the op is a
    # stub elsewhere). VLLM_MOE_HIP overrides: "1" forces on, "0" forces Triton.
    val = envs.VLLM_MOE_HIP
    if val is not None:
        return val == "1"
    return on_gfx11()


def _shape_supported(K: int, N: int, G: int, top_k: int) -> bool:
    """Divisibility constraints a (K, N, G, top_k) GEMM must meet for a tuned
    tile family (top_k picks BN; K % G == 0 also gives K % GK == 0 since GK|G)."""
    bn = _TILE_N_DIVISOR_BY_TOPK.get(top_k)
    return bn is not None and G == _SUPPORTED_G and K % G == 0 and N % bn == 0


def prefill_uses_rdna_moe_gemm(
    K_hidden: int,
    N_gemm1: int,
    act_dim: int,
    top_k: int,
    group_size: int,
    in_dtype: torch.dtype,
) -> bool:
    """True iff BOTH MoE prefill GEMMs will run on the rdna_moe_gemm kernel for this
    shape+dtype. Tensor-free (takes the dtype, not the tensor), so workspace_shapes
    and apply() can agree on block_m up front. gemm1 = (K_hidden, N_gemm1, top_k);
    gemm2 = down proj (act_dim, K_hidden, top_k=1). The WMMA kernel is bf16-only
    (wmma_f32_16x16x16_bf16); any other dtype falls through to the Triton path,
    which handles fp16."""
    if not is_enabled() or not on_gfx11():
        return False
    if in_dtype != torch.bfloat16:
        return False
    return _shape_supported(K_hidden, N_gemm1, group_size, top_k) and _shape_supported(
        act_dim, K_hidden, group_size, 1
    )


def _moe_gemm_w4a16_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    C: torch.Tensor,
    n_valid_tokens: int,
    top_k: int,
    block_size_m: int,
    num_blocks: int,
) -> None:
    """Run the rdna_moe_gemm WMMA W4A16 MoE GEMM, writing into C in place.

    Callers MUST gate on ``prefill_uses_rdna_moe_gemm`` first (shape+dtype, so the
    block_m choice and this dispatch agree). The kernel validates shape and dtype
    with TORCH_CHECK, so a mismatch raises rather than silently falling back.

    num_blocks: sync-free launch upper bound (Triton's EM cdiv block_m); padding
        blocks carry expert_id == -1 and early-return in the kernel.
    """
    torch.ops._rocm_C.moe_gemm_w4a16(
        A,
        B,
        B_scale,
        sorted_token_ids,
        expert_ids,
        C,
        int(n_valid_tokens),
        int(top_k),
        int(block_size_m),
        int(num_blocks),
    )


def _moe_gemm_w4a16_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    C: torch.Tensor,
    n_valid_tokens: int,
    top_k: int,
    block_size_m: int,
    num_blocks: int,
) -> None:
    # C is pre-allocated and mutated in place; nothing to allocate.
    return None


direct_register_custom_op(
    op_name="moe_gemm_w4a16",
    op_func=_moe_gemm_w4a16_impl,
    mutates_args=["C"],
    fake_impl=_moe_gemm_w4a16_fake,
)
