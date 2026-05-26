# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped-GEMM Triton kernel for shuffle-packed INT4 W4A16 MoE.

Consumes pre-permuted activations + per-block (expert_id, m_start,
m_count) routing, so the kernel reads contiguous rows of A and never
sees padding blocks.  Built to compare against the moe_align_block_size
+ sorted_token_ids path used by ``HybridW4A16MoEExperts.apply``.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (  # noqa: F401  (re-export for callers)
    direct_register_custom_op,
)


@triton.jit
def fused_moe_kernel_hybrid_w4a16_grouped(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    block_expert_ids_ptr,
    block_m_starts_ptr,
    block_m_counts_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Grouped-GEMM shuffle_w4a16 kernel.

    Parameters
    ----------
    a_ptr : pointer to permuted activations [num_routed_tokens, K]
    b_ptr : pointer to weights [E, N, K//8] int32 (ExLlama shuffle packed)
    c_ptr : pointer to output [num_routed_tokens, N]
    b_scale_ptr : pointer to scales [E, N, K//G] (fp16/bf16)
    block_expert_ids_ptr : [num_blocks] int32 — expert per block
    block_m_starts_ptr   : [num_blocks] int32 — start row in A for each block
    block_m_counts_ptr   : [num_blocks] int32 — valid rows in each block (<= BM)
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Linear pid mapping.  Consecutive blocks belong to different experts
    # in the grouped layout, so the original swizzled GROUP_SIZE_M-style
    # B-reuse heuristic does not apply here.  GROUP_SIZE_M is accepted but
    # ignored.
    _ = GROUP_SIZE_M
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    expert_id = tl.load(block_expert_ids_ptr + pid_m).to(tl.int64)
    m_start = tl.load(block_m_starts_ptr + pid_m).to(tl.int64)
    m_count = tl.load(block_m_counts_ptr + pid_m).to(tl.int32)

    offs_m_local = tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs_m_local < m_count
    offs_token = m_start + offs_m_local.to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # Shuffle-packed INT4 weight setup (mirrors fused_moe_kernel_gptq_awq).
    offs_k8 = tl.arange(0, BLOCK_SIZE_K // 8)
    b_packed_ptrs = (
        b_ptr
        + expert_id * stride_be
        + offs_bn[:, None] * stride_bn
        + offs_k8[None, :] * stride_bk
    )
    _exl_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    _exl_shifts_1d = tl.reshape(
        tl.broadcast_to(_exl_shifts_row[None, :], (BLOCK_SIZE_K // 8, 8)),
        (BLOCK_SIZE_K,),
    )
    exl_shifts = tl.broadcast_to(_exl_shifts_1d[None, :], (BLOCK_SIZE_N, BLOCK_SIZE_K))
    b_zp_num = 8

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )

        b_packed = tl.load(b_packed_ptrs)
        b_exp = tl.interleave(b_packed, b_packed)
        b_exp = tl.interleave(b_exp, b_exp)
        b_exp = tl.interleave(b_exp, b_exp)
        b_nk = (b_exp >> exl_shifts) & 0xF
        b = tl.trans(b_nk)

        if group_size >= BLOCK_SIZE_K:
            g_idx = (k * BLOCK_SIZE_K) // group_size
            b_scale_ptrs = (
                b_scale_ptr
                + expert_id * stride_bse
                + offs_bn * stride_bsn
                + g_idx * stride_bsk
            )
            b_scale = tl.load(b_scale_ptrs).to(tl.float32)
            b = ((b.to(tl.float32) - b_zp_num) * b_scale[None, :]).to(compute_type)
        else:
            b_scale_ptrs = (
                b_scale_ptr
                + expert_id * stride_bse
                + offs_bn[None, :] * stride_bsn
                + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
            )
            b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other).to(tl.float32)
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_packed_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = row_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def build_block_table(
    expert_first_token_offset: torch.Tensor,
    block_size_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate per-expert token offsets into a per-block routing table.

    Parameters
    ----------
    expert_first_token_offset : [E+1] int64, cumulative tokens per expert
        (output of moe_permute).
    block_size_m : kernel BLOCK_SIZE_M.

    Returns
    -------
    block_expert_ids : [num_blocks] int32 — expert per block
    block_m_starts   : [num_blocks] int32 — start row in permuted A
    block_m_counts   : [num_blocks] int32 — valid rows in this block (<= BM)
    """
    device = expert_first_token_offset.device
    counts64 = expert_first_token_offset[1:] - expert_first_token_offset[:-1]  # [E]
    blocks_per_expert = ((counts64 + block_size_m - 1) // block_size_m).to(torch.int32)

    cum_blocks = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            blocks_per_expert.cumsum(0).to(torch.int32),
        ]
    )  # [E+1]
    total_blocks = int(cum_blocks[-1].item())

    if total_blocks == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, empty

    block_indices = torch.arange(total_blocks, dtype=torch.int32, device=device)
    # which expert does each block belong to
    block_expert_ids = torch.searchsorted(cum_blocks[1:], block_indices, right=True).to(
        torch.int32
    )
    # index of the block within its expert (0..blocks_per_expert[e]-1)
    block_in_expert = block_indices - cum_blocks[block_expert_ids.long()]
    # start row in permuted activations
    block_m_starts = (
        expert_first_token_offset[block_expert_ids.long()].to(torch.int32)
        + block_in_expert * block_size_m
    )
    # how many valid rows in this block (last block of each expert is partial)
    block_m_counts = torch.minimum(
        torch.full((total_blocks,), block_size_m, dtype=torch.int32, device=device),
        counts64[block_expert_ids.long()].to(torch.int32)
        - block_in_expert * block_size_m,
    )
    return block_expert_ids, block_m_starts, block_m_counts


def apply_hybrid_w4a16_grouped(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int,
    activation,
    group_size: int,
    gemm1_config: dict,
    gemm2_config: dict,
) -> None:
    """Grouped prefill path: moe_permute -> grouped kernel x2 -> moe_unpermute.

    No moe_align_block_size, no padding, no virtual gather.  Activations
    are physically permuted into expert-contiguous order; per-block routing
    (expert_id, m_start, m_count) is built from expert_first_token_offset
    and consumed by the grouped Triton kernel.
    """
    from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
        moe_permute,
        moe_unpermute,
    )

    M = hidden_states.size(0)
    K = hidden_states.size(1)
    E = w1.size(0)
    N_w1 = w1.size(1)  # 2 * intermediate
    top_k = topk_ids.size(1)
    P = M * top_k
    if global_num_experts == -1:
        global_num_experts = E

    # Permute hidden_states into expert-contiguous order.
    permuted_hidden, _, e_offsets, inv_perm, _ = moe_permute(
        hidden_states=hidden_states,
        a1q_scale=None,
        topk_ids=topk_ids,
        n_expert=global_num_experts,
    )

    # GEMM 1: permuted [P, K] -> gemm1_out [P, N_w1]
    bt1 = build_block_table(e_offsets, gemm1_config["BLOCK_SIZE_M"])
    gemm1_out = torch.empty(
        P,
        N_w1,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    compute_type = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16
    invoke_fused_moe_kernel_hybrid_w4a16_grouped(
        A=permuted_hidden,
        B=w1,
        C=gemm1_out,
        B_scale=w1_scale,
        block_expert_ids=bt1[0],
        block_m_starts=bt1[1],
        block_m_counts=bt1[2],
        config=gemm1_config,
        compute_type=compute_type,
        group_size=group_size,
    )

    # Activation: in-place along N (halves N_w1 -> intermediate).
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEExpertsModular,
    )

    activation_out_dim = FusedMoEExpertsModular.adjust_N_for_activation(
        N_w1,
        activation,
    )
    act_out = torch.empty(
        P,
        activation_out_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    apply_moe_activation(activation, act_out, gemm1_out)

    # GEMM 2: act_out [P, intermediate] -> gemm2_out [P, K]
    bt2 = build_block_table(e_offsets, gemm2_config["BLOCK_SIZE_M"])
    gemm2_out = torch.empty(
        P,
        K,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    invoke_fused_moe_kernel_hybrid_w4a16_grouped(
        A=act_out,
        B=w2,
        C=gemm2_out,
        B_scale=w2_scale,
        block_expert_ids=bt2[0],
        block_m_starts=bt2[1],
        block_m_counts=bt2[2],
        config=gemm2_config,
        compute_type=compute_type,
        group_size=group_size,
    )

    # Unpermute + topk-weight fold + reduce.
    moe_unpermute(
        out=output,
        permuted_hidden_states=gemm2_out,
        topk_weights=topk_weights,
        inv_permuted_idx=inv_perm,
    )


def invoke_fused_moe_kernel_hybrid_w4a16_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    block_expert_ids: torch.Tensor,
    block_m_starts: torch.Tensor,
    block_m_counts: torch.Tensor,
    config: dict,
    compute_type,
    group_size: int,
) -> None:
    assert B.dtype == torch.int32
    assert B_scale is not None and B_scale.ndim == 3

    K = A.size(1)
    N = B.size(1)
    num_blocks = block_expert_ids.size(0)

    cfg = config.copy()
    BLOCK_SIZE_M = cfg.pop("BLOCK_SIZE_M")
    BLOCK_SIZE_N = cfg.pop("BLOCK_SIZE_N")
    BLOCK_SIZE_K = cfg.pop("BLOCK_SIZE_K")
    GROUP_SIZE_M = cfg.pop("GROUP_SIZE_M")
    num_warps = cfg.pop("num_warps")
    num_stages = cfg.pop("num_stages")
    assert not cfg, f"unexpected config keys: {list(cfg)}"
    assert BLOCK_SIZE_K % 8 == 0
    assert group_size >= BLOCK_SIZE_K or BLOCK_SIZE_K % group_size == 0

    grid = (num_blocks * triton.cdiv(N, BLOCK_SIZE_N),)
    fused_moe_kernel_hybrid_w4a16_grouped[grid](
        A,
        B,
        C,
        B_scale,
        block_expert_ids,
        block_m_starts,
        block_m_counts,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        block_k_diviable=K % BLOCK_SIZE_K == 0,
        group_size=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        compute_type=compute_type,
        num_warps=num_warps,
        num_stages=num_stages,
    )
