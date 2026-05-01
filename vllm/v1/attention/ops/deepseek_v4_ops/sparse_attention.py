# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fully Triton sparse-attention kernels for the DeepSeek-V4 ROCm path.

Both prefill and decode kernels compute the sparse attention end-to-end inside
Triton: sparse score evaluation, stable softmax normalization, and output
accumulation. Decode dequantizes referenced FP8 KV rows on the fly and never
materializes a gathered KV activation on the host.
"""

import torch

from vllm.triton_utils import tl, triton

_DSV4_NOPE_DIM = 448
_DSV4_ROPE_DIM = 64
_NUM_WARPS = 4


@triton.jit
def _load_prefill_row(
    kv_ptr,
    row_idx,
    kv_stride0,
    nope_offsets,
    rope_offsets,
    nope_mask,
    NOPE_DIM: tl.constexpr,
):
    k_nope = tl.load(
        kv_ptr + row_idx * kv_stride0 + nope_offsets,
        mask=nope_mask,
        other=0.0,
    ).to(tl.float32)
    k_rope = tl.load(
        kv_ptr + row_idx * kv_stride0 + NOPE_DIM + rope_offsets,
    ).to(tl.float32)
    k_nope = tl.where(k_nope == k_nope, k_nope, 0.0)
    k_rope = tl.where(k_rope == k_rope, k_rope, 0.0)
    return k_nope, k_rope


@triton.jit
def _load_decode_row(
    cache_ptr,
    row_idx,
    cache_stride0,
    block_size,
    nope_offsets,
    rope_offsets,
    nope_mask,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    block_idx = row_idx // block_size
    pos_in_block = row_idx % block_size

    cache_block_ptr = cache_ptr + block_idx.to(tl.int64) * cache_stride0
    token_data_ptr = cache_block_ptr + pos_in_block * 576
    token_scale_ptr = cache_block_ptr + block_size * 576 + pos_in_block * 8

    x_uint8 = tl.load(token_data_ptr + nope_offsets, mask=nope_mask, other=0)
    x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
    encoded_scales = tl.load(
        token_scale_ptr + nope_offsets // 64,
        mask=nope_mask,
        other=127,
    )
    scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
    k_nope = (x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)).to(tl.float32)
    k_nope = tl.where(nope_mask, k_nope, 0.0)
    k_nope = tl.where(k_nope == k_nope, k_nope, 0.0)

    rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    k_rope = tl.load(rope_ptr + rope_offsets).to(tl.float32)
    k_rope = tl.where(k_rope == k_rope, k_rope, 0.0)
    return k_nope, k_rope


@triton.jit
def _sparse_attn_prefill_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    kv_stride0,
    indices_stride0,
    out_stride0,
    out_stride1,
    num_kv,
    topk,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_idx * q_stride1
    q_nope = tl.load(q_row_ptr + nope_offsets, mask=nope_mask, other=0.0).to(tl.float32)
    q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)

    neg_inf = float('-inf')
    max_score = neg_inf

    for topk_idx in range(topk):
        kv_idx = tl.load(indices_ptr + query_idx * indices_stride0 + topk_idx)
        valid = (kv_idx >= 0) & (kv_idx < num_kv)
        if valid:
            k_nope, k_rope = _load_prefill_row(
                kv_ptr,
                kv_idx,
                kv_stride0,
                nope_offsets,
                rope_offsets,
                nope_mask,
                NOPE_DIM,
            )
            score = tl.zeros((), dtype=tl.float32)
            for chunk_start in tl.static_range(0, NOPE_DIM, 64):
                chunk_mask = (nope_offsets >= chunk_start) & (
                    nope_offsets < chunk_start + 64
                )
                score += tl.sum(
                    tl.where(chunk_mask, q_nope * k_nope, 0.0),
                    axis=0,
                )
            q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)
            score += tl.sum(q_rope * k_rope, axis=0)
            score *= scale
            max_score = tl.maximum(max_score, score)

    has_valid = max_score != neg_inf
    max_score_safe = tl.where(has_valid, max_score, 0.0)
    sum_exp = tl.zeros((), dtype=tl.float32)
    acc_nope = tl.zeros((NOPE_BLOCK,), dtype=tl.float32)
    acc_rope = tl.zeros((ROPE_DIM,), dtype=tl.float32)

    for topk_idx in range(topk):
        kv_idx = tl.load(indices_ptr + query_idx * indices_stride0 + topk_idx)
        valid = (kv_idx >= 0) & (kv_idx < num_kv)
        if valid:
            k_nope, k_rope = _load_prefill_row(
                kv_ptr,
                kv_idx,
                kv_stride0,
                nope_offsets,
                rope_offsets,
                nope_mask,
                NOPE_DIM,
            )
            score = tl.zeros((), dtype=tl.float32)
            for chunk_start in tl.static_range(0, NOPE_DIM, 64):
                chunk_mask = (nope_offsets >= chunk_start) & (
                    nope_offsets < chunk_start + 64
                )
                score += tl.sum(
                    tl.where(chunk_mask, q_nope * k_nope, 0.0),
                    axis=0,
                )
            q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)
            score += tl.sum(q_rope * k_rope, axis=0)
            score *= scale
            weight = tl.where(has_valid, tl.exp(score - max_score_safe), 0.0)
            sum_exp += weight
            acc_nope += weight * k_nope
            acc_rope += weight * k_rope

    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
        sink_term = tl.where(has_valid, tl.exp(sink - max_score_safe), 0.0)
        denom = sum_exp + sink_term
    else:
        denom = sum_exp

    inv_denom = tl.where(has_valid, 1.0 / denom, 0.0)
    out_row_ptr = out_ptr + query_idx * out_stride0 + head_idx * out_stride1
    tl.store(
        out_row_ptr + nope_offsets,
        (acc_nope * inv_denom).to(tl.bfloat16),
        mask=nope_mask,
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets,
        (acc_rope * inv_denom).to(tl.bfloat16),
    )


@triton.jit
def _sparse_attn_decode_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    main_indices_stride0,
    extra_indices_stride0,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    main_topk,
    extra_topk,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_idx * q_stride1
    q_nope = tl.load(q_row_ptr + nope_offsets, mask=nope_mask, other=0.0).to(tl.float32)
    q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)

    neg_inf = float('-inf')
    max_score = neg_inf

    for topk_idx in range(main_topk):
        kv_idx = tl.load(main_indices_ptr + query_idx * main_indices_stride0 + topk_idx)
        valid = (kv_idx >= 0) & (kv_idx < main_num_rows)
        if valid:
            k_nope, k_rope = _load_decode_row(
                main_cache_ptr,
                kv_idx,
                main_cache_stride0,
                main_block_size,
                nope_offsets,
                rope_offsets,
                nope_mask,
                NOPE_DIM,
                ROPE_DIM,
            )
            score = tl.zeros((), dtype=tl.float32)
            for chunk_start in tl.static_range(0, NOPE_DIM, 64):
                chunk_mask = (nope_offsets >= chunk_start) & (
                    nope_offsets < chunk_start + 64
                )
                score += tl.sum(
                    tl.where(chunk_mask, q_nope * k_nope, 0.0),
                    axis=0,
                )
            q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)
            score += tl.sum(q_rope * k_rope, axis=0)
            score *= scale
            max_score = tl.maximum(max_score, score)

    if HAS_EXTRA:
        for topk_idx in range(extra_topk):
            kv_idx = tl.load(extra_indices_ptr + query_idx * extra_indices_stride0 + topk_idx)
            valid = (kv_idx >= 0) & (kv_idx < extra_num_rows)
            if valid:
                k_nope, k_rope = _load_decode_row(
                    extra_cache_ptr,
                    kv_idx,
                    extra_cache_stride0,
                    extra_block_size,
                    nope_offsets,
                    rope_offsets,
                    nope_mask,
                    NOPE_DIM,
                    ROPE_DIM,
                )
                score = tl.zeros((), dtype=tl.float32)
                for chunk_start in tl.static_range(0, NOPE_DIM, 64):
                    chunk_mask = (nope_offsets >= chunk_start) & (
                        nope_offsets < chunk_start + 64
                    )
                    score += tl.sum(
                        tl.where(chunk_mask, q_nope * k_nope, 0.0),
                        axis=0,
                    )
                q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)
                score += tl.sum(q_rope * k_rope, axis=0)
                score *= scale
                max_score = tl.maximum(max_score, score)

    has_valid = max_score != neg_inf
    max_score_safe = tl.where(has_valid, max_score, 0.0)
    sum_exp = tl.zeros((), dtype=tl.float32)
    acc_nope = tl.zeros((NOPE_BLOCK,), dtype=tl.float32)
    acc_rope = tl.zeros((ROPE_DIM,), dtype=tl.float32)

    for topk_idx in range(main_topk):
        kv_idx = tl.load(main_indices_ptr + query_idx * main_indices_stride0 + topk_idx)
        valid = (kv_idx >= 0) & (kv_idx < main_num_rows)
        if valid:
            k_nope, k_rope = _load_decode_row(
                main_cache_ptr,
                kv_idx,
                main_cache_stride0,
                main_block_size,
                nope_offsets,
                rope_offsets,
                nope_mask,
                NOPE_DIM,
                ROPE_DIM,
            )
            score = tl.zeros((), dtype=tl.float32)
            for chunk_start in tl.static_range(0, NOPE_DIM, 64):
                chunk_mask = (nope_offsets >= chunk_start) & (
                    nope_offsets < chunk_start + 64
                )
                score += tl.sum(
                    tl.where(chunk_mask, q_nope * k_nope, 0.0),
                    axis=0,
                )
            q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)
            score += tl.sum(q_rope * k_rope, axis=0)
            score *= scale
            weight = tl.where(has_valid, tl.exp(score - max_score_safe), 0.0)
            sum_exp += weight
            acc_nope += weight * k_nope
            acc_rope += weight * k_rope

    if HAS_EXTRA:
        for topk_idx in range(extra_topk):
            kv_idx = tl.load(extra_indices_ptr + query_idx * extra_indices_stride0 + topk_idx)
            valid = (kv_idx >= 0) & (kv_idx < extra_num_rows)
            if valid:
                k_nope, k_rope = _load_decode_row(
                    extra_cache_ptr,
                    kv_idx,
                    extra_cache_stride0,
                    extra_block_size,
                    nope_offsets,
                    rope_offsets,
                    nope_mask,
                    NOPE_DIM,
                    ROPE_DIM,
                )
                score = tl.zeros((), dtype=tl.float32)
                for chunk_start in tl.static_range(0, NOPE_DIM, 64):
                    chunk_mask = (nope_offsets >= chunk_start) & (
                        nope_offsets < chunk_start + 64
                    )
                    score += tl.sum(
                        tl.where(chunk_mask, q_nope * k_nope, 0.0),
                        axis=0,
                    )
                q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)
                score += tl.sum(q_rope * k_rope, axis=0)
                score *= scale
                weight = tl.where(has_valid, tl.exp(score - max_score_safe), 0.0)
                sum_exp += weight
                acc_nope += weight * k_nope
                acc_rope += weight * k_rope

    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
        sink_term = tl.where(has_valid, tl.exp(sink - max_score_safe), 0.0)
        denom = sum_exp + sink_term
    else:
        denom = sum_exp

    inv_denom = tl.where(has_valid, 1.0 / denom, 0.0)
    out_row_ptr = out_ptr + query_idx * out_stride0 + head_idx * out_stride1
    tl.store(
        out_row_ptr + nope_offsets,
        (acc_nope * inv_denom).to(tl.bfloat16),
        mask=nope_mask,
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets,
        (acc_rope * inv_denom).to(tl.bfloat16),
    )


def sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_dim: int = _DSV4_NOPE_DIM,
    rope_dim: int = _DSV4_ROPE_DIM,
) -> torch.Tensor:
    assert q.ndim == 3, f'expected q=[sq,h,d], got {q.shape}'
    assert kv.ndim == 2, f'expected kv=[skv,d], got {kv.shape}'
    assert indices.ndim == 2, f'expected indices=[sq,topk], got {indices.shape}'
    assert q.is_cuda and kv.is_cuda and indices.is_cuda

    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.contiguous()
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert head_dim == nope_dim + rope_dim, (
        f'expected head_dim={nope_dim + rope_dim}, got {head_dim}'
    )

    out = torch.empty_like(q, dtype=torch.bfloat16)
    _sparse_attn_prefill_kernel[(num_queries, num_heads)](
        q,
        kv,
        indices,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        kv.shape[0],
        indices.shape[-1],
        scale,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        NOPE_DIM=nope_dim,
        NOPE_BLOCK=triton.next_power_of_2(nope_dim),
        ROPE_DIM=rope_dim,
        num_warps=_NUM_WARPS,
    )
    return out


def sparse_attn_decode_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    nope_dim: int = _DSV4_NOPE_DIM,
    rope_dim: int = _DSV4_ROPE_DIM,
) -> torch.Tensor:
    assert q.ndim == 3, f'expected q=[b,h,d], got {q.shape}'
    assert main_cache.ndim == 3, f'expected main_cache=[blocks,block,bytes], got {main_cache.shape}'
    assert main_indices.ndim == 2, f'expected main_indices=[b,topk], got {main_indices.shape}'
    assert q.is_cuda and main_cache.is_cuda and main_indices.is_cuda

    q = q.contiguous()
    main_indices = main_indices.contiguous()
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    has_extra = extra_cache is not None and extra_indices is not None
    if has_extra:
        assert extra_cache is not None and extra_indices is not None
        extra_cache = extra_cache.contiguous()
        extra_indices = extra_indices.contiguous()
    else:
        extra_cache = main_cache
        extra_indices = main_indices[:, :1]

    num_queries, num_heads, head_dim = q.shape
    assert head_dim == nope_dim + rope_dim, (
        f'expected head_dim={nope_dim + rope_dim}, got {head_dim}'
    )

    out = torch.empty_like(q, dtype=torch.bfloat16)
    _sparse_attn_decode_kernel[(num_queries, num_heads)](
        q,
        main_cache,
        main_indices,
        extra_cache,
        extra_indices,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        main_indices.stride(0),
        extra_indices.stride(0),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        main_indices.shape[-1],
        extra_indices.shape[-1] if has_extra else 0,
        scale,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_dim,
        NOPE_BLOCK=triton.next_power_of_2(nope_dim),
        ROPE_DIM=rope_dim,
        num_warps=_NUM_WARPS,
    )
    return out
