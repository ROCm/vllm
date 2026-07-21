# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm replacement for Inkling's FA4 CuTe-DSL relative attention.

Paged, varlen, GQA, causal + optional sliding-window attention with the
bounded learned relative-position bias
    bias = rel_logits[q, h, rel_dist]   for 0 <= rel_dist < rel_extent
where rel_dist = q_abs_pos - kv_pos and q_abs_pos = (seqlen_k - seqlen_q) + q.
Reference PyTorch implementation; a fused Triton kernel can replace it later.
"""

from __future__ import annotations

import torch


def inkling_rocm_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    # q (T, Hq, D); key/value cache (num_blocks, block_size, Hkv, D);
    # rel_logits (T, Hq, rel_extent). window_size is (-1, -1) for global layers,
    # (local_extent - 1, 0) for local (sliding-window) layers.
    T, Hq, D = q.shape
    _, block_size, Hkv, _ = key_cache.shape
    rep = Hq // Hkv

    if out is None:
        out = torch.empty_like(q)

    device = q.device
    win_left, _ = window_size
    q_starts = cu_seqlens_q.to("cpu", torch.int64).tolist()
    seqlens_k = cache_seqlens.to("cpu", torch.int64).tolist()
    block_table = block_table.to(torch.int64)

    for r in range(len(seqlens_k)):
        qs, qe = q_starts[r], q_starts[r + 1]
        Lq = qe - qs
        if Lq <= 0:
            continue
        Lk = seqlens_k[r]
        if Lk <= 0:
            out[qs:qe].zero_()
            continue

        n_blk = (Lk + block_size - 1) // block_size
        blocks = block_table[r, :n_blk]
        k = key_cache[blocks].reshape(n_blk * block_size, Hkv, D)[:Lk]
        v = value_cache[blocks].reshape(n_blk * block_size, Hkv, D)[:Lk]
        k = k.repeat_interleave(rep, dim=1)  # GQA: (Lk, Hq, D)
        v = v.repeat_interleave(rep, dim=1)

        qh = q[qs:qe].permute(1, 0, 2).float()  # (Hq, Lq, D)
        kh = k.permute(1, 0, 2).float()
        vh = v.permute(1, 0, 2).float()
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * softmax_scale

        q_pos = torch.arange(Lk - Lq, Lk, device=device)
        rel_dist = q_pos[:, None] - torch.arange(Lk, device=device)[None, :]
        mask = rel_dist < 0
        if win_left >= 0:
            mask = mask | (rel_dist > win_left)
        scores = scores.masked_fill(mask[None], float("-inf"))

        in_rel = (rel_dist >= 0) & (rel_dist < rel_extent)
        rel_idx = rel_dist.clamp_(0, rel_extent - 1)
        gather_idx = rel_idx[:, None, :].expand(Lq, Hq, Lk)
        bias = torch.gather(rel_logits[qs:qe].float(), 2, gather_idx) * in_rel[:, None, :]
        scores = scores + bias.permute(1, 0, 2)

        probs = torch.softmax(scores, dim=-1)
        out[qs:qe] = torch.matmul(probs, vh).permute(1, 0, 2).to(out.dtype)

    return out
