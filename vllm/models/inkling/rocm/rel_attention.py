# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm replacement for Inkling's FA4 CuTe-DSL relative attention.

Paged, varlen, GQA, causal + optional sliding-window attention with the
bounded learned relative-position bias
    bias = rel_logits[q, h, rel_dist]   for 0 <= rel_dist < rel_extent
where rel_dist = q_abs_pos - kv_pos and q_abs_pos = (seqlen_k - seqlen_q) + q.

Two implementations:
  * ``_rel_attention_torch``  -- a correctness-first PyTorch reference. It does a
    per-request Python loop and syncs seqlens to the host, so it is NOT CUDA
    graph capturable; used for prefill (fast, GEMM-based) and as the numeric
    oracle.
  * ``_rel_attention_triton`` -- a fused flash-style Triton kernel that touches
    no host memory and iterates a compile-time-constant number of KV blocks, so
    it can be captured into a CUDA graph. Used for decode.

``inkling_rocm_rel_attention`` dispatches decode (max_seqlen_q == 1) to Triton
and prefill to the reference. Override with ``INKLING_ROCM_ATTN=torch|triton``.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# PyTorch reference (oracle; used for prefill)
# ---------------------------------------------------------------------------
def _rel_attention_torch(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _, block_size, Hkv, D = key_cache.shape
    Hq = q.shape[1]
    rep = Hq // Hkv
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


# ---------------------------------------------------------------------------
# Triton flash kernel (CUDA-graph capturable; used for decode)
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _rel_attn_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr,
        bt_ptr, seqlens_ptr, req_ptr, qpos_ptr, rl_ptr,
        softmax_scale, rep, win_left, rel_extent,
        s_qt, s_qh, s_qd,
        s_kb, s_kp, s_kh, s_kd,
        s_vb, s_vp, s_vh, s_vd,
        s_ot, s_oh, s_od,
        s_btr, s_btb,
        s_rt, s_rh, s_re,
        CAUSAL: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_P: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        q_idx = tl.program_id(0)
        h = tl.program_id(1)
        r = tl.load(req_ptr + q_idx)
        qpos = tl.load(qpos_ptr + q_idx)
        Lk = tl.load(seqlens_ptr + r)
        kv_h = h // rep

        offs_d = tl.arange(0, BLOCK_D)
        offs_p = tl.arange(0, BLOCK_P)

        q = tl.load(q_ptr + q_idx * s_qt + h * s_qh + offs_d * s_qd).to(tl.float32)

        m_i = -1e38
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        rl_base = rl_ptr + q_idx * s_rt + h * s_rh

        for blk in range(0, MAX_BLOCKS):
            block_id = tl.load(bt_ptr + r * s_btr + blk * s_btb)
            kv_pos = blk * BLOCK_P + offs_p  # (P,)
            valid = kv_pos < Lk

            k_ptrs = (
                k_ptr + block_id * s_kb + offs_p[:, None] * s_kp
                + kv_h * s_kh + offs_d[None, :] * s_kd
            )
            k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)  # (P,D)
            scores = tl.sum(q[None, :] * k, axis=1) * softmax_scale  # (P,)

            rel_dist = qpos - kv_pos  # (P,)
            in_rel = (rel_dist >= 0) & (rel_dist < rel_extent)
            rel_idx = tl.maximum(tl.minimum(rel_dist, rel_extent - 1), 0)
            bias = tl.load(rl_base + rel_idx * s_re, mask=in_rel, other=0.0).to(tl.float32)
            scores += bias

            keep = valid
            if CAUSAL:
                keep = keep & (rel_dist >= 0)
            keep = keep & ((win_left < 0) | (rel_dist <= win_left))
            scores = tl.where(keep, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)  # (P,)
            l_i = l_i * alpha + tl.sum(p, axis=0)

            v_ptrs = (
                v_ptr + block_id * s_vb + offs_p[:, None] * s_vp
                + kv_h * s_vh + offs_d[None, :] * s_vd
            )
            v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)  # (P,D)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)  # (D,)
            m_i = m_new

        out = acc / tl.where(l_i > 0.0, l_i, 1.0)
        tl.store(
            out_ptr + q_idx * s_ot + h * s_oh + offs_d * s_od,
            out.to(out_ptr.dtype.element_ty),
        )


def _rel_attention_triton(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    T, Hq, D = q.shape
    _, block_size, Hkv, _ = key_cache.shape
    R = cache_seqlens.shape[0]
    max_blocks = block_table.shape[1]
    device = q.device

    # Per-query-token request id and absolute position, computed entirely on
    # device (no host sync -> capturable). For token t in request r:
    #   q_abs_pos = cache_seqlens[r] - cu_seqlens_q[r+1] + t
    t_ar = torch.arange(T, device=device)
    req = torch.searchsorted(cu_seqlens_q, t_ar, right=True) - 1
    req = req.clamp_(0, R - 1)
    q_abs_pos = (cache_seqlens[req] - cu_seqlens_q[req + 1] + t_ar).to(torch.int32)
    req = req.to(torch.int32)

    rel_logits = rel_logits.contiguous()
    win_left = window_size[0]

    grid = (T, Hq)
    _rel_attn_kernel[grid](
        q, key_cache, value_cache, out,
        block_table, cache_seqlens, req, q_abs_pos, rel_logits,
        softmax_scale, Hq // Hkv, win_left, rel_extent,
        q.stride(0), q.stride(1), q.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_table.stride(0), block_table.stride(1),
        rel_logits.stride(0), rel_logits.stride(1), rel_logits.stride(2),
        CAUSAL=causal,
        BLOCK_D=D,
        BLOCK_P=block_size,
        MAX_BLOCKS=max_blocks,
    )
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
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
    if out is None:
        out = torch.empty_like(q)

    mode = os.environ.get("INKLING_ROCM_ATTN", "")
    use_triton = _HAS_TRITON and (mode == "triton" or (mode != "torch" and max_seqlen_q == 1))

    if use_triton:
        return _rel_attention_triton(
            q, key_cache, value_cache,
            block_table=block_table, cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q, softmax_scale=softmax_scale,
            causal=causal, window_size=window_size, rel_extent=rel_extent,
            rel_logits=rel_logits, out=out,
        )
    return _rel_attention_torch(
        q, key_cache, value_cache,
        block_table=block_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, softmax_scale=softmax_scale,
        causal=causal, window_size=window_size, rel_extent=rel_extent,
        rel_logits=rel_logits, out=out,
    )
