# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .op import make_tensor_descriptor


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    desc_A = make_tensor_descriptor(
        A + (bos * H + i_h) * BT, [T, BT], [H * BT, 1], [BT, BT]
    )
    desc_v = make_tensor_descriptor(
        v + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, BV]
    )
    desc_u = make_tensor_descriptor(
        u + (bos * H + i_h) * V, [T, V], [H * V, 1], [BT, BV]
    )
    desc_k = make_tensor_descriptor(
        k + (bos * Hg + i_h // (H // Hg)) * K, [T, K], [Hg * K, 1], [BT, BK]
    )
    desc_w = make_tensor_descriptor(
        w + (bos * H + i_h) * K, [T, K], [H * K, 1], [BT, BK]
    )
    b_beta = tl.load(beta + (bos + o_t) * H + i_h, mask=m_t, other=0.0)
    b_A = desc_A.load([i_t * BT, 0])
    b_g = tl.exp(tl.load(g + (bos + o_t) * H + i_h, mask=m_t, other=0.0))

    for i_v in range(tl.cdiv(V, BV)):
        b_v = desc_v.load([i_t * BT, i_v * BV])
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        desc_u.store([i_t * BT, i_v * BV], b_u.to(desc_u.dtype))

    for i_k in range(tl.cdiv(K, BK)):
        b_k = desc_k.load([i_t * BT, i_k * BK])
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        desc_w.store([i_t * BT, i_k * BK], b_w.to(desc_w.dtype))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u
