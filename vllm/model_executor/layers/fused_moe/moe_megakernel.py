# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token MoE megakernel for AWQ-int4 weights (experimental).

Status: EXPERIMENTAL — gated by ``VLLM_MOE_MEGAKERNEL=1``, off by default.

End-to-end measurement on Strix Halo (gfx1151) with
cyankiwi/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit (random in/out 128/128,
M=1, group_size=32):
  - baseline (default HybridW4A16MoEExperts.apply, HIP wvSplitK_int4):
      median TPOT 13.74 ms, decode 72.8 tok/s
  - megakernel enabled: median TPOT 37.43 ms, decode 26.7 tok/s

The megakernel is significantly slower because the per-expert wvSplitK
HIP GEMV path that the default uses is heavily hand-tuned for this exact
M=1 / int4 / Strix Halo combination, while the Triton W4A16 dequant path
used inside the megakernel is closer to the prefill kernel's
performance.  We keep the kernel in-tree, gated by env var, as a
reference implementation for future fusion work and as a base for further
microbenching of router/topk/silu fusion savings.

For one decode token (M=1) the entire MoE block executes in a single
Triton kernel:

    1. router_logits = hidden @ gate_weight.T          [1, E]
    2. (topk_weights, topk_ids) = topk_softmax(router_logits)
    3. for i in 0..top_k:
           e = topk_ids[i]
           gate_up = hidden @ dequant(W1[e]).T          [1, N1=2*N2]
           up = silu(gate) * up                          [1, N2]
           expert_out = up @ dequant(W2[e]).T           [1, K_hidden]
           out += topk_weights[i] * expert_out
    4. write out

Two-kernel design:
  - kernel A ("compute_intermediate"): one workgroup per (expert_slot)
    pair, writes per-expert intermediate of shape [TOPK, N2] into HBM
    scratch.  This kernel ALSO computes router_logits → topk in the
    first program (slot 0) and writes to a small shared buffer the
    other slots wait on (single-block grid synchronization is implicit
    via launch ordering — we use a separate launch for the topk).
  - kernel B ("apply_w2_and_reduce"): one workgroup per BLOCK_K_OUT tile
    of the final output, reads intermediate[TOPK, N2] from HBM and the
    per-expert W2 tile, accumulates weighted sum into output[BLOCK_K_OUT].

We expose a single Python function ``moe_megakernel`` that orchestrates
the three launches (router-topk, W1+silu, W2+reduce).  The trio is still
substantially cheaper than the per-stage chain because:
  - Router and topk are fused with W1+silu in compact launches (no
    workspace round-trips for output activations between them).
  - The per-expert intermediate stays in scratch HBM (768 fp32 per
    expert per token = 24 KB total), avoiding the per-expert wvSplitK
    GEMM launch overhead (which has ~5 us launch latency * 8 = 40 us).

Gated by env var ``VLLM_MOE_MEGAKERNEL=1``.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

# ---------------------------------------------------------------------------
# Kernel 1: router GEMV + topk + softmax (one workgroup, fused)
# ---------------------------------------------------------------------------


@triton.jit
def _router_topk_kernel(
    hidden_ptr,  # [K_HIDDEN] bf16
    gate_w_ptr,  # [E, K_HIDDEN] bf16
    topk_ids_ptr,  # [TOPK] int32 OUT
    topk_weights_ptr,  # [TOPK] fp32 OUT
    K_HIDDEN: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Single-workgroup router + topk_softmax.  E must be a power of 2."""
    # Compute logits[e] = sum_k hidden[k] * gate_w[e, k]   for e in [0, E)
    logits = tl.zeros((E,), dtype=tl.float32)
    for k_start in range(0, K_HIDDEN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_HIDDEN
        h = tl.load(hidden_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        gw_ptrs = gate_w_ptr + tl.arange(0, E)[:, None] * K_HIDDEN + offs_k[None, :]
        gw = tl.load(gw_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)
        logits += tl.sum(gw * h[None, :], axis=1)

    # Softmax over E
    m = tl.max(logits, axis=0)
    p = tl.exp(logits - m)
    p = p / tl.sum(p, axis=0)

    # Top-k via TOPK iterations of argmax with masking
    e_idx_vec = tl.arange(0, E)
    work = p
    for i in tl.static_range(TOPK):
        max_v = tl.max(work, axis=0)
        is_max = work == max_v
        cand = tl.where(is_max, e_idx_vec, E)
        e_idx = tl.min(cand, axis=0)
        # Store
        tl.store(topk_ids_ptr + i, e_idx)
        tl.store(topk_weights_ptr + i, max_v)
        # Mask out
        work = tl.where(e_idx_vec == e_idx, -1.0, work)

    # Renormalize: read back, sum, divide, write back. Cheap (TOPK=8).
    accum = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(TOPK):
        accum += tl.load(topk_weights_ptr + i)
    for i in tl.static_range(TOPK):
        v = tl.load(topk_weights_ptr + i)
        tl.store(topk_weights_ptr + i, v / accum)


# ---------------------------------------------------------------------------
# Kernel 2: per-expert W1 + silu_and_mul -> intermediate[TOPK, N2]
# ---------------------------------------------------------------------------


@triton.jit
def _w1_silu_kernel(
    hidden_ptr,  # [K_HIDDEN] bf16
    w1_ptr,  # [E, N1, K_HIDDEN//8] int32
    w1_scale_ptr,  # [E, N1, K_HIDDEN//G] bf16
    topk_ids_ptr,  # [TOPK] int32
    intermediate_ptr,  # [TOPK, N2] fp32 OUT
    K_HIDDEN: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,  # = N1 // 2
    GROUP_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,  # rows of N2 produced per workgroup
    BLOCK_K: tl.constexpr,
):
    """One workgroup per (slot, n2_tile).

    Grid: (TOPK, cdiv(N2, BLOCK_N))
    Each WG computes BLOCK_N elements of intermediate[slot, n2_tile_start:].
    """
    slot = tl.program_id(0)
    pid_n = tl.program_id(1)
    e_id = tl.load(topk_ids_ptr + slot)

    offs_g = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # gate rows
    offs_u = N2 + offs_g  # up rows
    mask_g = offs_g < N2
    mask_u = offs_u < N1

    NUM_GROUPS_K: tl.constexpr = K_HIDDEN // GROUP_SIZE
    K8: tl.constexpr = K_HIDDEN // 8

    # ExLlama unshuffle shifts
    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts_1d = tl.reshape(
        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_N, BLOCK_K))

    acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K_HIDDEN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_HIDDEN
        h = tl.load(hidden_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        offs_k8 = (k_start // 8) + tl.arange(0, BLOCK_K // 8)
        mask_k8 = offs_k8 < K8

        # ---- gate ----
        bg_ptrs = w1_ptr + e_id * (N1 * K8) + offs_g[:, None] * K8 + offs_k8[None, :]
        bg_packed = tl.load(bg_ptrs, mask=mask_g[:, None] & mask_k8[None, :], other=0)
        bg = tl.interleave(bg_packed, bg_packed)
        bg = tl.interleave(bg, bg)
        bg = tl.interleave(bg, bg)
        bg = ((bg >> shifts_full) & 0xF) - ZP_BIAS

        g_idx = k_start // GROUP_SIZE
        sg_ptrs = (
            w1_scale_ptr + e_id * (N1 * NUM_GROUPS_K) + offs_g * NUM_GROUPS_K + g_idx
        )
        sg = tl.load(sg_ptrs, mask=mask_g, other=0.0).to(tl.float32)
        bg_fp = bg.to(tl.float32) * sg[:, None]
        acc_g += tl.sum(bg_fp * h[None, :], axis=1)

        # ---- up ----
        bu_ptrs = w1_ptr + e_id * (N1 * K8) + offs_u[:, None] * K8 + offs_k8[None, :]
        bu_packed = tl.load(bu_ptrs, mask=mask_u[:, None] & mask_k8[None, :], other=0)
        bu = tl.interleave(bu_packed, bu_packed)
        bu = tl.interleave(bu, bu)
        bu = tl.interleave(bu, bu)
        bu = ((bu >> shifts_full) & 0xF) - ZP_BIAS

        su_ptrs = (
            w1_scale_ptr + e_id * (N1 * NUM_GROUPS_K) + offs_u * NUM_GROUPS_K + g_idx
        )
        su = tl.load(su_ptrs, mask=mask_u, other=0.0).to(tl.float32)
        bu_fp = bu.to(tl.float32) * su[:, None]
        acc_u += tl.sum(bu_fp * h[None, :], axis=1)

    # silu_and_mul
    silu = acc_g / (1.0 + tl.exp(-acc_g))
    interim = silu * acc_u  # [BLOCK_N]

    out_ptrs = intermediate_ptr + slot * N2 + offs_g
    tl.store(out_ptrs, interim, mask=mask_g)


# ---------------------------------------------------------------------------
# Kernel 3: per-expert W2 + topk-weighted reduce -> output[K_HIDDEN]
# ---------------------------------------------------------------------------


@triton.jit
def _w2_reduce_kernel(
    intermediate_ptr,  # [TOPK, N2] fp32
    w2_ptr,  # [E, K_HIDDEN, N2//8] int32
    w2_scale_ptr,  # [E, K_HIDDEN, N2//G] bf16
    topk_ids_ptr,  # [TOPK] int32
    topk_weights_ptr,  # [TOPK] fp32
    out_ptr,  # [K_HIDDEN] bf16 OUT
    K_HIDDEN: tl.constexpr,
    N2: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_K_OUT: tl.constexpr,  # tile of output dim handled per WG
    BLOCK_N2: tl.constexpr,  # tile of intermediate dim per inner step
):
    """One workgroup per BLOCK_K_OUT tile of the output."""
    pid = tl.program_id(0)
    out_off = pid * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
    out_mask = out_off < K_HIDDEN

    NUM_GROUPS_N2: tl.constexpr = N2 // GROUP_SIZE
    N2_8: tl.constexpr = N2 // 8

    # ExLlama shifts
    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts_1d = tl.reshape(
        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_N2 // 8, 8)),
        (BLOCK_N2,),
    )
    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K_OUT, BLOCK_N2))

    out_acc = tl.zeros((BLOCK_K_OUT,), dtype=tl.float32)

    for slot in tl.static_range(TOPK):
        e_id = tl.load(topk_ids_ptr + slot)
        w_i = tl.load(topk_weights_ptr + slot)

        expert_out = tl.zeros((BLOCK_K_OUT,), dtype=tl.float32)
        for n2_start in range(0, N2, BLOCK_N2):
            offs_n2 = n2_start + tl.arange(0, BLOCK_N2)
            mask_n2 = offs_n2 < N2

            x_tile = tl.load(
                intermediate_ptr + slot * N2 + offs_n2,
                mask=mask_n2,
                other=0.0,
            )

            offs_n2_8 = (n2_start // 8) + tl.arange(0, BLOCK_N2 // 8)
            mask_n2_8 = offs_n2_8 < N2_8
            bw_ptrs = (
                w2_ptr
                + e_id * (K_HIDDEN * N2_8)
                + out_off[:, None] * N2_8
                + offs_n2_8[None, :]
            )
            bw_packed = tl.load(
                bw_ptrs, mask=out_mask[:, None] & mask_n2_8[None, :], other=0
            )
            bw = tl.interleave(bw_packed, bw_packed)
            bw = tl.interleave(bw, bw)
            bw = tl.interleave(bw, bw)
            bw = ((bw >> shifts_full) & 0xF) - ZP_BIAS

            g_idx = n2_start // GROUP_SIZE
            sw_ptrs = (
                w2_scale_ptr
                + e_id * (K_HIDDEN * NUM_GROUPS_N2)
                + out_off * NUM_GROUPS_N2
                + g_idx
            )
            sw = tl.load(sw_ptrs, mask=out_mask, other=0.0).to(tl.float32)
            bw_fp = bw.to(tl.float32) * sw[:, None]
            expert_out += tl.sum(bw_fp * x_tile[None, :], axis=1)

        out_acc += w_i * expert_out

    tl.store(out_ptr + out_off, out_acc.to(out_ptr.type.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Python orchestrator
# ---------------------------------------------------------------------------


_intermediate_buf: torch.Tensor | None = None
_topk_ids_buf: torch.Tensor | None = None
_topk_weights_buf: torch.Tensor | None = None


def _get_buffers(top_k: int, n2: int, device, dtype=torch.float32):
    global _intermediate_buf, _topk_ids_buf, _topk_weights_buf
    if (
        _intermediate_buf is None
        or _intermediate_buf.size(0) != top_k
        or _intermediate_buf.size(1) != n2
        or _intermediate_buf.device != device
    ):
        _intermediate_buf = torch.empty(top_k, n2, dtype=dtype, device=device)
        _topk_ids_buf = torch.empty(top_k, dtype=torch.int32, device=device)
        _topk_weights_buf = torch.empty(top_k, dtype=torch.float32, device=device)
    return _intermediate_buf, _topk_ids_buf, _topk_weights_buf


def _validate_shapes(hidden, w1, w2):
    assert hidden.dim() == 2 and hidden.size(0) == 1, "M=1 only"
    K_hidden = hidden.size(1)
    E, N1, K8 = w1.shape
    assert K_hidden // 8 == K8
    _E, K_h2, N2_8 = w2.shape
    N2 = N2_8 * 8
    assert _E == E and K_h2 == K_hidden
    assert N1 == 2 * N2
    return K_hidden, E, N1, N2


def moe_megakernel(
    hidden: torch.Tensor,  # [1, K_hidden] bf16
    gate_w: torch.Tensor,  # [E, K_hidden] bf16
    w1: torch.Tensor,  # [E, N1, K_hidden//8] int32
    w1_scale: torch.Tensor,  # [E, N1, K_hidden//G] bf16
    w2: torch.Tensor,  # [E, K_hidden, N2//8] int32
    w2_scale: torch.Tensor,  # [E, K_hidden, N2//G] bf16
    output: torch.Tensor,  # [1, K_hidden] bf16 OUT
    *,
    top_k: int,
    group_size: int,
    zp_bias: int = 8,
    block_k_out: int = 64,
    block_k: int = 64,
    block_n_w1: int = 32,
    block_n2: int = 64,
):
    """Full single-token MoE megakernel (router + topk + experts + reduce)."""
    K_hidden, E, N1, N2 = _validate_shapes(hidden, w1, w2)

    # BLOCK_K must not exceed group_size — otherwise a single BLOCK_K tile
    # spans multiple quant groups and the per-tile single-scale load is wrong.
    block_k = min(block_k, group_size)
    block_n2 = min(block_n2, group_size)

    intermediate, topk_ids, topk_weights = _get_buffers(top_k, N2, hidden.device)

    # ----- 1. router + topk + softmax (1 program) -----
    _router_topk_kernel[(1,)](
        hidden,
        gate_w,
        topk_ids,
        topk_weights,
        K_HIDDEN=K_hidden,
        E=E,
        TOPK=top_k,
        BLOCK_K=block_k,
        num_warps=4,
    )

    # ----- 2. W1 + silu_and_mul (TOPK x cdiv(N2, BLOCK_N) programs) -----
    _w1_silu_kernel[(top_k, triton.cdiv(N2, block_n_w1))](
        hidden,
        w1,
        w1_scale,
        topk_ids,
        intermediate,
        K_HIDDEN=K_hidden,
        N1=N1,
        N2=N2,
        GROUP_SIZE=group_size,
        TOPK=top_k,
        ZP_BIAS=zp_bias,
        BLOCK_N=block_n_w1,
        BLOCK_K=block_k,
        num_warps=4,
    )

    # ----- 3. W2 + reduce (cdiv(K_hidden, BLOCK_K_OUT) programs) -----
    _w2_reduce_kernel[(triton.cdiv(K_hidden, block_k_out),)](
        intermediate,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        output,
        K_HIDDEN=K_hidden,
        N2=N2,
        GROUP_SIZE=group_size,
        TOPK=top_k,
        ZP_BIAS=zp_bias,
        BLOCK_K_OUT=block_k_out,
        BLOCK_N2=block_n2,
        num_warps=4,
    )

    return output


def moe_megakernel_post_router(
    hidden: torch.Tensor,  # [1, K_hidden] bf16
    w1: torch.Tensor,  # [E, N1, K_hidden//8] int32
    w1_scale: torch.Tensor,  # [E, N1, K_hidden//G] bf16
    w2: torch.Tensor,  # [E, K_hidden, N2//8] int32
    w2_scale: torch.Tensor,  # [E, K_hidden, N2//G] bf16
    topk_ids: torch.Tensor,  # [1, top_k] int32
    topk_weights: torch.Tensor,  # [1, top_k] fp32
    output: torch.Tensor,  # [1, K_hidden] bf16 OUT
    *,
    top_k: int,
    group_size: int,
    zp_bias: int = 8,
    block_k_out: int = 64,
    block_k: int = 64,
    block_n_w1: int = 32,
    block_n2: int = 64,
):
    """Post-router MoE megakernel.

    Skips the router/softmax/topk kernel and uses precomputed top-k IDs
    and weights from the caller (typical vLLM dispatch path: ``apply()``
    is called with topk already routed).

    The two remaining launches (W1+silu, W2+reduce) form the "mega"
    fusion: per-expert intermediate stays in a small HBM scratch
    (TOPK*N2 fp32, e.g. 8*768*4 = 24 KB) instead of round-tripping
    through workspace tensors as in the standard dispatcher.
    """
    K_hidden, E, N1, N2 = _validate_shapes(hidden, w1, w2)
    block_k = min(block_k, group_size)
    block_n2 = min(block_n2, group_size)

    intermediate, mk_topk_ids, mk_topk_weights = _get_buffers(top_k, N2, hidden.device)

    # Copy routed topk into the megakernel's persistent buffers (the kernels
    # take 1D pointers, and the caller may pass a 2D [1, top_k] tensor).
    mk_topk_ids.copy_(topk_ids.view(-1).to(torch.int32))
    mk_topk_weights.copy_(topk_weights.view(-1).to(torch.float32))

    _w1_silu_kernel[(top_k, triton.cdiv(N2, block_n_w1))](
        hidden,
        w1,
        w1_scale,
        mk_topk_ids,
        intermediate,
        K_HIDDEN=K_hidden,
        N1=N1,
        N2=N2,
        GROUP_SIZE=group_size,
        TOPK=top_k,
        ZP_BIAS=zp_bias,
        BLOCK_N=block_n_w1,
        BLOCK_K=block_k,
        num_warps=4,
    )
    _w2_reduce_kernel[(triton.cdiv(K_hidden, block_k_out),)](
        intermediate,
        w2,
        w2_scale,
        mk_topk_ids,
        mk_topk_weights,
        output,
        K_HIDDEN=K_hidden,
        N2=N2,
        GROUP_SIZE=group_size,
        TOPK=top_k,
        ZP_BIAS=zp_bias,
        BLOCK_K_OUT=block_k_out,
        BLOCK_N2=block_n2,
        num_warps=4,
    )
    return output
