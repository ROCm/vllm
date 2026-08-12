# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A pad-aware batched MoE activation -- bnell's TODO, landed.

``fused_batched_moe.py`` calls::

    # TODO (bnell): use triton utility from batched deep gemm.
    self.activation(
        activation,
        intermediate_cache2.view(-1, activation_out_dim),
        intermediate_cache1.view(-1, N),
    )

with no mask arguments, so it writes **every** row of a
``[E, max_num_tokens, N]`` buffer of which only the rows MM1 was delivered were
written; the rest is elementwise work on rows no GEMM will read.

**This is not a new kernel.** It is ``_silu_mul_fp8_quant_deep_gemm``
(``experts/batched_deep_gemm_moe.py``) with the scale side-output removed and
an extent mask kept. That kernel is already batched-layout
``(E, T, 2H) -> (E, T, H)``, already masks on ``counts_ptr =
expert_num_tokens``, already loops tokens on the device (so a static grid is
capture-safe), and already parameterises its numerics as ``constexpr``. What it
cannot do is serve a per-expert or per-tensor scale, which is a *global*
reduction and cannot be finished inside a single token's iteration -- so the
reduction stays with the quantizer and this kernel only writes activations.

**Safety.** This kernel leaves ``out``'s dead rows holding whatever the shared
workspace held, so every consumer of ``out`` must be bounded to the delivered
rows -- in particular the a2 quantize, whose per-tensor amax would otherwise
take its value from an arbitrary earlier allocation.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _silu_mul_masked_batched_kernel(
    input_ptr,  # (E, T, 2H) 16/32-bit activations
    out_ptr,  # (E, T, H)
    counts_ptr,  # (E,) int32 expert_num_tokens
    T,
    H,
    stride_i_e,
    stride_i_t,
    stride_i_h,
    stride_o_e,
    stride_o_t,
    stride_o_h,
    stride_c_e,
    MASKED: tl.constexpr,
    CANON_ZERO: tl.constexpr,
    ROUND_TRIP: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """``silu(gate) * up`` over the rows the expert was actually delivered.

    The grid tiles the token dimension, unlike the template
    ``_silu_mul_fp8_quant_deep_gemm``, which loops tokens on the device because
    each program must own a whole row of its group to write a per-(token,
    group) scale. This kernel writes activations only, so the token dimension
    is free -- and it must be taken: the template's grid is one workgroup per
    (expert, H-group), which underfills a large GPU at realistic E and H.

    Two lines are load-bearing and are *not* ulp fussiness -- they are what
    makes this bit-identical to ``torch.ops._C.silu_and_mul`` in bf16
    (0 of 2^32 gate/up pairs mismatch, exhaustively):

    * ``ROUND_TRIP`` -- ``packed_silu_kernel`` returns the *storage* dtype, so
      silu is rounded before the multiply and only then widened again.
    * ``CANON_ZERO`` -- the CUDA kernel multiplies by ``up + beta`` with
      ``beta = 0.0``, which turns a ``-0.0`` up into ``+0.0``.  ``ops.moe_sum``'s
      exactness argument is "an accumulator initialised to +0.0 cannot become
      -0.0"; ``_swiglu_limit_pad_aware_kernel``, which lacks this line, violates
      it.

    Note the divide: ``gate / (1 + exp(-gate))``, which is the form measured
    bit-exact against the CUDA kernel.  ``_silu_mul_fp8_quant_deep_gemm`` writes
    ``gate * (1 / (1 + exp(-gate)))`` -- a reciprocal multiply, and a different
    result.  That difference is invisible there because its output is quantized
    to fp8 per group immediately; here the bf16 buffer is the thing an A/B
    compares.
    """
    e = tl.program_id(0).to(tl.int64)
    m_block = tl.program_id(1)
    h_block = tl.program_id(2).to(tl.int64)

    offs_m = m_block * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)

    row_ok = offs_m < T
    if MASKED:
        # Whole-tile early exit, then a per-row mask for the partial tile.
        # Precondition, the same one `batched_triton_kernel` already makes:
        # `expert_num_tokens[e] <= T`.  Not clamped -- a clamp would hide a
        # dispatch that overflowed its own buffer -- but `row_ok` bounds the
        # accesses regardless, which the template's token loop did not.
        n_tokens = tl.load(counts_ptr + e * stride_c_e)
        if m_block * BLOCK_T >= n_tokens:
            return
        row_ok = row_ok & (offs_m < n_tokens)

    cmask = offs_h < H
    mask = row_ok[:, None] & cmask[None, :]

    gate = tl.load(
        input_ptr
        + e * stride_i_e
        + offs_m[:, None] * stride_i_t
        + offs_h[None, :] * stride_i_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        input_ptr
        + e * stride_i_e
        + offs_m[:, None] * stride_i_t
        + (offs_h[None, :] + H) * stride_i_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    silu = gate / (1.0 + tl.exp(-gate))
    if ROUND_TRIP:
        silu = silu.to(out_ptr.dtype.element_ty).to(tl.float32)
    if CANON_ZERO:
        up = tl.where(up == 0.0, 0.0, up)

    y = (silu * up).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr
        + e * stride_o_e
        + offs_m[:, None] * stride_o_t
        + offs_h[None, :] * stride_o_h,
        y,
        mask=mask,
    )


def silu_mul_batched_is_exact(dtype: torch.dtype) -> bool:
    """Whether this kernel reproduces ``torch.ops._C.silu_and_mul`` bit for bit.

    Same gate and the same reason as ``silu_and_mul_is_pad_aware``: no spelling
    of the Triton side reproduces HIP's ``expf`` in fp32, and bf16 rounds the
    difference away.  A bitwise claim only -- the fp16 and fp32 deviation stays
    under 1e-6 relative and all but a handful of elements per 4M survive fp8
    quantization unchanged, so this gates the acceptance test rather than
    correctness.
    """
    return dtype == torch.bfloat16


def silu_mul_batched(
    out: torch.Tensor,  # (E, T, H)
    inp: torch.Tensor,  # (E, T, 2H), gate first half, up second
    expert_num_tokens: torch.Tensor | None = None,
    *,
    canon_zero: bool = True,
    round_trip: bool = True,
    block_t: int = 8,
    block_h: int = 512,
    num_warps: int = 4,
) -> None:
    """In-place ``out[e, :n_e] = silu(inp[e, :n_e, :H]) * inp[e, :n_e, H:]``.

    Rows ``>= expert_num_tokens[e]`` are **left untouched** when a mask is
    given. See the module docstring's safety note before calling with one.
    """
    assert inp.ndim == 3 and out.ndim == 3, (inp.shape, out.shape)
    E, T, H2 = inp.shape
    assert H2 % 2 == 0, inp.shape
    H = H2 // 2
    assert out.shape == (E, T, H), (out.shape, (E, T, H))
    if E == 0 or T == 0 or H == 0:
        return

    masked = expert_num_tokens is not None
    if expert_num_tokens is not None:
        expert_num_tokens = expert_num_tokens.to(torch.int32)
        assert expert_num_tokens.ndim == 1 and expert_num_tokens.numel() == E
        stride_c_e = expert_num_tokens.stride(0)
    else:
        stride_c_e = 0

    nm, nh = triton.cdiv(T, block_t), triton.cdiv(H, block_h)
    si_e, si_t, si_h = inp.stride()
    so_e, so_t, so_h = out.stride()

    grid = (E, nm, nh)
    _silu_mul_masked_batched_kernel[grid](
        inp,
        out,
        expert_num_tokens,
        T,
        H,
        si_e,
        si_t,
        si_h,
        so_e,
        so_t,
        so_h,
        stride_c_e,
        MASKED=masked,
        CANON_ZERO=canon_zero,
        ROUND_TRIP=round_trip,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
