# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch for the wide MHA decode-attention kernel (gfx1151 / RDNA3.5).

Enabled by default on gfx1151, where the kernel is built and tuned; set
``VLLM_ROCM_WIDE_DECODE_ATTN=0`` to force the Triton path or ``=1`` to
force-enable.  When the shape qualifies, this replaces Triton's unified
attention for the decode part of the batch; anything else falls through
unchanged, so the path is A/B-able.

The kernel is compiled into ``_rocm_C`` (``csrc/rocm/wide_decode_attn.cu``; the
body is gfx11-only, a stub elsewhere) and exposed as
``torch.ops._rocm_C.wide_decode_attn``.  This module makes the dispatch decision
host-side and wraps the call as a registered vLLM custom op with a no-op fake,
so the path is graph-safe under torch.compile.

Why the constraints below are what they are -- each is a property of the kernel,
not a policy choice, and routing a case that violates one produces wrong output
rather than an error:

* **MHA only.**  The kernel assigns one query head per KV head by construction
  (``q_head = kv_head``), so GQA and MQA shapes would read the wrong head.
* **head_size in {64,128,256,512}.**  The head dimension is a template
  parameter and splits across a 32-lane wave as ``head_size/64`` vec2 per lane.
* **M in {1,2,3,4}.**  Only M=1 and M=4 are instantiated; 2 and 3 are padded up
  to 4 here.  Padding is cheap (a measured geomean 1.024x, worst 1.081x) and
  keeps every launch on one of the two M values the tuning sweep covered.
* **fp16 / bf16, unquantized KV.**  There is no scale parameter in the ABI, so
  an fp8 cache would be read as raw bits.
* **Causal, no sliding window, no ALiBi, no sinks.**  The causal mask is
  unconditional and the other three do not exist in the kernel.
* **NHD cache layout.**  The kernel indexes
  ``block_id*block_stride + (pos%block_size)*num_kv_heads*head_size +
  kv_head*head_size``; the HND stride order permutes block_size and
  num_kv_heads, which silently changes what it reads.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.platforms.rocm import on_gfx1151
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.kv_cache_interface import KVQuantMode

# Instantiated head dimensions. head_size % 64 == 0 is a static_assert in the
# kernel; these four are what the dispatch actually builds.
SUPPORTED_HEAD_SIZES = (64, 128, 256, 512)

# Shapes the kernel is instantiated for but LOSES on, so the predicate declines
# them and Triton keeps the work. Keyed on (num_heads, head_size, max_query_len)
# because the answer depends on all three.
#
# Only one entry, and only at M=1. head_size 64 is the shape where a lane holds
# the least: even with KV heads grouped 4 to a wave (b128 loads, 3-step
# reduction) the kernel does one dot2 per token where Triton feeds a 16x16x16
# WMMA tile, reusing each K across 16 rows. At M=1 there are no query rows to
# amortise, so Triton's tile wins on traffic per FLOP.
#
# Measured against Triton's 2D path (tuned/T2D), after head grouping:
#
#   Hq=8   1.33x / 2.13x      Hq=32 M=1  0.90x / 0.96x   <-- declined
#   Hq=16  1.05x / 1.26x      Hq=32 M=4  1.19x / 1.42x   <-- kept
#   Hq=64  1.07x / 1.10x                       (ctx 512 / 4096)
#
# This is not a tuning gap. With every axis open the rule is within geomean
# 1.021 of the best config per cell there, both kernels sit at 97% of the
# machine's measured streaming ceiling (216 GiB/s, not the 238 nominal), and
# neither more waves (8x) nor deeper unroll (8x) moves it. Closing it needs a
# WMMA kernel, not a config.
_LOSING_SHAPES = frozenset({(32, 64, 1)})  # (num_heads, head_size, max_query_len)

# Only M=1 (plain decode) and M=4 (speculative) are instantiated. Query lengths
# 2 and 3 run on the M=4 kernel with the leading rows padded.
KERNEL_M = 4
MAX_QUERY_LEN = 4


def is_enabled() -> bool:
    """Tri-state: unset = default-on on gfx1151, "1" forces on, "0" forces off."""
    val = envs.VLLM_ROCM_WIDE_DECODE_ATTN
    if val is not None:
        return val == "1"
    return on_gfx1151()


def tuned_num_kv_segments(num_q_tokens: int, head_size: int, num_kv_heads: int) -> int:
    """The KV segment count the kernel will pick, mirroring wide_attn_tuned().

    Must agree with the C++ rule exactly: the partial buffers are indexed
    ``[num_seqs, num_heads, nseg, M, head_size]``, so a disagreement would be a
    memory fault rather than a compile error. The op re-derives the count and
    checks it against ``partial_out.size(2)`` with TORCH_CHECK, which turns a
    drift between the two copies into an exception instead of corruption.

    The result is NOT always a power of two, and must not be rounded to one: it
    targets a constant task count, and rounding down costs 11-43% of the
    parallelism.

    Both constants are conditioned on head_size 64, where a wave groups several
    KV heads (see :func:`tuned_heads_per_wave`) and each lane therefore holds a
    quarter of what it holds at 128. Applying either everywhere regresses
    head_size >= 128 -- measured, not assumed.
    """
    vec2_per_lane = head_size // 64
    numerator = 256 if vec2_per_lane == 1 else 512
    cap = 32 if vec2_per_lane == 1 else 16
    segments = numerator // (num_kv_heads * vec2_per_lane)
    return max(1, min(cap, segments))


def tuned_heads_per_wave(head_size: int, num_kv_heads: int) -> int:
    """Adjacent KV heads one wave loads together, mirroring wide_attn_tuned().

    Only at head_size 64. There a single head leaves each lane with 4 bytes and
    the compiler emits ``global_load_b32``, while the whole 32-lane wave reduces
    one dot2 per token -- the DPP chain becomes most of the inner loop. Adjacent
    KV heads are contiguous in the paged layout, so grouping them widens the
    lane to 16 bytes (``b128``) and shortens the reduction from 5 steps to 3.

    Above head_size 64 a single head already fills the lane, so grouping buys no
    load width and this returns 1.
    """
    if head_size // 64 != 1:
        return 1
    want = 2 if num_kv_heads <= 16 else 4
    for c in (want, 2):
        if num_kv_heads % c == 0:
            return c
    return 1


def can_run(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_query_len: int,
    dtype: torch.dtype,
    kv_quant_mode: KVQuantMode,
    alibi_slopes: torch.Tensor | None,
    sinks: torch.Tensor | None,
    sliding_window: int | None,
    output_scale: torch.Tensor | None,
    kv_cache_layout: str,
    causal: bool = True,
) -> bool:
    """True iff this attention layer's decode batch can run on the kernel.

    Takes scalars and dtypes rather than tensors so the result is cacheable by
    the caller and cheap to evaluate per forward.
    """
    if not is_enabled() or not on_gfx1151():
        return False
    # MHA only: one query head per KV head is baked into the wave mapping.
    if num_heads != num_kv_heads:
        return False
    if head_size not in SUPPORTED_HEAD_SIZES:
        return False
    if not 1 <= max_query_len <= MAX_QUERY_LEN:
        return False
    # Instantiated but slower than Triton on this shape; see _LOSING_SHAPES.
    # Bounds-checked first, since the key below is the kernel M this maps to:
    # query lengths 2-4 all run on the M=4 kernel and share its verdict.
    kernel_m = 1 if max_query_len == 1 else KERNEL_M
    if (num_heads, head_size, kernel_m) in _LOSING_SHAPES:
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    # No dequant in the kernel: there is no k_scale/v_scale in the ABI at all,
    # so a quantized cache would be read as raw bits.
    if kv_quant_mode != KVQuantMode.NONE:
        return False
    if alibi_slopes is not None or sinks is not None:
        return False
    # The MHA kernel has no window parameter (its GQA sibling does).
    if sliding_window is not None and sliding_window > 0:
        return False
    # No fused output quantization.
    if output_scale is not None:
        return False
    # HND permutes block_size and num_kv_heads inside a block.
    if kv_cache_layout != "NHD":
        return False
    # The causal mask is unconditional.
    return causal


def workspace_shapes(
    max_num_seqs: int, num_heads: int, head_size: int, num_kv_heads: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Shapes for the fp32 partial buffers, sized for the worst case.

    Returns ``(partial_out_shape, partial_stat_shape)``. Sized at M=KERNEL_M
    because query lengths 2 and 3 are padded up to it, and at the segment count
    for that M -- the count does not depend on M, but deriving it from the same
    helper keeps the two in step.
    """
    nseg = tuned_num_kv_segments(KERNEL_M, head_size, num_kv_heads)
    return (
        (max_num_seqs, num_heads, nseg, KERNEL_M, head_size),
        (max_num_seqs, num_heads, nseg, KERNEL_M),
    )


def build_gather_index(
    query_start_loc: torch.Tensor, num_decodes: int, scratch_row: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row indices mapping the padded [num_decodes, KERNEL_M] block to/from Q.

    vLLM's query is varlen and flat -- ``[num_tokens, num_heads, head_size]``
    with ``query_start_loc`` marking each request -- while the kernel wants a
    dense ``[num_seqs, M, num_heads, head_size]`` block with a fixed M. This
    builds the indices for that copy and for the copy back.

    Padding goes at the FRONT, which is forced by the kernel's mask: it applies
    ``pos > ctx_len + m`` with ``ctx_len = seq_len - M``, so row ``M-1`` is the
    one that sees the real last token. Real query token ``j`` of a request with
    ``len_i`` tokens therefore lands at row ``M - len_i + j``, where the mask
    lets it attend to exactly ``[0, num_computed + j]`` -- the ``-M`` and the
    ``+m`` cancel, which is why ``seq_lens`` is passed through unmodified and
    why this reproduces Triton's ``context_len = seq_len - cur_batch_query_len``
    exactly.

    Padding rows duplicate the request's first real query rather than being
    zeroed: a zero row would be a softmax over a null vector, and the values are
    discarded on the way back anyway. If ``seq_len < M`` the leading rows come
    out fully masked, which the reduce turns into zeros rather than NaN.

    Both returned tensors have a FIXED shape of ``num_decodes * KERNEL_M``,
    independent of the query lengths. That matters: the obvious formulation
    selects the real rows with a boolean mask, but a mask produces a
    data-dependent output shape, which forces a device-to-host sync to
    materialize and cannot be captured in a cudagraph -- and TRITON_ATTN
    declares AttentionCGSupport.ALWAYS. Instead every padded row gets a
    destination, and the padding ones are aimed at ``scratch_row``: a slot the
    caller appends past the real output, whose contents are then discarded.

    Returns ``(gather_rows, scatter_rows)``: ``gather_rows[r]`` is the flat
    query row to copy into padded row ``r``, and ``scatter_rows[r]`` is where
    padded row ``r`` goes on the way back (``scratch_row`` when it is padding).
    """
    starts = query_start_loc[:num_decodes]
    ends = query_start_loc[1 : num_decodes + 1]
    lens = ends - starts

    m_idx = torch.arange(KERNEL_M, device=query_start_loc.device)
    # offset of row m within the request: m - (M - len) == m - M + len
    offset = m_idx.unsqueeze(0) - KERNEL_M + lens.unsqueeze(1)
    real = offset >= 0
    # Padding rows clamp to offset 0, i.e. the request's first real query.
    gather_rows = starts.unsqueeze(1) + offset.clamp(min=0)
    scatter_rows = torch.where(
        real, gather_rows, torch.full_like(gather_rows, scratch_row)
    )
    return gather_rows.reshape(-1), scatter_rows.reshape(-1)


def _wide_decode_attn_impl(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    partial_out: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    scale: float,
    softcap: float,
) -> None:
    """Run the kernel, writing into ``out`` in place.

    Callers MUST gate on :func:`can_run` first. The kernel validates everything
    it can with TORCH_CHECK, so a mismatch raises rather than silently leaving
    ``out`` at its previous contents.
    """
    torch.ops._rocm_C.wide_decode_attn(
        out,
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partial_out,
        partial_max,
        partial_sum,
        float(scale),
        float(softcap),
    )


def _wide_decode_attn_fake(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    partial_out: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    scale: float,
    softcap: float,
) -> None:
    # Everything is pre-allocated and mutated in place; nothing to allocate.
    return None


direct_register_custom_op(
    op_name="rocm_wide_decode_attn",
    op_func=_wide_decode_attn_impl,
    mutates_args=["out", "partial_out", "partial_max", "partial_sum"],
    fake_impl=_wide_decode_attn_fake,
)
