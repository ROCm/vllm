# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A single Triton dynamic-quantization kernel, parameterised on three axes.

``SCALE_GRANULARITY``
    ``PER_TENSOR`` -- one scale for the whole buffer, matching
    ``ops.scaled_fp8_quant``'s dynamic path.
    ``PER_EXPERT`` -- one scale per expert of a batched ``[E, T, K]`` buffer,
    matching what the eager per-expert loop in
    ``batched_moe_kernel_quantize_input`` already produces.

``MASK_MODE`` -- named by *semantics*, not by mechanism.
    ``MASK_NONE`` -- every row counts.
    ``MASK_DELIVERED`` -- ``row < expert_num_tokens[e]``. The rows an all2all
    *delivered* to this expert. Under DP this population **includes other
    ranks' dummy tokens**: they are real deliveries as far as the receiving
    expert is concerned.
    ``MASK_ROUTED`` -- ``topk_ids[row] >= 0`` and, with an expert map,
    ``expert_map[topk_ids[row]] >= 0``. The rows that are *valid*, which
    excludes those dummies.
    Delivered and routed are **different populations**, not two spellings of
    one predicate. Conflating them is what makes a bound look like a fix and
    behave like a no-op. On the batched layout only ``MASK_DELIVERED`` is
    constructible -- ``DeepEPLLPrepareAndFinalize._receiver`` returns
    ``expert_topk_ids = None`` -- which is why the routing predicate is
    applied at the source and not here.

Dtype is the third axis and costs four constants: ``qmax`` (448.0 / 127.0), the
clamp pair, and the divide convention -- see ``_QuantSpec``.
"""

import torch

from vllm.triton_utils import tl, triton

PER_TENSOR: int = 0
PER_EXPERT: int = 1

MASK_NONE: int = 0
MASK_DELIVERED: int = 1
MASK_ROUTED: int = 2

# Triton refuses to read a plain global from a @jit'ed body, so the same values
# are re-exported as constexpr for the kernels to compare against.
_PER_TENSOR = tl.constexpr(PER_TENSOR)
_MASK_DELIVERED = tl.constexpr(MASK_DELIVERED)
_MASK_ROUTED = tl.constexpr(MASK_ROUTED)


@triton.jit
def _row_mask(
    e,
    offs_m,
    T,
    ent_ptr,
    tid_ptr,
    emap_ptr,
    MASK_MODE: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
):
    """Which of ``offs_m`` are rows this reduction is allowed to see."""
    ok = offs_m < T
    if MASK_MODE == _MASK_DELIVERED:
        ok = ok & (offs_m < tl.load(ent_ptr + e))
    elif MASK_MODE == _MASK_ROUTED:
        # `tid_ptr` is the flattened [E*T] (batched) or [num_tokens*topk]
        # (contiguous, E == 1) slot -> expert id array, the same one
        # `_silu_and_mul_pad_aware_kernel` reads.
        eid = tl.load(tid_ptr + e * T + offs_m, mask=ok, other=-1)
        routed = eid >= 0
        if HAS_EXPERT_MAP:
            routed = routed & (tl.load(emap_ptr + eid, mask=routed, other=-1) >= 0)
        ok = ok & routed
    return ok


@triton.jit
def _masked_amax_kernel(
    a_ptr,
    ent_ptr,
    tid_ptr,
    emap_ptr,
    scale_ptr,
    part_ptr,
    T,
    K,
    QMAX,
    stride_ae,
    stride_am,
    stride_ak,
    NM: tl.constexpr,
    USE_PARTIALS: tl.constexpr,
    SCALE_GRANULARITY: tl.constexpr,
    MASK_MODE: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Pass 1 of 2: amax over the rows the mask admits, combined atomically.

    ``atomic_max`` is the right cross-block combine, and this is the obvious
    reviewer question. Max is associative, commutative and *exact* in fp32, so
    the result does not depend on which block lands first, on how many blocks
    there are, or on the tile shape. A sum has none of those properties; this is
    why the per-tensor amax can be tiled freely and a per-tensor mean could not.
    Order independence here is a property of the operator, not of the launch.

    The division by ``QMAX`` happens *before* the atomic, so this kernel writes
    the scale directly and the whole op stays at two launches. That is exact:
    correctly-rounded division by a positive constant is monotone, so
    ``max(a_i) / q == max(a_i / q)`` bit for bit. It is also what the HIP kernel
    does (``atomicMaxFloat(scale, cache[0] / quant_type_max_v)``).

    Divide, do not reciprocal-multiply: the two differ for a large fraction of
    inputs and the HIP kernel divides. ``ieee_rounding=True`` is a no-op on
    ROCm but is required on NVIDIA, where the default lowers to the approximate
    ``div.full.f32``.
    """
    e = tl.program_id(0).to(tl.int64)
    m_block = tl.program_id(1)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    # Not merged into one `and`: `ent_ptr` is None unless the mask mode uses
    # it, so the load must stay inside the constexpr guard.
    if MASK_MODE == _MASK_DELIVERED:  # noqa: SIM102
        # Whole-block early exit; most of the buffer is typically dead.
        if m_block * BLOCK_M >= tl.load(ent_ptr + e):
            if USE_PARTIALS:
                # A tile with nothing to contribute still owns its slot, so the
                # buffer never needs pre-initialising and 0.0 -- the identity
                # of a max over absolute values -- is written rather than
                # memset.
                tl.store(part_ptr + e * NM + m_block, 0.0)
            return

    row_ok = _row_mask(
        e, offs_m, T, ent_ptr, tid_ptr, emap_ptr, MASK_MODE, HAS_EXPERT_MAP
    )

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask = row_ok[:, None] & (offs_k[None, :] < K)
        vals = tl.load(
            a_ptr
            + e * stride_ae
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        vals = tl.abs(vals)
        # `fmaxf` discards NaN where `torch.amax` propagates it, and the HIP
        # accumulator starts at 0.0, so a NaN already in the buffer can never
        # set the scale. Match that explicitly rather than rely on how
        # `tl.maximum` lowers.
        vals = tl.where(vals != vals, 0.0, vals)
        acc = tl.maximum(acc, vals)

    # No floor: neither shipped per-tensor kernel applies one.
    s = tl.fdiv(tl.max(acc), QMAX, ieee_rounding=True)
    if USE_PARTIALS:
        tl.store(part_ptr + e * NM + m_block, s)
    elif SCALE_GRANULARITY == _PER_TENSOR:
        tl.atomic_max(scale_ptr, s)
    else:
        tl.atomic_max(scale_ptr + e, s)


@triton.jit
def _masked_quant_kernel(
    a_ptr,
    q_ptr,
    scale_ptr,
    part_ptr,
    ent_ptr,
    tid_ptr,
    emap_ptr,
    T,
    K,
    QLO,
    QHI,
    stride_ae,
    stride_am,
    stride_ak,
    stride_qe,
    stride_qm,
    stride_qk,
    NM: tl.constexpr,
    BLOCK_NM: tl.constexpr,
    USE_PARTIALS: tl.constexpr,
    SCALE_GRANULARITY: tl.constexpr,
    MASK_MODE: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    USE_RECIPROCAL: tl.constexpr,
    ROUND_TO_INT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Pass 2 of 2: quantize only the rows the mask admits.

    Dead rows are **not written**. They are not zeroed either: a fill is a
    second, invisible invariant that drifts away from the mask, and the
    consumers already skip them (the batched MM2 early-exits on
    ``cta_m_start >= e_num_tokens``; ``fused_moe_kernel`` never puts an
    unrouted slot in ``sorted_token_ids``).

    The divide convention is per dtype and is the reverse of pass 1's:

    * fp8 -- the HIP kernel computes ``reciprocal_scale = 1.0f / (*scale)``
      once and then *multiplies*. To be bit-identical we must do the same.
    * int8 -- the HIP kernel *divides* each element (``src / scale``) and then
      ``nearbyint``s. To be bit-identical we must do that instead.

    So "divide, do not reciprocal-multiply" applies to the scale, and the
    opposite applies to the elements, for fp8. Both halves are load-bearing.
    """
    e = tl.program_id(0).to(tl.int64)
    m_block = tl.program_id(1)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    if USE_PARTIALS:
        # Every block folds the same NM partials redundantly.  The alternative
        # is a third launch, which costs more than the fold at any NM this
        # kernel sees.  Done before the early exit so that an expert with no
        # live row still gets its scale written.
        offs_r = tl.arange(0, BLOCK_NM)
        parts = tl.load(part_ptr + e * NM + offs_r, mask=offs_r < NM, other=0.0)
        s_fold = tl.max(parts)
        if m_block == 0:
            tl.store(scale_ptr + e, s_fold)

    if MASK_MODE == _MASK_DELIVERED:  # noqa: SIM102  (see pass 1)
        if m_block * BLOCK_M >= tl.load(ent_ptr + e):
            return

    row_ok = _row_mask(
        e, offs_m, T, ent_ptr, tid_ptr, emap_ptr, MASK_MODE, HAS_EXPERT_MAP
    )

    zero_m = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if USE_PARTIALS:
        s = s_fold + zero_m
    elif SCALE_GRANULARITY == _PER_TENSOR:
        s = tl.load(scale_ptr) + zero_m
    else:
        s = tl.load(scale_ptr + e) + zero_m

    # An expert with no live row, or whose live rows are all exactly zero, gives
    # amax 0 and hence scale 0. Every value it quantizes is 0, and 0 is the
    # right answer under any positive scale, so produce it directly instead of
    # inventing an epsilon. See `dynamic_quantize.__doc__`.
    degenerate = s == 0.0
    denom = tl.where(degenerate, 1.0, s)
    recip = 1.0 / denom

    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask = row_ok[:, None] & (offs_k[None, :] < K)
        vals = tl.load(
            a_ptr
            + e * stride_ae
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        if USE_RECIPROCAL:
            x = vals * recip[:, None]
        else:
            x = tl.fdiv(vals, denom[:, None], ieee_rounding=True)
        x = tl.where(degenerate[:, None], 0.0, x)

        if ROUND_TO_INT:
            # `float_to_int8_rn`: nearbyint under FE_TONEAREST, then saturate
            # into an *asymmetric* [-128, 127].
            x = tl.extra.libdevice.nearbyint(x)
            x = tl.clamp(x, QLO, QHI)
        else:
            # `scaled_fp8_conversion`: saturate first, then the hardware cvt.
            # `tl.clamp` with the default NaN policy is `fmaxf`/`fminf`.
            x = tl.clamp(x, QLO, QHI)

        tl.store(
            q_ptr
            + e * stride_qe
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk,
            x.to(q_ptr.dtype.element_ty),
            mask=mask,
        )


class _QuantSpec:
    """The per-dtype cost of this kernel: three constants and two flags."""

    __slots__ = ("qmax", "qlo", "qhi", "use_reciprocal", "round_to_int")

    def __init__(self, qmax, qlo, qhi, use_reciprocal, round_to_int):
        self.qmax = qmax
        self.qlo = qlo
        self.qhi = qhi
        self.use_reciprocal = use_reciprocal
        self.round_to_int = round_to_int


def _spec(dtype: torch.dtype) -> _QuantSpec:
    """The constants that make this kernel match the shipped one for ``dtype``.

    Neither shipped per-tensor kernel floors the scale, so neither does this,
    and `_int8_quantize`'s python-side 1e-10 floor only changes the returned
    scale value, never the emitted bytes.
    """
    if dtype == torch.int8:
        info = torch.iinfo(dtype)
        return _QuantSpec(127.0, float(info.min), float(info.max), False, True)
    info = torch.finfo(dtype)
    return _QuantSpec(info.max, -info.max, info.max, True, False)


def _next_pow2(n: int) -> int:
    p = 16  # Triton wants a power of two; 16 keeps the E=1 case legal
    while p < n:
        p *= 2
    return p


def dynamic_quantize(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    *,
    granularity: int = PER_TENSOR,
    mask_mode: int = MASK_NONE,
    expert_num_tokens: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    block_m: int = 8,
    block_k: int = 1024,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic symmetric quantization with a valid-extent bound.

    ``x`` is ``[E, T, K]`` (batched) or ``[M, K]`` (contiguous, taken as
    ``E == 1``). Two kernel launches, independent of ``E``; the per-expert loop
    it replaces was ``E`` launches over the *full* ``[max_num_tokens, hidden]``
    slice including the dead rows.

    Returns ``(q, scale)`` with ``scale`` shaped ``[1]`` for ``PER_TENSOR`` and
    ``[E, 1, 1]`` for ``PER_EXPERT`` -- the shapes the two existing branches of
    ``batched_moe_kernel_quantize_input`` already return.

    An expert with no live row gives amax 0, and this returns that scale
    unfloored and emits exact zeros. ``ops.scaled_fp8_quant`` instead emits
    448.0 in every element there (``0 * (1/0)`` is NaN and ``fminf(NaN, 448)``
    returns 448), which dequantizes to 0.0 only because the dequantize is a
    multiply by the 0.0 scale.
    """
    assert x.ndim in (2, 3), x.shape
    batched = x.ndim == 3
    E, T, K = x.shape if batched else (1, x.shape[0], x.shape[1])

    if mask_mode == MASK_DELIVERED:
        assert expert_num_tokens is not None
        assert batched, "delivered-row masking is only defined on [E, T, K]"
        expert_num_tokens = expert_num_tokens.to(torch.int32)
    if mask_mode == MASK_ROUTED:
        assert topk_ids is not None
        topk_ids = topk_ids.reshape(-1)
        assert topk_ids.numel() == E * T, (topk_ids.numel(), E, T)
    if granularity == PER_EXPERT:
        assert batched, "per-expert scales are only defined on [E, T, K]"

    spec = _spec(quant_dtype)
    q = torch.empty_like(x, dtype=quant_dtype) if out is None else out

    n_scales = E if granularity == PER_EXPERT else 1

    sa = x.stride() if batched else (0, *x.stride())
    sq = q.stride() if batched else (0, *q.stride())
    nm = triton.cdiv(T, block_m)
    grid = (E, nm)

    # Per-expert with a reduction to do: each tile stores a plain partial and
    # pass 2 folds them, instead of `atomic_max` into a buffer that first has
    # to be memset.  One fewer device node, no atomic, and bit-identical --
    # max is exact and order-independent, so the fold order cannot matter.
    # Restricted to PER_EXPERT: at per-tensor granularity every block would
    # have to fold E*NM partials rather than NM, and the atomic is into a
    # single address anyway.
    use_partials = granularity == PER_EXPERT
    part = (
        torch.empty(E * nm, dtype=torch.float32, device=x.device)
        if use_partials
        else None
    )
    common = dict(
        NM=nm,
        USE_PARTIALS=use_partials,
        SCALE_GRANULARITY=granularity,
        MASK_MODE=mask_mode,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
    )

    # With partials, pass 2's designated block writes every entry, so no
    # pre-initialisation.  Without them, `atomic_max` needs its identity
    # element in place first -- a memset, not a third kernel.
    s_flat = (
        torch.empty(n_scales, dtype=torch.float32, device=x.device)
        if use_partials
        else torch.zeros(n_scales, dtype=torch.float32, device=x.device)
    )
    _masked_amax_kernel[grid](
        x, expert_num_tokens, topk_ids, expert_map, s_flat, part,
        T, K, spec.qmax, sa[0], sa[1], sa[2], **common,
    )  # fmt: skip

    _masked_quant_kernel[grid](
        x, q, s_flat, part, expert_num_tokens, topk_ids, expert_map,
        T, K, spec.qlo, spec.qhi,
        sa[0], sa[1], sa[2], sq[0], sq[1], sq[2],
        USE_RECIPROCAL=spec.use_reciprocal,
        ROUND_TO_INT=spec.round_to_int,
        BLOCK_NM=_next_pow2(nm),
        **common,
    )  # fmt: skip

    if granularity == PER_EXPERT:
        return q, s_flat.view(E, 1, 1)
    return q, s_flat
