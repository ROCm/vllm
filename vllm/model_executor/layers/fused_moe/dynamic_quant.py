# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A single Triton dynamic-quantization kernel, parameterised on three axes.

Three capabilities beyond a plain dynamic quantize:

1. ``PER_TOKEN`` granularity, so that ``batched_moe_kernel_quantize_input``'s
   per-act-token scheme can go through the same code and the
   ``_is_capturing_or_compiling()`` branch can be deleted rather than narrowed.
   Pass 1 stores a row scale instead of combining atomically -- at row
   granularity one program already owns the whole reduction, so there is no
   atomic. PER_GROUP is deliberately absent: at group granularity the second pass
   buys nothing, and the per-token scale here is a path-count convenience
   rather than a performance claim.
2. ``scale=`` -- a caller-provided *static* scale. Pass 1 is skipped entirely.
   This is what makes the static-per-tensor a2 scheme take the same path.
3. ``amax=`` -- a caller-provided *raw* amax buffer, which is how the fused
   pad-aware batched activation hands over the reduction it did
   for free while it was already reading every live row. The division by
   ``QMAX`` stays here, in one place, rather than being pushed into the
   activation: an amax is an amax, and a producer that divides by 448.0 is
   coupled to the quantization scheme downstream of it.

The three axes:

``SCALE_GRANULARITY``
    ``PER_TENSOR`` -- one scale for the whole buffer, matching
    ``ops.scaled_fp8_quant``'s dynamic path.
    ``PER_EXPERT`` -- one scale per expert of a batched ``[E, T, K]`` buffer,
    matching what the eager per-expert loop in
    ``batched_moe_kernel_quantize_input`` already produces.
    ``PER_TOKEN`` -- one scale per row, ``[E, T, 1]``.

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

Dtype is the third axis and costs three constants: ``qmax`` (448.0 / 127.0),
the clamp pair, and the divide convention -- see ``_QuantSpec``.
"""

import torch

from vllm.triton_utils import tl, triton

PER_TENSOR: int = 0
PER_EXPERT: int = 1
PER_TOKEN: int = 2

MASK_NONE: int = 0
MASK_DELIVERED: int = 1
MASK_ROUTED: int = 2

# Triton refuses to read a plain global from a @jit'ed body, so the same values
# are re-exported as constexpr for the kernels to compare against.
_PER_TENSOR = tl.constexpr(PER_TENSOR)
_PER_EXPERT = tl.constexpr(PER_EXPERT)
_PER_TOKEN = tl.constexpr(PER_TOKEN)
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
    FLOOR,
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

    At ``PER_TOKEN`` there is no cross-block combine at all: one program owns a
    whole row's K loop, so it stores the row scale directly.

    The division by ``QMAX`` happens *before* the atomic, so this kernel writes
    the scale directly and the whole op stays at two launches. That is exact:
    correctly-rounded division by a positive constant is monotone, so
    ``max(a_i) / q == max(a_i / q)`` bit for bit. It is also what the HIP kernel
    does (``atomicMaxFloat(scale, cache[0] / quant_type_max_v)``).

    What is load-bearing is the *divide*, not the flag. Measured on gfx950 over
    1M amax draws: a reciprocal multiply differs from the correctly-rounded
    quotient on 570862/1048576 (54.4%) at qmax=448 and 47509/1048576 at
    qmax=127, and torch's ``a / 448.0`` matches the reciprocal multiply
    byte for byte -- that is the 1-ULP gate.
    Triton's ``/`` on this build is already correctly rounded with the flag off
    (0/1048576 either way), so ``ieee_rounding=True`` buys nothing here; it is
    kept because on NVIDIA the default lowers to the approximate
    ``div.full.f32``, and this kernel must not be correct only on ROCm.
    """
    e = tl.program_id(0).to(tl.int64)
    m_block = tl.program_id(1)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    # Not merged into one `and`: `ent_ptr` is None unless the mask mode uses
    # it, so the load must stay inside the constexpr guard.
    if MASK_MODE == _MASK_DELIVERED:  # noqa: SIM102
        # Whole-block early exit. Measured 1303-3426 of 16384 rows live on this
        # buffer, so this is where the 5-12x of dead work goes away.
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

    # `FLOOR` is 0.0 for every granularity except fp8 PER_TOKEN, where the
    # shipped kernel floors at `1/(qmax*512)`.  See `_spec`: the floor is a
    # property of the *shipped kernel for that granularity*, not of the dtype,
    # and applying it uniformly would change the per-tensor result.  `amax` is
    # non-negative so `maximum(s, 0.0)` is exactly a no-op in the other cases.
    if SCALE_GRANULARITY == _PER_TOKEN:
        # One program owns the row, so no atomic and no second combine. Dead
        # rows keep the buffer's 0.0; nothing reads them (the batched MM2
        # loads a_scale under `mask_m`).
        s = tl.fdiv(tl.max(acc, axis=1), QMAX, ieee_rounding=True)
        tl.store(scale_ptr + e * T + offs_m, tl.maximum(s, FLOOR), mask=row_ok)
    else:
        s = tl.maximum(tl.fdiv(tl.max(acc), QMAX, ieee_rounding=True), FLOOR)
        if USE_PARTIALS:
            tl.store(part_ptr + e * NM + m_block, s)
        elif SCALE_GRANULARITY == _PER_TENSOR:
            tl.atomic_max(scale_ptr, s)
        else:
            tl.atomic_max(scale_ptr + e, s)


@triton.jit
def _scale_from_amax_kernel(
    amax_ptr,
    scale_ptr,
    N,
    R,
    QMAX,
    FLOOR,
    BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """``scale = amax / QMAX`` over an ``N``-element buffer, one launch.

    This exists so that a *producer* which already had to read every live row
    (the fused pad-aware activation) can hand over a raw amax without also
    having to know what dtype it is eventually quantized to. The division stays
    here, in the one place that already gets it right: a correctly-rounded
    ``fdiv``, not torch's scalar division, which lowers to a reciprocal
    multiply and bit-differs from the HIP kernel about half the time.
    """
    offs = tl.arange(0, BLOCK)
    m = offs < N
    # `amax_ptr` is `[N, R]`: R partial maxima per scale, one per producer tile.
    # R == 1 is the plain case. Max is exact and order-independent, so any fold
    # order is bit-identical -- but a *scalar* fold over R is not free: at
    # R = 128 the one-lane-per-scale version measured **+16 us**, which is more
    # than the pass it exists to delete. Tile it.
    offs_r = tl.arange(0, BLOCK_R)
    a = tl.zeros((BLOCK,), dtype=tl.float32)
    for r0 in tl.range(0, R, BLOCK_R):
        rr = r0 + offs_r
        v = tl.load(
            amax_ptr + offs[:, None] * R + rr[None, :],
            mask=m[:, None] & (rr[None, :] < R),
            other=0.0,
        )
        a = tl.maximum(a, tl.max(v, axis=1))
    s = tl.maximum(tl.fdiv(a, QMAX, ieee_rounding=True), FLOOR)
    tl.store(scale_ptr + offs, s, mask=m)


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
    elif SCALE_GRANULARITY == _PER_TOKEN:
        s = tl.load(scale_ptr + e * T + offs_m, mask=row_ok, other=0.0)
    elif SCALE_GRANULARITY == _PER_TENSOR:
        s = tl.load(scale_ptr) + zero_m
    else:
        s = tl.load(scale_ptr + e) + zero_m

    # An expert with no live row, or whose live rows are all exactly zero, gives
    # amax 0 and hence scale 0. Every value it quantizes is 0, and 0 is the
    # right answer under any positive scale, so produce it directly instead of
    # inventing an epsilon. See `dynamic_quantize.__doc__` for what this changes
    # relative to each existing behaviour.
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
    """The per-(dtype, granularity) cost of this kernel: four constants."""

    __slots__ = ("qmax", "qlo", "qhi", "floor", "use_reciprocal", "round_to_int")

    def __init__(self, qmax, qlo, qhi, floor, use_reciprocal, round_to_int):
        self.qmax = qmax
        self.qlo = qlo
        self.qhi = qhi
        self.floor = floor
        self.use_reciprocal = use_reciprocal
        self.round_to_int = round_to_int


def _spec(dtype: torch.dtype, granularity: int) -> _QuantSpec:
    """Keyed on the granularity as well as the dtype, because it has to be.

    The per-dtype cost would be three constants if the shipped fp8 kernels
    were self-consistent.  They are not -- see
    `csrc/libtorch_stable/quantization/w8a8/fp8/common.cu`:

    `scaled_fp8_quant_kernel_strided_dynamic` (per-tensor)
        scale:   `atomicMax(amax / qmax)`, no floor
        element: `1.0f / scale` then multiply

    `dynamic_per_token_scaled_fp8_quant_kernel_strided` (per-token)
        scale:   `fmaxf(amax / qmax, 1 / (qmax * 512))`, floored at 4.36e-6
        element: divide

    So the per-token path floors where the per-tensor path does not, *and*
    divides where the per-tensor path reciprocal-multiplies.  Neither is
    documented as a deliberate difference anywhere I can find.  This kernel
    reproduces both rather than picking one, because the point of a unified
    kernel is to be a drop-in; the *observation* that they differ belongs in a
    bug report, not in a silent unification.
    """
    if dtype == torch.int8:
        info = torch.iinfo(dtype)
        # int8 is unreachable on both MoE paths (see the module note); the
        # shipped `min_scaling_factor<int8_t>` is FLT_EPSILON, and
        # `_int8_quantize`'s python-side floor is 1e-10.  Left unfloored, which
        # is measured to change nothing on the case it looks like it guards.
        return _QuantSpec(127.0, float(info.min), float(info.max), 0.0, False, True)
    info = torch.finfo(dtype)
    floor = 1.0 / (info.max * 512.0) if granularity == PER_TOKEN else 0.0
    use_reciprocal = granularity != PER_TOKEN
    return _QuantSpec(info.max, -info.max, info.max, floor, use_reciprocal, False)


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
    scale: torch.Tensor | None = None,
    amax: torch.Tensor | None = None,
    block_m: int = 8,
    block_k: int = 1024,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic symmetric quantization with a valid-extent bound.

    ``x`` is ``[E, T, K]`` (batched) or ``[M, K]`` (contiguous, taken as
    ``E == 1``). Two kernel launches, independent of ``E``; the per-expert loop
    it replaces was ``E`` launches over the *full* ``[max_num_tokens, hidden]``
    slice including the dead rows.

    Returns ``(q, scale)`` with ``scale`` shaped ``[1]`` for ``PER_TENSOR``,
    ``[E, 1, 1]`` for ``PER_EXPERT`` and ``[E, T, 1]`` for ``PER_TOKEN`` -- the
    shapes the two existing branches of ``batched_moe_kernel_quantize_input``
    already return.

    ``scale`` (in) short-circuits pass 1 with a *static* scale. ``amax`` (in)
    short-circuits pass 1 with a raw amax someone else already computed -- one
    ``N``-element launch replaces a full read of ``x``.

    Zero amax, decided deliberately: the scale is returned unfloored (0.0) and
    the kernel emits exact zeros for the rows it covers.

    * against ``_fp8_quantize`` / the HIP kernel, which also floors nothing:
      the *scale* is unchanged (both 0.0). The *bytes* change from ``0x7e`` to
      ``0x00`` -- measured, not inferred: ``ops.scaled_fp8_quant`` on an
      all-zero input returns scale 0.0 and **448.0 in every element**, because
      ``0 * (1/0)`` is NaN and ``fminf(NaN, 448)`` returns 448. It dequantizes
      to 0.0 through a multiply, so it is absorbed today, but nothing says so
      and any consumer that clamps or floors the scale would surface it.
    * against ``_int8_quantize``, which floors at ``1e-10``: no change at all.
      Measured: ``ops.scaled_int8_quant`` on zeros emits 0 with the scale at
      1e-10 *and* with the scale at 0.0. The floor is not doing what its
      presence suggests; it only changes the returned scale value.
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
    assert scale is None or amax is None, "give a static scale or an amax, not both"

    spec = _spec(quant_dtype, granularity)
    q = torch.empty_like(x, dtype=quant_dtype) if out is None else out

    if granularity == PER_TOKEN:
        n_scales = E * T
    elif granularity == PER_EXPERT:
        n_scales = E
    else:
        n_scales = 1

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
    use_partials = granularity == PER_EXPERT and scale is None and amax is None
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

    if scale is not None:
        # Static. No reduction at all, so nothing to bound; the mask still
        # decides which rows get written.
        s_flat = scale.reshape(-1).to(torch.float32)
        assert s_flat.numel() == n_scales, (s_flat.numel(), n_scales)
        s_flat = s_flat.contiguous()
    elif amax is not None:
        s_flat = torch.zeros(n_scales, dtype=torch.float32, device=x.device)
        a_flat = amax.reshape(-1)
        assert a_flat.numel() % n_scales == 0, (a_flat.numel(), n_scales)
        _scale_from_amax_kernel[(1,)](
            a_flat, s_flat, n_scales, a_flat.numel() // n_scales,
            spec.qmax, spec.floor, BLOCK=_next_pow2(n_scales),
            BLOCK_R=min(256, _next_pow2(a_flat.numel() // n_scales)),
        )  # fmt: skip
    else:
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
            T, K, spec.qmax, spec.floor, sa[0], sa[1], sa[2], **common,
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
    if granularity == PER_TOKEN:
        return q, s_flat.view(E, T, 1)
    return q, s_flat
