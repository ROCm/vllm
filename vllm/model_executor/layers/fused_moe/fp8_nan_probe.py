# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMPORARY investigation scaffolding. Delete before any upstream PR.

Everything here is gated behind ``VLLM_FP8_NAN_*`` environment variables and is
inert unless one is set, so several agents can share one checkout and one build
while probing different hypotheses about two open defects on the batched fp8
MoE path:

  1. A NaN under cudagraphs with a dynamic per-tensor activation scheme.
  2. Mode-off cudagraph capture not being run-to-run repeatable, which the
     per-expert-scale fix appears to make worse (11/64 and 9/64 sequences
     bit-exact, against 0/64 and 0/64).

Removal is one ``git rm`` of this file plus the ``fp8_nan_probe`` grep in
``fused_batched_moe.py``.

The variables
-------------

``VLLM_FP8_NAN_PATH`` selects how the batched activation scale is computed:

``stock``
    Untouched upstream behaviour. One dynamic scale for the whole
    ``[E, max_rows, K]`` buffer under capture, via ``ops.scaled_fp8_quant``;
    a per-expert loop in eager. **The two differ**, which is the confound this
    module exists to remove.
``perexpert``
    The pending ``patch_maskedamax.diff``: a masked per-expert amax under
    capture, matching eager's granularity. Still two code paths.
``triton``
    One Triton implementation used in **both** eager and capture, reading
    ``expert_num_tokens`` on the device so no host sync is needed and no
    ``_is_capturing_or_compiling()`` branch exists. This is the arm that can
    answer whether the eager/capture repeatability gap survives when the two
    modes genuinely run the same code.

``VLLM_FP8_NAN_SCALE_FLOOR`` (default ``1e-10``) is the lower clamp on the
computed scale. **Setting it to 0 is a live experiment, not a cleanup.**
``ops.scaled_fp8_quant``'s dynamic per-tensor path memsets its accumulator to 0
and reduces with ``fmaxf``, which discards NaN, so a tile that is entirely zero
-- or entirely NaN -- yields a scale of exactly ``0.0`` and the quantize kernel
then computes ``0/0``. That is a candidate mechanism for defect 1, and this
floor is what would suppress it. The prediction to test: with the floor at 0,
the NaN returns even on the ``perexpert`` and ``triton`` paths.

``VLLM_FP8_NAN_PIN_SCALE`` (unset by default) replaces the computed scale with a
constant. Pinning it restored repeatability to 61/64, which is how the dynamic
activation scale was identified as the active ingredient in defect 2. Choose a
value above the largest observed amax or the output degenerates, and check the
output is real text before believing any repeatability number measured with it:
a model clipped to a constant is trivially repeatable.
"""

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

PATH_STOCK = "stock"
PATH_PER_EXPERT = "perexpert"
PATH_TRITON = "triton"
_VALID_PATHS = (PATH_STOCK, PATH_PER_EXPERT, PATH_TRITON)


def probe_path() -> str:
    path = os.environ.get("VLLM_FP8_NAN_PATH", PATH_STOCK).strip().lower()
    if path not in _VALID_PATHS:
        raise ValueError(
            f"VLLM_FP8_NAN_PATH={path!r} is not one of {_VALID_PATHS}. "
            "An unrecognised value silently selecting stock would make an arm "
            "look like a null result."
        )
    return path


def scale_floor() -> float:
    return float(os.environ.get("VLLM_FP8_NAN_SCALE_FLOOR", "1e-10"))


def pinned_scale() -> float | None:
    raw = os.environ.get("VLLM_FP8_NAN_PIN_SCALE", "").strip()
    return float(raw) if raw else None


def probe_active() -> bool:
    """True when any toggle is set, so the stock path can be left untouched."""
    return (
        probe_path() != PATH_STOCK
        or pinned_scale() is not None
        or ("VLLM_FP8_NAN_SCALE_FLOOR" in os.environ)
    )


def describe() -> str:
    return f"path={probe_path()} floor={scale_floor():g} pin={pinned_scale()}"


def _apply_floor_and_pin(scale: torch.Tensor) -> torch.Tensor:
    pin = pinned_scale()
    if pin is not None:
        return torch.full_like(scale, pin)
    floor = scale_floor()
    return scale.clamp(min=floor) if floor > 0.0 else scale


# --------------------------------------------------------------------------
# The `perexpert` path: torch ops, mode-agnostic, one scale per expert over the
# rows that expert actually received.
# --------------------------------------------------------------------------


def batched_delivered_rows_mask(
    x: torch.Tensor, expert_num_tokens: torch.Tensor
) -> torch.Tensor:
    """Mark the rows of a batched ``[E, max_rows, ...]`` buffer holding a token.

    Only the first ``expert_num_tokens[e]`` rows of each expert were written by
    whatever produced ``x``; the rest are whatever the workspace last held.
    Device-side throughout, so this is safe inside a cudagraph capture and under
    ``torch.compile``.
    """
    num_experts, max_rows = x.size(0), x.size(1)
    rows = torch.arange(max_rows, device=x.device).view(1, max_rows)
    return rows < expert_num_tokens.view(num_experts, 1).to(rows.dtype)


def batched_dynamic_per_expert_scale(
    x: torch.Tensor, expert_num_tokens: torch.Tensor, quant_dtype: torch.dtype
) -> torch.Tensor:
    """One dynamic scale per expert, over the rows that expert wrote. ``[E,1,1]``.

    The reduction is per row first, so no buffer-sized temporary is
    materialized, and ``aminmax`` rather than a separate ``amax`` and ``amin``
    keeps it to a single pass.
    """
    row_min, row_max = torch.aminmax(x, dim=-1)
    row_amax = torch.maximum(row_max, row_min.neg())
    # `amax` propagates NaN where the quantization kernels' `fmaxf` reduction
    # drops it. Match the kernels, so a NaN already in the buffer cannot newly
    # poison every row its scale covers.
    row_amax = torch.nan_to_num(row_amax, nan=0.0)
    written = batched_delivered_rows_mask(x, expert_num_tokens)
    amax = torch.where(written, row_amax, torch.zeros_like(row_amax)).amax(dim=1)

    if quant_dtype.is_floating_point:
        qmax = torch.finfo(quant_dtype).max
    else:
        qmax = torch.iinfo(quant_dtype).max

    return _apply_floor_and_pin(amax.to(torch.float32) / qmax).view(-1, 1, 1)


# --------------------------------------------------------------------------
# The `triton` path: one implementation for eager and capture alike.
# --------------------------------------------------------------------------


@triton.jit
def _masked_amax_kernel(
    a_ptr,
    ent_ptr,
    amax_ptr,
    K,
    stride_ae,
    stride_am,
    stride_ak,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-expert amax over the first ``expert_num_tokens[e]`` rows.

    ``expert_num_tokens`` is read on the device, which is the whole point: the
    eager path's ``int(expert_num_tokens[e].item())`` is a host sync, and that
    sync is the only reason a separate capture path had to exist.

    The cross-block combine is ``atomic_max``. Max is associative, commutative
    and exact in fp32, so the result does not depend on which block arrives
    first -- unlike a sum, this is safe for batch invariance.
    """
    e = tl.program_id(0)
    m_block = tl.program_id(1)
    n_e = tl.load(ent_ptr + e)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < n_e

    acc = tl.zeros((), dtype=tl.float32)
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
        # Match `fmaxf`, which discards NaN, rather than `amax`, which
        # propagates it. A NaN already in the buffer must not newly poison the
        # scale for every row it covers.
        vals = tl.where(vals != vals, 0.0, vals)
        acc = tl.maximum(acc, tl.max(vals))

    tl.atomic_max(amax_ptr + e, acc)


@triton.jit
def _quantize_kernel(
    a_ptr,
    q_ptr,
    scale_ptr,
    ent_ptr,
    K,
    QMAX,
    stride_ae,
    stride_am,
    stride_ak,
    stride_qe,
    stride_qm,
    stride_qk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Quantize only the rows an expert received, with that expert's scale.

    Rows past ``expert_num_tokens[e]`` are left unwritten rather than zeroed.
    MM2 early-exits on ``cta_m_start >= e_num_tokens`` and masks its loads with
    ``mask_m``, so it never reads them; zeroing them would be dead work and, as
    the deleted ``fill_`` showed, invites a comment claiming the fill is what
    keeps them safe.
    """
    e = tl.program_id(0)
    m_block = tl.program_id(1)
    n_e = tl.load(ent_ptr + e)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < n_e
    inv_scale = 1.0 / tl.load(scale_ptr + e)

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
        q = tl.clamp(vals * inv_scale, -QMAX, QMAX)
        tl.store(
            q_ptr
            + e * stride_qe
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk,
            q.to(q_ptr.dtype.element_ty),
            mask=mask,
        )


def triton_batched_quantize(
    x: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One code path for eager and cudagraph capture alike.

    Returns ``(A_q, scale)`` with scale shaped ``[E, 1, 1]``, matching what the
    eager per-expert loop produces.
    """
    assert x.ndim == 3, x.shape
    E, max_rows, K = x.shape

    if quant_dtype.is_floating_point:
        qmax = torch.finfo(quant_dtype).max
    else:
        qmax = torch.iinfo(quant_dtype).max

    amax = torch.zeros(E, dtype=torch.float32, device=x.device)
    block_m = 32
    block_k = 256
    grid = (E, triton.cdiv(max_rows, block_m))

    _masked_amax_kernel[grid](
        x,
        expert_num_tokens,
        amax,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
    )

    scale = _apply_floor_and_pin(amax / qmax)

    x_q = torch.empty_like(x, dtype=quant_dtype)
    _quantize_kernel[grid](
        x,
        x_q,
        scale,
        expert_num_tokens,
        K,
        qmax,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x_q.stride(0),
        x_q.stride(1),
        x_q.stride(2),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
    )
    return x_q, scale.view(-1, 1, 1)


def handles(
    A_scale: torch.Tensor | None,
    qtype: torch.dtype | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None,
    A: torch.Tensor,
) -> bool:
    """Whether a probe path applies to this call.

    Deliberately narrow: dynamic per-tensor fp8 only, which is the scheme both
    defects live in. Every other scheme keeps stock behaviour so an arm cannot
    move something it was not meant to touch.
    """
    return (
        probe_path() != PATH_STOCK
        and A_scale is None
        and qtype == current_platform.fp8_dtype()
        and not per_act_token_quant
        and block_shape is None
        and A.numel() > 0
        and A.ndim == 3
    )
