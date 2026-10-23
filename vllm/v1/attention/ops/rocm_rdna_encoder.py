# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_RDNA_ATTN for MMEncoderAttention (gfx1151): an encoder's segments,
each token seeing every key of its segment, in one launch of the non-causal
prefill builds, which read Q, K and V in place.
"""

import functools
from typing import NamedTuple

import numpy as np
import torch

from vllm.v1.attention.ops.rocm_rdna_hip import (
    PREFILL_MIN_M,
    SUPPORTED_HEAD_SIZES,
    PrefillVariant,
    load_prefill,
    make_prefill_scratch,
    plan_prefill,
    prefill_batch_variant_for,
    prefill_target_workgroups,
    prefill_variant_for,
    prefill_variants,
)

PAGE_SIZE = 16
_HND = 1
# Per (shape, device): prefill scratch, sized for every non-causal build.
_PREFILL_SCRATCH: dict = {}


def unsupported(q: torch.Tensor, num_kv_heads: int, seg_lens: list[int]) -> str | None:
    """Why the kernels cannot serve this call, or None."""
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {q.dtype}"
    hq, d = q.shape[1], q.shape[2]
    if d not in SUPPORTED_HEAD_SIZES:
        return f"head size {d}"
    if hq % num_kv_heads:
        return "q heads must divide evenly over kv heads"
    if torch.cuda.is_current_stream_capturing():
        return "CUDA graph capture (the launch follows the host)"
    if not prefill_variants(hq, num_kv_heads, d, PAGE_SIZE, 0, q.dtype, False):
        return f"no non-causal build for {hq}/{num_kv_heads}/{d} {q.dtype}"
    return None


class _Layout(NamedTuple):
    """How one list of segment lengths runs, on the device."""

    variant: PrefillVariant
    seq_lens: torch.Tensor
    # The plan and query starts; None for one segment.
    plan: torch.Tensor | None
    query_starts: torch.Tensor | None
    head_group: int
    longest: int


@functools.lru_cache(maxsize=16)
def _layout(
    seg_lens: tuple[int, ...],
    hq: int,
    hkv: int,
    d: int,
    dtype: torch.dtype,
    dev: torch.device,
) -> _Layout:
    def to_dev(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev, dtype=torch.int32)

    lens = np.asarray(seg_lens, dtype=np.int64)
    shape = (hq, hkv, d, PAGE_SIZE)
    if len(lens) == 1:
        # The first row also serves fewer than PREFILL_MIN_M tokens.
        variant = prefill_variant_for(
            *shape, _HND, 0, dtype, max(int(lens[0]), PREFILL_MIN_M), causal=False
        )
        assert variant is not None, "unsupported() checks the builds"
        return _Layout(variant, to_dev(lens), None, None, 0, int(lens[0]))
    starts = np.concatenate([[0], np.cumsum(lens)])
    variant, head_group = prefill_batch_variant_for(
        *shape, _HND, 0, dtype, starts, lens, causal=False
    )
    assert variant is not None, "unsupported() checks the builds"
    plan, head_group = plan_prefill(
        starts, lens, variant, 0, prefill_target_workgroups(dev), head_group
    )
    return _Layout(
        variant, to_dev(lens), to_dev(plan), to_dev(starts), head_group,
        int(lens.max()),
    )  # fmt: skip


def _in_place(x: torch.Tensor) -> torch.Tensor:
    """x, or a copy with each token's heads contiguous."""
    return x if x.stride(2) == 1 and x.stride(1) == x.shape[2] else x.contiguous()


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seg_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Attention of each segment's tokens over its keys, no mask.

    Args:
        q: (tokens, Hq, D), the segments one after the other.
        k: (tokens, Hkv, D).
        v: (tokens, Hkv, D).
        seg_lens: Each segment's tokens, on the host.
        scale: Softmax scale.

    Returns:
        (tokens, Hq, D). `unsupported` must have returned None.

    """
    dev, dtype = q.device, q.dtype
    hq, d, hkv = q.shape[1], q.shape[2], k.shape[1]
    lay = _layout(tuple(seg_lens), hq, hkv, d, dtype, dev)
    q, k, v = _in_place(q), _in_place(k), _in_place(v)
    out = torch.empty(q.shape, dtype=dtype, device=dev)
    prefill = load_prefill(lay.variant)
    assert prefill is not None
    shape = (hq, hkv, d, PAGE_SIZE, 0, dtype)
    scratch = _PREFILL_SCRATCH.get((shape, dev))
    if scratch is None or scratch[1].numel() < 2 * lay.variant.scratch_rows(dev):
        scratch = make_prefill_scratch(
            [*prefill_variants(*shape, causal=False), lay.variant], dev
        )
        _PREFILL_SCRATCH[(shape, dev)] = scratch
    if lay.plan is None or lay.plan.shape[0]:
        prefill.encoder_attn(
            q, k, v, out, *scratch, lay.seq_lens, lay.longest, scale,
            plan=lay.plan, query_start_loc=lay.query_starts,
            head_group=lay.head_group,
        )  # fmt: skip
    return out
