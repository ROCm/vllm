# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pad-aware activation kernels: equality on live slots, safety on dead ones.

These two kernels skip the token-expert slots no expert GEMM will read. They
had no coverage at all, and the gap mattered: `_silu_and_mul_pad_aware_kernel`
indexed `expert_map` with `mask=expert_id >= 0`, bounding only the low end, so
a corrupt or out-of-range id read past the end of the tensor and faulted the
device.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.utils import (
    silu_and_mul_is_pad_aware,
    silu_and_mul_pad_aware,
    swiglu_limit_func,
)
from vllm.platforms import current_platform

NUM_EXPERTS = 32
LOCAL_EXPERTS = 8
HIDDEN = 512


def _expert_map(device):
    """Global -> local, with only the first LOCAL_EXPERTS resident here."""
    m = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
    m[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return m


def _reference(input_, topk_ids_flat, expert_map):
    """Dense silu_and_mul, then zero the slots the kernel is entitled to skip.

    Uses the CUDA kernel rather than torch float math: the pad-aware kernel
    documents bit-exactness against `torch.ops._C.silu_and_mul` specifically,
    which rounds silu to the storage dtype before the multiply and maps a -0.0
    `up` to +0.0. Reimplementing that in fp32 does not reproduce it.
    """
    d = input_.shape[1] // 2
    out = torch.empty(input_.shape[0], d, dtype=input_.dtype, device=input_.device)
    torch.ops._C.silu_and_mul(out, input_)
    ids = topk_ids_flat
    live = ids >= 0
    in_range = live & (ids < expert_map.numel())
    local = torch.where(
        in_range, expert_map[ids.clamp(0, expert_map.numel() - 1).long()], -1
    )
    out[~(in_range & (local >= 0))] = 0
    return out


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
@pytest.mark.parametrize("num_rows", [1, 7, 64, 513])
def test_matches_dense_on_live_slots(num_rows):
    """Computed slots must be bit-identical to the dense kernel."""
    device = "cuda"
    torch.manual_seed(num_rows)
    inp = torch.randn(num_rows, 2 * HIDDEN, dtype=torch.bfloat16, device=device)
    assert silu_and_mul_is_pad_aware(inp)
    ids = torch.randint(0, LOCAL_EXPERTS, (num_rows,), dtype=torch.int64, device=device)
    emap = _expert_map(device)

    out = torch.zeros(num_rows, HIDDEN, dtype=torch.bfloat16, device=device)
    silu_and_mul_pad_aware(out, inp, ids, emap)
    torch.testing.assert_close(out, _reference(inp, ids, emap), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
def test_skips_negative_and_non_local():
    """Dead slots are left untouched, not garbage."""
    device = "cuda"
    torch.manual_seed(0)
    n = 128
    inp = torch.randn(n, 2 * HIDDEN, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, NUM_EXPERTS, (n,), dtype=torch.int64, device=device)
    ids[::4] = -1  # unrouted
    emap = _expert_map(device)

    out = torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=device)
    silu_and_mul_pad_aware(out, inp, ids, emap)
    torch.testing.assert_close(out, _reference(inp, ids, emap), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
@pytest.mark.parametrize("bad", [NUM_EXPERTS, NUM_EXPERTS + 1, 1 << 20, 1 << 40])
def test_out_of_range_expert_id_is_skipped_not_dereferenced(bad):
    """An id past the end of `expert_map` must be treated as non-local.

    This is the regression guard. Before the fix the kernel masked only
    `expert_id >= 0`, so `expert_map_ptr + expert_id` was issued for any large
    positive id and faulted the device. `expert_map` has one entry per global
    expert, so an id outside [0, num_experts) is not routable here by
    definition and the slot must simply be skipped.
    """
    device = "cuda"
    torch.manual_seed(1)
    n = 64
    inp = torch.randn(n, 2 * HIDDEN, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, LOCAL_EXPERTS, (n,), dtype=torch.int64, device=device)
    ids[n // 2] = bad
    emap = _expert_map(device)

    out = torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=device)
    silu_and_mul_pad_aware(out, inp, ids, emap)
    # Sync explicitly: a bad address faults asynchronously, so without this the
    # assertion below could pass on a queue that has not run yet.
    torch.accelerator.synchronize()

    assert torch.count_nonzero(out[n // 2]) == 0, "out-of-range slot was computed"
    torch.testing.assert_close(out, _reference(inp, ids, emap), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
@pytest.mark.parametrize("bad", [NUM_EXPERTS, 1 << 30])
def test_swiglu_limit_out_of_range_expert_id(bad):
    """Same guard in the swiglu-limit twin, which shares the defect."""
    device = "cuda"
    torch.manual_seed(2)
    n = 64
    inp = torch.randn(n, 2 * HIDDEN, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, LOCAL_EXPERTS, (n,), dtype=torch.int64, device=device)
    ids[0] = bad
    emap = _expert_map(device)

    out = torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=device)
    swiglu_limit_func(out, inp, 7.0, topk_ids=ids, expert_map=emap)
    torch.accelerator.synchronize()

    assert torch.count_nonzero(out[0]) == 0, "out-of-range slot was computed"
