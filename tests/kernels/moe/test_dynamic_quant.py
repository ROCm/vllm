# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`dynamic_quantize`'s routed-row mask: only routable rows set the scale.

A per-tensor scale is one amax over the whole buffer, so a single row the mask
wrongly admits sets the scale for every other row. `MASK_ROUTED` reads the
expert id from device memory, which is why the id is bounded at both ends
before it indexes `expert_map` -- the same guard the pad-aware activation
kernels carry.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe import dynamic_quant
from vllm.platforms import current_platform

NUM_EXPERTS = 16
LOCAL_EXPERTS = 8
HIDDEN = 128


def _expert_map(device):
    """Global -> local, backed by a larger buffer whose tail reads as local.

    The padding is what gives this test power: an out-of-bounds read lands on
    a non-negative value, so an unbounded kernel admits the row instead of
    being rescued by whatever the allocator happened to leave there.
    """
    backing = torch.zeros(NUM_EXPERTS * 4, dtype=torch.int32, device=device)
    backing[:NUM_EXPERTS] = -1
    m = backing[:NUM_EXPERTS]
    m[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return m


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
@pytest.mark.parametrize("bad", [LOCAL_EXPERTS, NUM_EXPERTS, NUM_EXPERTS + 5, 1 << 20])
def test_unroutable_row_cannot_set_the_scale(bad):
    """A non-local or out-of-range id must be excluded exactly as -1 is.

    Baseline and arm differ only in the id on row 1, whose amax is three orders
    of magnitude above every other row, so admitting it is unmissable.
    """
    device = "cuda"
    torch.manual_seed(0)
    n = 128
    x = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device=device)
    x[1] = 1e4
    ids = torch.randint(0, LOCAL_EXPERTS, (n,), dtype=torch.int32, device=device)
    emap = _expert_map(device)

    def scale_for(expert_id):
        ids[1] = expert_id
        return dynamic_quant.dynamic_quantize(
            x,
            current_platform.fp8_dtype(),
            mask_mode=dynamic_quant.MASK_ROUTED,
            topk_ids=ids,
            expert_map=emap,
        )[1]

    unrouted = scale_for(-1)
    # Sync explicitly: an out-of-bounds address faults asynchronously.
    torch.accelerator.synchronize()
    # Positive control: the row must be able to move the scale when it is
    # routed, or an agreement with `-1` says nothing about exclusion.
    assert not torch.equal(unrouted, scale_for(0))
    assert torch.equal(unrouted, scale_for(bad))
