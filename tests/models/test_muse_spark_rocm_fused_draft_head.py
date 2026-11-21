# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A fused DSpark draft head that declines its deployment never launches. The
launch and the steps it leaves to the vocab-sharded chain are tested against
the chain on 8 GPUs in tests/distributed/test_muse_spark_fused_draft_head.py."""

from types import SimpleNamespace

import torch


def test_declined_head_never_launches(monkeypatch):
    from vllm.models.muse_spark.amd.ops import fused_draft_head

    class _NoLaunch:
        def __getitem__(self, grid):
            raise AssertionError("launched")

    monkeypatch.setattr(fused_draft_head, "_draft_head_kernel", _NoLaunch())
    model = SimpleNamespace(
        vllm_config=SimpleNamespace(kernel_config=SimpleNamespace(monokernel=False))
    )
    head = fused_draft_head.MuseSparkFusedDraftHead(model, SimpleNamespace())
    spec = SimpleNamespace(
        num_speculative_steps=7, use_fp64_gumbel=False, use_confidence_head=False
    )
    stage = torch.zeros(2, 7, 128)
    for num_reqs in (1, 2, 1):
        assert not head.sample(spec, num_reqs, torch.zeros(1), stage)
