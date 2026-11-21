# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's fused DSpark draft head (`--monokernel`, 1-2 requests): when it
declines, and that it leaves every other step to the vocab-sharded chain. The
launch itself is tested against the chain on 8 GPUs in
tests/distributed/test_muse_spark_fused_draft_head.py."""

from types import SimpleNamespace

import pytest
import torch

import vllm.distributed

SHARD, HIDDEN, RANK = 576, 512, 16


def _head(
    monkeypatch,
    *,
    monokernel: bool = True,
    greedy: bool = False,
    world_size: int = 8,
    hidden: int = HIDDEN,
    rank: int = RANK,
    head_dtype: torch.dtype = torch.bfloat16,
    w2_dtype: torch.dtype = torch.bfloat16,
    logits_dtype: torch.dtype | None = None,
):
    from vllm.models.muse_spark.amd.ops.fused_draft_head import (
        MuseSparkFusedDraftHead,
    )

    monkeypatch.setattr(
        vllm.distributed,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=world_size, rank_in_group=0),
    )
    model = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(monokernel=monokernel)
        ),
        lm_head=SimpleNamespace(weight=torch.zeros(SHARD, hidden, dtype=head_dtype)),
        logits_processor=SimpleNamespace(head_dtype=logits_dtype),
        output_soft_cap_temp=20.0,
        output_multiplier=1.0,
    )
    shard = SimpleNamespace(
        greedy=greedy,
        w1=torch.zeros(8 * SHARD, rank, dtype=torch.bfloat16),
        w2_shard=SimpleNamespace(weight=torch.zeros(SHARD, rank, dtype=w2_dtype)),
        shard_size=SHARD,
        vocab_start=0,
        vocab_size=8 * SHARD,
    )
    return MuseSparkFusedDraftHead(model, shard)


def _spec(**overrides):
    spec = dict(
        num_speculative_steps=7, use_fp64_gumbel=False, use_confidence_head=False
    )
    return SimpleNamespace(**(spec | overrides))


@pytest.mark.parametrize(
    "head_kwargs, spec_kwargs, reason",
    [
        ({"monokernel": False}, {}, "not --monokernel"),
        ({"greedy": True}, {}, "greedy drafts"),
        ({}, {"num_speculative_steps": 9}, "more than 8"),
        ({"logits_dtype": torch.float32}, {}, "fp32 lm_head"),
        ({"world_size": 4}, {}, "tensor parallel 4"),
        ({"hidden": HIDDEN + 128}, {}, "hidden size"),
        ({"rank": 8}, {}, "Markov rank"),
        ({"head_dtype": torch.float8_e4m3fn}, {}, "not bf16"),
        ({"w2_dtype": torch.float8_e4m3fn}, {}, "not bf16"),
    ],
)
def test_fused_draft_head_declines(monkeypatch, head_kwargs, spec_kwargs, reason):
    """Each unsupported deployment turns the launch off, with the reason logged."""
    head = _head(monkeypatch, **head_kwargs)
    got = head._setup(_spec(**spec_kwargs))
    assert got != "on" and reason in got


def test_declined_head_leaves_every_step_to_the_chain(monkeypatch):
    """The first call (the warm-up step) sets up and leaves the step to the
    chain; a declined head leaves every later step to it too."""
    head = _head(monkeypatch, monokernel=False)
    stage = torch.zeros(2, 7, SHARD)
    for num_reqs in (1, 2, 1):
        assert not head.sample(_spec(), num_reqs, torch.zeros(1), stage)
    assert head._state == "not --monokernel"


@pytest.mark.parametrize(
    "num_reqs, stage, spec_kwargs",
    [
        (3, True, {}),
        (16, True, {}),
        (1, False, {}),
        (1, True, {"use_fp64_gumbel": True}),
        (2, True, {"use_confidence_head": True}),
    ],
    ids=["3-requests", "16-requests", "greedy", "fp64", "confidence-head"],
)
def test_head_leaves_other_steps_to_the_chain(
    monkeypatch, num_reqs, stage, spec_kwargs
):
    """An enabled head takes only probabilistic steps of 1-2 requests with fp32
    Gumbel noise and no confidence head."""
    head = _head(monkeypatch)
    head._state = "on"
    staged = torch.zeros(num_reqs, 7, SHARD) if stage else None
    assert not head.sample(_spec(**spec_kwargs), num_reqs, torch.zeros(1), staged)
