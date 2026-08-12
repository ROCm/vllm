# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EPLB gate in `VllmConfig.__post_init__`, checked without a GPU.

`num_redundant_experts > 0` gives a logical expert more than one physical
replica, and `_eplb_map_and_record_i32_kernel` picks between them by hashing
`token_idx = offs // num_active_experts` -- the token's row index in the
current forward. Which replica, and therefore which rank, serves a token then
depends on what else was batched with it, so the mode refuses the
configuration. Plain EPLB keeps the same routing for every token and is
admitted with a warning.

Note what is *not* refused, because it is the distinction the gate turns on.
EPLB's load-driven rearrangement makes a request's output depend on the
traffic that preceded it -- but `eplb_step()` runs after the forward pass, so
every token in a step sees the same placement. That is a **temporal**
dependence, and batch invariance does not forbid it; it forbids a token's
output depending on its *batch-mates*. The replica hash is the only part of
EPLB that does that, and it is the only part refused here.

The measurement behind the refusal, which has no test of its own: DeepSeek-V2-
Lite, DP=4 x EP=4, `num_redundant_experts=8`, expert placement held still with
zero committed rearrangements -- 64 of 64 needle logprobs moved when 32
companions shared its rank, and 0 of 64 when 24 companions ran on the other
three ranks instead. Only companions that change the needle's own row index
move it, which is the signature of the hash rather than of the collective. The
same sweep at `num_redundant_experts=0` moved 0 of 64 in every condition.

This module is deliberately cheap: it constructs the config objects directly,
costs milliseconds, and needs neither a model nor a device, so it runs in a
single-GPU CI job.

The warning on plain EPLB is deliberately not asserted: `warning_once` caches
per process, so whether it is emitted depends on what ran before it in the
same pytest session. What is asserted is the part that matters operationally
-- that plain EPLB is *admitted*, and that the mode does not quietly rewrite
`num_redundant_experts` instead of refusing.
"""

import pytest
from utils import skip_if_not_cuda_alike

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.parallel import EPLBConfig, ParallelConfig

# `ParallelConfig` refuses EPLB outright off CUDA/ROCm, so the gate under test
# is unreachable elsewhere and there is nothing to check.
pytestmark = [skip_if_not_cuda_alike]

REFUSAL = "EPLB with redundant experts is not supported"

# EPLB needs an EP group wider than one rank; DP=4 is what the end-to-end
# module measures.
DP = 4


def _config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch_invariant: bool,
    num_redundant_experts: int,
    enable_eplb: bool = True,
) -> VllmConfig:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    return VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=DP,
            enable_expert_parallel=True,
            enable_eplb=enable_eplb,
            eplb_config=EPLBConfig(num_redundant_experts=num_redundant_experts),
        )
    )


@pytest.mark.parametrize("redundant", [1, 8])
def test_redundant_experts_are_refused(monkeypatch, redundant):
    """The replica hash reads the batch, so the mode must not start at all."""
    with pytest.raises(ValueError, match=REFUSAL):
        _config(monkeypatch, batch_invariant=True, num_redundant_experts=redundant)


@pytest.mark.parametrize("redundant", [1, 8])
def test_redundant_experts_are_admitted_with_the_mode_off(monkeypatch, redundant):
    """The refusal is the mode's, not EPLB's: it must not leak into normal use."""
    config = _config(
        monkeypatch, batch_invariant=False, num_redundant_experts=redundant
    )
    assert config.parallel_config.eplb_config.num_redundant_experts == redundant


def test_plain_eplb_is_admitted(monkeypatch):
    """Plain EPLB is invariant to batch composition and is only warned about.

    Asserting the surviving `num_redundant_experts` matters as much as the
    absence of a raise: a gate that clamped the value to 0 would let a
    deployment that asked for replicas run silently without them.
    """
    config = _config(monkeypatch, batch_invariant=True, num_redundant_experts=0)
    assert config.parallel_config.enable_eplb
    assert config.parallel_config.eplb_config.num_redundant_experts == 0
