# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's fused DSpark draft head on 8 ranks against the vocab-sharded
chain it replaces (`MarkovVocabShard` without the head).

The weights and hidden states are small multiples of powers of two, so the
lm_head and bias GEMMs are exact in fp32 whatever their accumulation order:
the drafted tokens must be equal on every rank, and the staged logits within
one bf16 ulp (the fp32 tanh)."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

from ..utils import multi_gpu_test
from .eplb_utils import distributed_run, set_env_vars_and_device

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Muse Spark ROCm draft path"
)

WORLD = 8
HIDDEN = 512


class _ShardedLogits:
    """The draft logits processor on this rank's rows of an exact lm_head."""

    scale, soft_cap, head_dtype = 1.0, None, None

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def __call__(self, lm_head, hidden, skip_gather=False):
        assert skip_gather
        return hidden.to(torch.bfloat16) @ self.weight.t()

    def _apply_head(self, head, x, bias):
        return head.quant_method.apply(head, x, bias=bias)


def _exact(shape, k: int, scale: float, generator: torch.Generator) -> torch.Tensor:
    x = torch.randint(-k, k + 1, shape, generator=generator, device="cuda")
    return (x.float() * scale).to(torch.bfloat16)


def _draft(rank: int, soft_cap: float | None):
    """The speculator, the shard's chain and the fused head of this rank."""
    from tests.kernels.models.amd.test_muse_spark_fused_markov import (
        MAX_REQS,
        N_SPEC,
        VOCAB,
        _speculator,
    )
    from vllm.models.muse_spark.amd.ops.fused_draft_head import (
        MuseSparkFusedDraftHead,
    )
    from vllm.models.muse_spark.amd.ops.markov_vocab_shard import MarkovVocabShard

    spec = _speculator(1.0, seed=WORLD, exact=True)
    spec.max_num_reqs, spec.device = MAX_REQS, torch.device("cuda")
    # The same lm_head and hidden states on every rank (one seed).
    g = torch.Generator(device="cuda").manual_seed(1)
    lm_head = _exact((VOCAB, HIDDEN), 8, 2.0**-4, g)
    hidden = _exact((MAX_REQS * N_SPEC, HIDDEN), 8, 0.5, g)
    shard = VOCAB // WORLD
    start, end = rank * shard, (rank + 1) * shard
    local = lm_head[start:end].contiguous()
    model = spec.model
    model.output_soft_cap_temp = soft_cap
    model.config = SimpleNamespace(draft_vocab_size=VOCAB)
    model.lm_head = SimpleNamespace(
        weight=local,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start, org_vocab_end_index=end
        ),
    )
    model.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_sample_method="probabilistic"),
        kernel_config=SimpleNamespace(monokernel=True),
    )
    model.logits_processor = _ShardedLogits(local)
    chain = MarkovVocabShard(model, None)
    return spec, chain, MuseSparkFusedDraftHead(model, chain), hidden


def _new_state(spec, generator: torch.Generator) -> None:
    """New seeds and anchor tokens, the same on every rank."""
    from tests.kernels.models.amd.test_muse_spark_fused_markov import VOCAB

    spec.seeds.copy_(
        torch.randint(0, 2**62, spec.seeds.shape, generator=generator, device="cuda")
    )
    ids = spec.input_buffers.input_ids
    ids.copy_(torch.randint(0, VOCAB, ids.shape, generator=generator, device="cuda"))


def _drafted(spec, chain, num_reqs: int, run) -> tuple:
    """(tokens, staged logits, cache rows) of one sampling call."""
    spec.draft_tokens.fill_(-1)
    spec.draft_logits.values.zero_()
    chain.stage.zero_()
    run()
    torch.accelerator.synchronize()
    return (
        spec.draft_tokens[:num_reqs].clone(),
        chain.stage[:num_reqs].clone(),
        spec.draft_logits.values.clone(),
    )


def _assert_within_one_ulp(got: torch.Tensor, ref: torch.Tensor) -> None:
    def ordered(x: torch.Tensor) -> torch.Tensor:
        bits = x.view(torch.int16).long()
        return torch.where(bits < 0, -32768 - bits, bits)

    assert (ordered(got) - ordered(ref)).abs().max() <= 1


def _check(got, ref) -> None:
    (tokens, stage, cache), (ref_tokens, ref_stage, ref_cache) = got, ref
    assert torch.equal(tokens, ref_tokens)
    _assert_within_one_ulp(stage, ref_stage)
    _assert_within_one_ulp(cache, ref_cache)
    parts = [torch.empty_like(tokens) for _ in range(WORLD)]
    torch.distributed.all_gather(parts, tokens)
    assert all(torch.equal(tokens, p) for p in parts)


def fused_draft_head_matches_the_chain(
    env: dict[str, str], world_size: int, soft_cap: float | None
) -> None:
    set_env_vars_and_device(env)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized

    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(world_size, 1)
    rank = int(env["RANK"])
    spec, chain, head, hidden = _draft(rank, soft_cap)
    g = torch.Generator(device="cuda").manual_seed(2)

    def run(num_reqs: int, fused: bool):
        chain.head = head if fused else None
        assert chain.sample_sequential(spec, num_reqs, hidden)

    with torch.inference_mode():
        # The first call sets the head up (collective) and runs the chain.
        run(1, fused=True)
        assert head._state == "on", head._state
        for temp in (0.0, 0.7, 1.0):
            spec.temperature.fill_(temp)
            for num_reqs in (1, 2):
                for _ in range(2):
                    _new_state(spec, g)
                    ref = _drafted(spec, chain, num_reqs, partial(run, num_reqs, False))
                    got = _drafted(spec, chain, num_reqs, partial(run, num_reqs, True))
                    _check(got, ref)

        # One captured launch replayed with new state: the epoch tags need no reset.
        spec.temperature.fill_(1.0)
        stage = chain.stage[:2]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert head.sample(spec, 2, hidden, stage)
        for _ in range(3):
            _new_state(spec, g)
            ref = _drafted(spec, chain, 2, partial(run, 2, False))
            spec.draft_tokens.fill_(-1)
            stage.zero_()
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(spec.draft_tokens[:2], ref[0])
            _assert_within_one_ulp(stage, ref[1])

    assert not head.health.any()
    torch.distributed.barrier()
    head.peers.close()


@multi_gpu_test(num_gpus=WORLD)
@pytest.mark.parametrize("soft_cap", [20.0, None])
def test_muse_spark_fused_draft_head_matches_the_chain(tmp_path, monkeypatch, soft_cap):
    """The fused draft head draws the vocab-sharded chain's tokens on every
    rank and stages its logits, at 1 and 2 requests, greedy and sampled rows,
    eager and replayed from a CUDA graph."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the fused draft head runs on gfx950")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    distributed_run(fused_draft_head_matches_the_chain, WORLD, soft_cap)
