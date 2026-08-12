# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline parallelism composed with tensor parallelism, PP=2 x TP=4.

Each half is already covered on its own: `test_pp_batch_invariant` for the
process layout and the scheduler at PP=2/TP=1, and
`test_tp_all_reduce_batch_invariant` for the collective at world size 4 and 8.
What neither covers is the composition, and the composition is not obviously
free: PP keeps `pipeline_parallel_size` batches in flight, so the token counts
reaching the TP all-reduce are drawn from a different distribution than at PP=1,
and a collective whose reduction order moves with the message size would show up
here and nowhere else.

TP must be 4. A 2-rank sum is order independent, so PP=2 x TP=2 would pass even
against a fully batch-variant collective.

In this configuration `ca_comm` is live on every rank, sized from
`max_num_batched_tokens`, so the custom kernel serves every all-reduce and the
all-gather fallback is left to `test_tp_all_reduce_batch_invariant`'s
`fallback_ar` worker. PP=2 really does shift the token counts arriving at the
collective relative to PP=1 -- over one identical batch the two sets of
observed sizes share a single element -- and the custom kernel's size
independence is a measurement rather than a structural guarantee, which is why
the needle is run at this layout instead of inferred from the PP=1 and TP-only
results.

Not covered here: PP x TP x DP, PP over more than two stages, TP over more than
four ranks under PP, and MLA/MoE checkpoints (the JoyAI MXFP8 checkpoint used
elsewhere in this suite is TP=1 only).
"""

import os
import warnings

from utils import (
    assert_needle_is_batch_invariant,
    shutdown_llm,
    skip_if_not_cuda_alike,
)

from tests.utils import multi_gpu_marks
from vllm import LLM, SamplingParams

# multi_gpu_test would also wrap the test in create_new_process_for_each_test,
# whose re-import would break the worker extension lookup below, so take its
# marks alone: `-m distributed` still selects the module and the skipif still
# enforces the eight GPUs PP=2 x TP=4 needs.
pytestmark = [skip_if_not_cuda_alike, *multi_gpu_marks(num_gpus=8)]

MODEL = os.getenv("VLLM_PP_TP_TEST_MODEL", "Qwen/Qwen3-1.7B")
PP_SIZE = 2
TP_SIZE = 4
MAX_BATCH_SIZE = int(os.getenv("VLLM_PP_TP_NEEDLE_BATCH_SIZE", "16"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("VLLM_PP_TP_MAX_NUM_BATCHED_TOKENS", "2048"))
# 800 repeats is 7208 tokens, four chunks at the resulting 2032-token cap.
_PROMPT_PADDING = "Some background context for the question that follows. "
_PADDING_REPEATS = int(os.getenv("VLLM_PP_TP_PROMPT_REPEATS", "800"))


class AllReduceProbe:
    """Worker extension: report the TP all-reduce path taken on this rank."""

    # Supplied by the Worker this class is mixed into.
    rank: int

    def probe_all_reduce_path(self) -> dict:
        import vllm.envs as envs
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from vllm.model_executor.layers import batch_invariant as bi

        communicator = get_tp_group().device_communicator
        ca_comm = communicator.ca_comm
        return {
            "rank": self.rank,
            "pp_rank": get_pp_group().rank_in_group,
            "tp_rank": get_tp_group().rank_in_group,
            "envs_batch_invariant": bool(envs.VLLM_BATCH_INVARIANT),
            "mode_enabled": bi._batch_invariant_MODE,
            "library_installed": bi._batch_invariant_LIB is not None,
            "ca_comm_live": ca_comm is not None and not ca_comm.disabled,
            "ca_max_size": None if ca_comm is None else ca_comm.max_size,
            # Any of these being live would mean a reduction whose order the
            # mode does not pin; `all_reduce` skips them, and this records that
            # the skip is what is actually happening rather than a no-op.
            "other_ar_backends_live": [
                name
                for name in ("qr_comm", "aiter_ar_comm", "fi_ar_comm", "symm_mem_comm")
                if (other := getattr(communicator, name, None)) is not None
                and not getattr(other, "disabled", False)
            ],
        }

    def install_all_reduce_counter(self) -> None:
        """Record, per call, the token count and which implementation served it."""
        from collections import Counter

        from vllm.distributed.parallel_state import get_tp_group

        communicator = get_tp_group().device_communicator
        if getattr(communicator, "_bi_ar_counts", None) is not None:
            return
        counts: Counter = Counter()
        communicator._bi_ar_counts = counts
        original = communicator.all_reduce

        def counting_all_reduce(input_):
            ca_comm = communicator.ca_comm
            served = (
                "custom"
                if (
                    ca_comm is not None
                    and not ca_comm.disabled
                    and ca_comm.should_custom_ar(input_)
                )
                else "fallback"
            )
            counts[(int(input_.shape[0]), served)] += 1
            return original(input_)

        communicator.all_reduce = counting_all_reduce

    def drain_all_reduce_counter(self) -> dict:
        from vllm.distributed.parallel_state import get_tp_group

        counts = getattr(get_tp_group().device_communicator, "_bi_ar_counts", None)
        drained = {f"{n}:{served}": c for (n, served), c in (counts or {}).items()}
        if counts is not None:
            counts.clear()
        return drained


def _make_llm(**overrides) -> LLM:
    kwargs: dict = dict(
        model=MODEL,
        pipeline_parallel_size=PP_SIZE,
        tensor_parallel_size=TP_SIZE,
        max_num_seqs=MAX_BATCH_SIZE,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_model_len=int(os.getenv("VLLM_PP_TP_TEST_MAX_MODEL_LEN", "16384")),
        gpu_memory_utilization=float(
            os.getenv("VLLM_PP_TP_TEST_GPU_MEMORY_UTILIZATION", "0.25")
        ),
        # Off so that the needle's prefill is recomputed, and therefore
        # re-chunked, in every trial. See the PP=1 test for the full argument.
        enable_prefix_caching=False,
        attention_config={"backend": "TRITON_ATTN"},
    )
    kwargs.update(overrides)
    return LLM(**kwargs)


def test_tp_all_reduce_path_under_pipeline_parallelism(record_property):
    """Record which all-reduce implementation the composed run actually uses.

    Batch invariance dispatches to the custom all-reduce below its size limit
    and to all-gather plus a fixed rank-order sum above it. Which one a PP x TP
    run lands on is a property of the engine config, not of the mode, so the
    generation test below covers whichever it is -- but a silently disabled
    `ca_comm` would mean the run covers only the fallback, and that is worth
    knowing rather than guessing.
    """
    llm = None
    try:
        llm = _make_llm(worker_extension_cls=f"{__name__}.AllReduceProbe")
        reports = llm.collective_rpc("probe_all_reduce_path")
        llm.collective_rpc("install_all_reduce_counter")
        llm.generate(
            [_PROMPT_PADDING * _PADDING_REPEATS + "Say something."],
            SamplingParams(temperature=0.0, max_tokens=4),
            use_tqdm=False,
        )
        served = llm.collective_rpc("drain_all_reduce_counter")
    finally:
        if llm is not None:
            shutdown_llm(llm)

    assert {(r["pp_rank"], r["tp_rank"]) for r in reports} == {
        (pp, tp) for pp in range(PP_SIZE) for tp in range(TP_SIZE)
    }, f"expected one report per (pp, tp) rank, got {reports}"
    for report in reports:
        rank = report["rank"]
        assert report["envs_batch_invariant"], f"rank {rank}: mode not set in envs"
        assert report["mode_enabled"] and report["library_installed"], (
            f"rank {rank} never ran enable_batch_invariant_mode, so its GEMMs "
            f"are the platform's batch-variant ones: {report}"
        )
        assert not report["other_ar_backends_live"], (
            f"rank {rank}: an all-reduce backend the mode does not pin is "
            f"enabled: {report}"
        )

    assert any(served), (
        "no TP all-reduce was observed during generation, so this module's "
        f"needle test does not exercise the collective at all: {served}"
    )
    # A mixed state is a real bug and nothing else here would notice it: half
    # the ranks reducing through the custom kernel and half through the library
    # is precisely the configuration whose reduction order is not pinned.
    live = {report["ca_comm_live"] for report in reports}
    assert len(live) == 1, (
        "ranks disagree about whether the custom all-reduce is live, so this "
        f"module is covering two different reduction paths at once: {reports}"
    )

    if all(report["ca_comm_live"] for report in reports):
        # `CudaCommunicator` sizes max_size from max_num_batched_tokens under
        # the mode precisely so the custom kernel serves every message the
        # scheduler can produce. A fallback here means that sizing has
        # regressed, which is a performance bug rather than a correctness one --
        # both paths are invariant -- but it silently changes which path the
        # rest of this module covers.
        fell_back = sorted(
            {key for counts in served for key in counts if key.endswith(":fallback")}
        )
        assert not fell_back, (
            f"custom all-reduce is live but declined these token counts, so "
            f"max_size no longer covers the scheduler's largest batch: "
            f"{fell_back}"
        )
    else:
        # Not a failure -- both paths are invariant, and the custom kernel is
        # legitimately unavailable in some configurations. But this branch used
        # to be an invisible pass with zero assertions, which reads exactly like
        # a verified `max_size`. Say that it was not verified.
        warnings.warn(
            "the custom all-reduce is not live on any rank, so the max_size "
            "coverage assertion above did not run and this module is covering "
            "the library collective instead of the custom kernel.",
            stacklevel=2,
        )
    record_property("ca_comm_live", sorted(live))


def test_pp_tp_generation_is_batch_invariant():
    llm = None
    try:
        llm = _make_llm()
        assert_needle_is_batch_invariant(
            llm,
            padding_unit=_PROMPT_PADDING,
            padding_repeats=_PADDING_REPEATS,
            max_batch_size=MAX_BATCH_SIZE,
            max_tokens=int(os.getenv("VLLM_PP_TP_NEEDLE_MAX_TOKENS", "24")),
            num_trials=int(os.getenv("VLLM_PP_TP_NEEDLE_TRIALS", "3")),
            seed=int(os.getenv("VLLM_TEST_SEED", "12345")),
        )
    finally:
        if llm is not None:
            shutdown_llm(llm)
