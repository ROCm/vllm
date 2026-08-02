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

Measured on 8x gfx950, Qwen3-1.7B bf16, TRITON_ATTN, 7208-token needle against
the 2032-token `batch_invariant_prefill_chunk` (four prefill chunks):

* the needle is bitwise stable in 4 of 4 trials with the mode on (batch sizes
  9/16/15/16, needle at positions 1/14/5/6), and moves in 4 of 4 with
  `VLLM_BATCH_INVARIANT=0` (max |delta| 6.9e-2 to 9.1e-2, same sampled tokens);
* every TP all-reduce is served by the custom kernel -- `ca_comm` is live on all
  eight ranks with `max_size` 8392704, sized from `max_num_batched_tokens` so
  that a full 2048-token by 2048-hidden bf16 message still fits. The all-gather
  fallback is therefore never reached in this configuration and is left to
  `test_tp_all_reduce_batch_invariant`'s `fallback_ar` worker;
* PP=2 does change which message sizes the collective sees relative to PP=1,
  which is the mechanism this test exists to cover. Counting token counts at the
  collective over one identical 8-prompt batch: PP=2 produces
  215/445/677/908/1114/1116/1578/1808/2033/2034/2035/2036 and PP=1 produces
  218/448/678/908/1117/1119/1579/1809/2034/2035/2036/2037/2038/2039 -- the two
  sets share one element. The custom kernel is size independent, so the shifted
  sizes cost nothing, but that is a measurement rather than a structural
  guarantee, which is why the needle below is run at this layout and not
  inferred from the PP=1 and TP-only results.

Not covered here: PP x TP x DP, PP over more than two stages, TP over more than
four ranks under PP, and MLA/MoE checkpoints (the JoyAI MXFP8 checkpoint used
elsewhere in this suite is TP=1 only).
"""

import contextlib
import os
import random

import torch
from utils import _extract_step_logprobs, skip_if_not_cuda_alike

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


def test_tp_all_reduce_path_under_pipeline_parallelism():
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
            with contextlib.suppress(Exception):
                llm.shutdown()

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


def test_pp_tp_generation_is_batch_invariant():
    """The needle's per-step logprobs must be bitwise equal at bs=1 and bs=N.

    The needle is never placed at batch index 0: that position keeps its token
    offset between the solo and the batched run, so it can stay invariant even
    when the rest of the batch does not.
    """
    random.seed(int(os.getenv("VLLM_TEST_SEED", "12345")))
    num_trials = int(os.getenv("VLLM_PP_TP_NEEDLE_TRIALS", "3"))
    assert MAX_BATCH_SIZE >= 3, "Batch size should be >= 3 to place the needle."

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(os.getenv("VLLM_PP_TP_NEEDLE_MAX_TOKENS", "24")),
        seed=20240919,
        logprobs=1,
    )
    padding = _PROMPT_PADDING * _PADDING_REPEATS
    needle_prompt = padding + "Write one factual sentence about the moon."

    llm = None
    try:
        llm = _make_llm()
        baseline_output = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        baseline_logprobs, baseline_token_ids = _extract_step_logprobs(baseline_output)
        assert baseline_logprobs is not None

        for _ in range(num_trials):
            batch_size = random.randint(3, MAX_BATCH_SIZE)
            needle_pos = random.randint(1, batch_size - 1)
            prompts = []
            for idx in range(batch_size):
                if idx == needle_pos:
                    prompts.append(needle_prompt)
                    continue
                # Staggered so the fillers finish prefilling on different steps
                # and the needle shares its forward passes with a changing mix
                # of prefill and decode.
                repeats = max(20, _PADDING_REPEATS * (idx + 1) // batch_size)
                prompts.append(
                    _PROMPT_PADDING * repeats
                    + f"Describe topic number {idx} in detail."
                )

            needle_output = llm.generate(prompts, sampling, use_tqdm=False)[needle_pos]
            needle_logprobs, needle_token_ids = _extract_step_logprobs(needle_output)
            assert needle_logprobs is not None

            assert needle_output.prompt == needle_prompt
            assert needle_token_ids == baseline_token_ids
            assert torch.equal(needle_logprobs, baseline_logprobs), (
                f"Logprobs differ at needle position {needle_pos} of batch "
                f"{batch_size}: max |delta| = "
                f"{(needle_logprobs - baseline_logprobs).abs().max().item()}"
            )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()
