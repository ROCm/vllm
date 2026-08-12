# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline parallelism must not make a token's output depend on its batch.

PP has no reduction whose order can move: the stages exchange activations with
point-to-point isend/irecv and the last stage broadcasts int32 token ids, all of
which are pure data movement. What it does change is the process layout and the
scheduler, and both are checked here.

`test_batch_invariance_is_installed_on_every_pipeline_rank` covers the layout.
The aten overrides are registered per process by `init_batch_invariance`, called
from `init_worker_distributed_environment`, so a rank that missed them would
still produce plausible numbers -- the failure would read as ordinary noise
rather than as a broken invariant. The probe therefore asks each worker directly,
and proves the question is not vacuous by destroying the override and re-asking.

`test_pipeline_parallel_generation_is_batch_invariant` covers the scheduler. PP
keeps `pipeline_parallel_size` batches in flight, so the set of tokens sharing a
forward pass with a given request changes step to step in a way it does not at
PP=1. The prompt below is 7208 tokens against a 2032-token
`batch_invariant_prefill_chunk`, i.e. four prefill chunks, so the chunked-prefill
split that cap exists to pin is exercised while the pipeline is full, and the
filler prompts are staggered in length so the needle decodes alongside other
requests' prefills.

Measured on 2x gfx950, PP=2, Qwen3-1.7B bf16, TRITON_ATTN, in exactly the
configuration below but over 6 trials rather than the default 3: the needle is
bitwise stable in 6 of 6 with the mode on and moves in 6 of 6 with it off (max
|delta| 6.1e-2 to 1.3e-1). Instrumenting the scheduler over the same prompts,
the needle's per-step scheduled-token sequence is 2032/2032/2032/1112 solo and
identical in every batched trial, at PP=2 and at PP=1 alike; with the cap
removed it becomes 1248/2048/2047/1865, 1710/2045/2046/1407, ... i.e.
batch-dependent. So the cap keeps holding once there are multiple batches in
flight.

MLA is left to the PP=1 tests rather than parametrized in here: the JoyAI MXFP8
MoE checkpoint was measured at PP=2 in the same shape as the case above (3608
tokens, two chunks, TRITON_MLA) and is bitwise stable 3 trials of 3 with the mode
on and differs 3 of 3 with it off, but the model is 55GB and adds nothing this
case does not once the chunk boundaries are pinned.

Not covered here: PP together with DP, and PP over more than 2 stages. PP=2 x
TP=4 is covered separately in `test_pp_tp_batch_invariant`, which needs 8 GPUs.
"""

import os

import torch
from utils import (
    assert_needle_is_batch_invariant,
    shutdown_llm,
    skip_if_not_cuda_alike,
)

from tests.utils import multi_gpu_marks
from vllm import LLM

# multi_gpu_test would also wrap the test in create_new_process_for_each_test,
# whose re-import would break the worker extension lookup below, so take its
# marks alone: `-m distributed` still selects the module and the skipif still
# enforces the two GPUs PP=2 needs.
pytestmark = [skip_if_not_cuda_alike, *multi_gpu_marks(num_gpus=2)]

MODEL = os.getenv("VLLM_PP_TEST_MODEL", "Qwen/Qwen3-1.7B")
MAX_BATCH_SIZE = int(os.getenv("VLLM_PP_NEEDLE_BATCH_SIZE", "16"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("VLLM_PP_MAX_NUM_BATCHED_TOKENS", "2048"))
# 800 repeats is 7208 tokens, four chunks at the resulting 2032-token cap.
_PROMPT_PADDING = "Some background context for the question that follows. "
_PADDING_REPEATS = int(os.getenv("VLLM_PP_PROMPT_REPEATS", "800"))

# Rows of a [M, N] product to compare across M. Row 0 is reported but never
# decides the verdict: it sits at offset 0 in every launch, so it can stay put
# under a split-k decomposition that moves everything else.
_PROBE_CHECK_ROWS = [0, 1, 3, 7, 63, 200]
_PROBE_M_VALUES = [1, 2, 3, 8, 17, 64, 128, 201, 512]


class BatchInvariantProbe:
    """Worker extension: report whether the overrides are live on this rank."""

    # Supplied by the Worker this class is mixed into.
    rank: int
    device: torch.device

    def probe_batch_invariance(self) -> dict:
        import vllm.envs as envs
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.model_executor.layers import batch_invariant as bi

        device = self.device
        generator = torch.Generator(device=device).manual_seed(1234)
        lhs = torch.randn(
            512, 4096, generator=generator, device=device, dtype=torch.bfloat16
        )
        rhs = torch.randn(
            4096, 4096, generator=generator, device=device, dtype=torch.bfloat16
        )

        def rows_that_move() -> dict[int, list[int]]:
            reference = torch.mm(lhs, rhs)
            moved: dict[int, list[int]] = {}
            for num_rows in _PROBE_M_VALUES:
                out = torch.mm(lhs[:num_rows], rhs)
                for row in _PROBE_CHECK_ROWS:
                    if row < num_rows and not torch.equal(out[row], reference[row]):
                        moved.setdefault(row, []).append(num_rows)
            return moved

        with_override = rows_that_move()
        # Deregistering the library restores the platform GEMM, which is where
        # the numbers this sweep is meant to catch come from. Destructive, so it
        # runs last and the caller shuts the engine down afterwards.
        if bi._batch_invariant_LIB is not None:
            bi._batch_invariant_LIB._destroy()
        without_override = rows_that_move()

        return {
            "rank": self.rank,
            "pp_rank": get_pp_group().rank_in_group,
            "envs_batch_invariant": bool(envs.VLLM_BATCH_INVARIANT),
            "mode_enabled": bi._batch_invariant_MODE,
            "library_installed": bi._batch_invariant_LIB is not None,
            "rows_that_move_with_override": with_override,
            "rows_that_move_without_override": without_override,
        }


def _make_llm(**overrides) -> LLM:
    kwargs: dict = dict(
        model=MODEL,
        pipeline_parallel_size=2,
        tensor_parallel_size=1,
        max_num_seqs=MAX_BATCH_SIZE,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_model_len=int(os.getenv("VLLM_PP_TEST_MAX_MODEL_LEN", "16384")),
        gpu_memory_utilization=float(
            os.getenv("VLLM_PP_TEST_GPU_MEMORY_UTILIZATION", "0.25")
        ),
        # Off so that the needle's prefill is recomputed, and therefore
        # re-chunked, in every trial. With it on the baseline run's blocks are
        # still cached when the batched trials start, the needle's prefill
        # collapses to a single token, and only the decodes are compared.
        enable_prefix_caching=False,
        attention_config={"backend": "TRITON_ATTN"},
    )
    kwargs.update(overrides)
    return LLM(**kwargs)


def test_batch_invariance_is_installed_on_every_pipeline_rank():
    """Every PP rank, not just rank 0, must have the aten overrides."""
    llm = None
    try:
        llm = _make_llm(
            worker_extension_cls=f"{__name__}.BatchInvariantProbe",
            enforce_eager=True,
        )
        reports = llm.collective_rpc("probe_batch_invariance")
    finally:
        if llm is not None:
            shutdown_llm(llm)

    assert {report["pp_rank"] for report in reports} == {0, 1}, (
        f"expected one report per pipeline stage, got {reports}"
    )
    for report in reports:
        rank = report["rank"]
        assert report["envs_batch_invariant"], f"rank {rank}: mode not set in envs"
        assert report["mode_enabled"] and report["library_installed"], (
            f"rank {rank} never ran enable_batch_invariant_mode, so its GEMMs "
            f"are the platform's batch-variant ones: {report}"
        )
        # Row 0 is excluded from the vacuity check: see _PROBE_CHECK_ROWS.
        sensitive = {
            row: counts
            for row, counts in report["rows_that_move_without_override"].items()
            if int(row) != 0
        }
        assert sensitive, (
            f"rank {rank}: the platform GEMM is already row-count independent "
            f"for these shapes, so the sweep below cannot fail and asserts "
            f"nothing: {report}"
        )
        assert not report["rows_that_move_with_override"], (
            f"rank {rank}: a row of A[:m] @ B changed with m under the mode, "
            f"so the override is registered but not serving: {report}"
        )


def test_pipeline_parallel_generation_is_batch_invariant():
    llm = None
    try:
        llm = _make_llm()
        assert_needle_is_batch_invariant(
            llm,
            padding_unit=_PROMPT_PADDING,
            padding_repeats=_PADDING_REPEATS,
            max_batch_size=MAX_BATCH_SIZE,
            max_tokens=int(os.getenv("VLLM_PP_NEEDLE_MAX_TOKENS", "24")),
            num_trials=int(os.getenv("VLLM_PP_NEEDLE_TRIALS", "3")),
            seed=int(os.getenv("VLLM_TEST_SEED", "12345")),
        )
    finally:
        if llm is not None:
            shutdown_llm(llm)
