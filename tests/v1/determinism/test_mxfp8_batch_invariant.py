# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end batch invariance for MXFP8-quantised models.

`test_mx_linear_batch_invariant.py` pins the MXFP8 GEMM at the kernel level;
this runs real MXFP8 checkpoints through the engine so attention, the sampler
and the ROCm dot_scaled linear path are covered together. The dense Qwen3 is
the cheap one; the JoyAI checkpoint is MoE and is the only coverage of
``Mxfp8NativeTritonExperts`` and of MLA under batch invariance.

Keep the MoE case at TP=1: its moe_intermediate_size is 768, and the native
grouped GEMM needs the per-partition intermediate divisible by 128, so TP=2
still works (384) but TP=4 (192) silently falls back to the BF16 emulation
backend -- the test would pass while covering nothing it was added for.

The prompts have to be long. ``RocmDotScaledMxfp8LinearKernel`` picks BLOCK_K
from M, and for this model's shapes it only changes above M=256 -- with the
short prompts the other e2e tests use, the batch never reaches that band and
the test passes with VLLM_BATCH_INVARIANT unset, i.e. asserts nothing. Measured
on gfx950 with the padding and batch size below, the needle's logprobs move with
the batch size in 6 of 8 trials when the mode is off, and are bitwise stable in
all of them when it is on. Shortening the padding or the batch makes it pass either
way -- if this test is ever made cheaper, re-check that it still fails with
VLLM_BATCH_INVARIANT unset.
"""

import contextlib
import os
import random

import pytest
import torch
from utils import _extract_step_logprobs, requires_mx

from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams

# (model, attention backend). The MoE checkpoint is 55GB, hence the size gate;
# it needs MLA, which is selected from model_type and rejects TRITON_ATTN.
MXFP8_CASES = [
    pytest.param("mgoin/Qwen3-0.6B-MXFP8", "TRITON_ATTN", id="qwen3-dense"),
    # The only coverage of Mxfp8NativeTritonExperts, and of MLA under the mode.
    # It was batch variant until MLA prefill stopped splitting at the scheduler's
    # batch-dependent chunk boundary: 2/3 needle trials differed at this batch
    # and padding before that fix, 0/3 after, with 3/3 still differing when the
    # mode is off.
    pytest.param(
        "mawong-amd/JoyAI-LLM-Flash-MXFP8-last-6-BF16-fixed",
        "TRITON_MLA",
        marks=large_gpu_mark(min_gb=80),
        id="joyai-moe",
    ),
]

# Long enough that a batch crosses the M band where BLOCK_K changes; see above.
_PROMPT_PADDING = "Some background context for the question that follows. "
_PADDING_REPEATS = int(os.getenv("VLLM_MXFP8_PROMPT_REPEATS", "400"))


def _make_llm(model: str, max_num_seqs: int, backend: str) -> LLM:
    return LLM(
        model=model,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(
            os.getenv("VLLM_MXFP8_TEST_GPU_MEMORY_UTILIZATION", "0.3")
        ),
        max_model_len=int(os.getenv("VLLM_MXFP8_TEST_MAX_MODEL_LEN", "8192")),
        dtype="auto",
        tensor_parallel_size=int(os.getenv("VLLM_MXFP8_TEST_TP_SIZE", "1")),
        enable_prefix_caching=False,
        attention_config={"backend": backend},
    )


@requires_mx
@pytest.mark.parametrize("model,backend", MXFP8_CASES)
def test_mxfp8_generation_is_bitwise_invariant_across_batch_sizes_e2e(model, backend):
    """The needle's per-step logprobs must be bitwise equal at bs=1 and bs=N.

    The needle is never placed at batch index 0: that position keeps its token
    offset between the solo and the batched run, so it can stay invariant even
    when the rest of the batch does not.
    """
    random.seed(int(os.getenv("VLLM_TEST_SEED", "12345")))

    num_trials = int(os.getenv("VLLM_MXFP8_NEEDLE_TRIALS", "3"))
    max_batch_size = int(os.getenv("VLLM_MXFP8_NEEDLE_BATCH_SIZE", "16"))
    assert max_batch_size >= 3, "Batch size should be >= 3 to place the needle."

    sampling = SamplingParams(
        temperature=float(os.getenv("VLLM_MXFP8_NEEDLE_TEMPERATURE", "0.6")),
        top_p=float(os.getenv("VLLM_MXFP8_NEEDLE_TOP_P", "0.95")),
        max_tokens=int(os.getenv("VLLM_MXFP8_NEEDLE_MAX_TOKENS", "16")),
        seed=20240919,
        logprobs=5,
    )
    padding = _PROMPT_PADDING * _PADDING_REPEATS
    needle_prompt = padding + "Write one factual sentence about the moon."

    llm = None
    try:
        llm = _make_llm(model, max_num_seqs=max_batch_size, backend=backend)
        baseline_output = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        baseline_completion = baseline_output.outputs[0]
        baseline_logprobs, baseline_token_ids = _extract_step_logprobs(baseline_output)
        assert baseline_logprobs is not None
        assert baseline_token_ids is not None

        for _ in range(num_trials):
            batch_size = random.randint(3, max_batch_size)
            needle_pos = random.randint(1, batch_size - 1)
            prompts = [
                needle_prompt
                if idx == needle_pos
                else padding + f"Describe topic number {idx} in detail."
                for idx in range(batch_size)
            ]

            needle_output = llm.generate(prompts, sampling, use_tqdm=False)[needle_pos]
            needle_completion = needle_output.outputs[0]
            needle_logprobs, _ = _extract_step_logprobs(needle_output)
            assert needle_logprobs is not None

            assert needle_output.prompt == needle_prompt
            assert needle_completion.token_ids == baseline_completion.token_ids
            assert torch.equal(needle_logprobs, baseline_logprobs), (
                f"Logprobs differ at needle position {needle_pos} of batch "
                f"{batch_size}: max |delta| = "
                f"{(needle_logprobs - baseline_logprobs).abs().max().item()}"
            )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()
