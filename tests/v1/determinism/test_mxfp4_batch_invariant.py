# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end batch invariance for an MXFP4-quantised MoE model.

`test_mx_linear_batch_invariant.py` pins the MX GEMMs at the kernel level; this
runs a real MXFP4 checkpoint through the engine so the MoE path, attention and
the sampler are all covered at once.
"""

import contextlib
import os
import random

import pytest
import torch
from utils import _extract_step_logprobs, _random_prompt

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

MXFP4_TEST_MODEL = os.getenv(
    "VLLM_TEST_MXFP4_MODEL",
    "mawong-amd/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4",
)

requires_mx = pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="requires a ROCm device with native MX support (gfx95x)",
)


def _make_llm(max_num_seqs: int, backend: str) -> LLM:
    return LLM(
        model=MXFP4_TEST_MODEL,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(
            os.getenv("VLLM_MXFP4_TEST_GPU_MEMORY_UTILIZATION", "0.6")
        ),
        max_model_len=int(os.getenv("VLLM_MXFP4_TEST_MAX_MODEL_LEN", "2048")),
        dtype="auto",
        tensor_parallel_size=int(os.getenv("VLLM_MXFP4_TEST_TP_SIZE", "1")),
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_config={"backend": backend},
    )


@requires_mx
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
def test_mxfp4_moe_generation_is_bitwise_invariant_across_batch_sizes_e2e(backend):
    """The needle's per-step logprobs must be bitwise equal at bs=1 and bs=N.

    The needle is never placed at batch index 0: that position keeps its token
    offset between the solo and the batched run, so it can stay invariant even
    when the rest of the batch does not.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    num_trials = int(os.getenv("VLLM_MXFP4_NEEDLE_TRIALS", "2"))
    max_batch_size = int(os.getenv("VLLM_MXFP4_NEEDLE_BATCH_SIZE", "8"))
    min_random_prompt = int(os.getenv("VLLM_MXFP4_MIN_PROMPT", "32"))
    max_random_prompt = int(os.getenv("VLLM_MXFP4_MAX_PROMPT", "96"))
    assert max_batch_size >= 3, "Batch size should be >= 3 to place the needle."

    sampling = SamplingParams(
        temperature=float(os.getenv("VLLM_MXFP4_NEEDLE_TEMPERATURE", "0.6")),
        top_p=float(os.getenv("VLLM_MXFP4_NEEDLE_TOP_P", "0.95")),
        max_tokens=int(os.getenv("VLLM_MXFP4_NEEDLE_MAX_TOKENS", "16")),
        seed=20240919,
        logprobs=5,
    )
    needle_prompt = "Write one factual sentence about the moon."

    llm = None
    try:
        llm = _make_llm(max_num_seqs=max_batch_size, backend=backend)
        baseline_output = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        baseline_completion = baseline_output.outputs[0]
        baseline_logprobs, baseline_token_ids = _extract_step_logprobs(baseline_output)
        assert baseline_logprobs is not None
        assert baseline_token_ids is not None

        for _ in range(num_trials):
            batch_size = random.randint(max(3, max_batch_size // 2), max_batch_size)
            needle_pos = random.randint(1, batch_size - 1)
            prompts = [
                needle_prompt
                if idx == needle_pos
                else _random_prompt(min_random_prompt, max_random_prompt)
                for idx in range(batch_size)
            ]

            needle_output = llm.generate(prompts, sampling, use_tqdm=False)[needle_pos]
            needle_completion = needle_output.outputs[0]
            needle_logprobs, needle_token_ids = _extract_step_logprobs(needle_output)
            assert needle_logprobs is not None
            assert needle_token_ids is not None

            assert needle_output.prompt == needle_prompt
            assert needle_completion.token_ids == baseline_completion.token_ids
            assert needle_completion.text == baseline_completion.text
            assert torch.equal(needle_logprobs, baseline_logprobs), (
                f"Logprobs differ at needle position {needle_pos} of batch "
                f"{batch_size}: max |delta| = "
                f"{(needle_logprobs - baseline_logprobs).abs().max().item()}"
            )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()
