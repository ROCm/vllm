# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end batch invariance for MXFP4-quantised MoE models.

`test_mx_linear_batch_invariant.py` pins the MX GEMMs at the kernel level; this
runs real MXFP4 checkpoints through the engine so the MoE path, attention and
the sampler are all covered at once.

Both checkpoints are sensitive at the short prompts used here -- measured on
gfx950, the needle's logprobs move with the batch size in 6 of 6 trials
(Llama-4-Scout) and 3 of 3 (gpt-oss) when VLLM_BATCH_INVARIANT is unset, and
are bitwise stable in all of them when it is set. Unlike the MXFP8 e2e test,
no prompt padding is needed to reach that sensitivity: the router turns a small
numerical difference into a different expert choice.
"""

import os
import random

import pytest
import torch
from utils import _extract_step_logprobs, _random_prompt, requires_mx, shutdown_llm

from vllm import LLM, SamplingParams

# The 2-layer Scout is the cheap one; gpt-oss additionally exercises fp8
# activations and an fp8 KV cache alongside the MXFP4 weights.
MXFP4_TEST_MODELS = os.getenv(
    "VLLM_TEST_MXFP4_MODELS",
    "mawong-amd/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4,"
    "amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8",
).split(",")


def _make_llm(model: str, max_num_seqs: int, backend: str) -> LLM:
    return LLM(
        model=model,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(
            os.getenv("VLLM_MXFP4_TEST_GPU_MEMORY_UTILIZATION", "0.6")
        ),
        max_model_len=int(os.getenv("VLLM_MXFP4_TEST_MAX_MODEL_LEN", "2048")),
        dtype="auto",
        tensor_parallel_size=int(os.getenv("VLLM_MXFP4_TEST_TP_SIZE", "1")),
        enable_prefix_caching=False,
        attention_config={"backend": backend},
    )


@requires_mx
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
@pytest.mark.parametrize("model", MXFP4_TEST_MODELS)
def test_mxfp4_moe_generation_is_bitwise_invariant_across_batch_sizes_e2e(
    model, backend
):
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
        llm = _make_llm(model, max_num_seqs=max_batch_size, backend=backend)
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
            shutdown_llm(llm)
