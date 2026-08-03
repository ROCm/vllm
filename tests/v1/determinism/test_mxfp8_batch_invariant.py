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

Activation dtype is an axis here, not a free parameter. A serialized MXFP8
checkpoint pins it: ``ModelOptMxFp8Config.get_supported_act_dtypes`` is
``[bfloat16]``, so ``dtype="float16"`` on either checkpoint above raises rather
than taking effect. Online MXFP8 quantization of a BF16 checkpoint
(``quantization="mxfp8"``) accepts half, which is how the fp16 case below
reaches the fp16 branch of ``matmul_persistent`` -- with its own BLOCK_SIZE_N
(``_fp16_block_size_n``, 256 on gfx950's 160KB LDS) and its own Triton
compilation. float32 is out of reach for every MXFP8 path: online quantization
supports ``[bfloat16, float16]``.
"""

import os
import random
from typing import NamedTuple

import pytest
import torch
from utils import _extract_step_logprobs, requires_mx, shutdown_llm

from tests.utils import create_new_process_for_each_test, large_gpu_mark
from vllm import LLM, SamplingParams


class Case(NamedTuple):
    model: str
    backend: str
    dtype: str = "auto"
    quantization: str | None = None


# The MoE checkpoint is 55GB, hence the size gate; it needs MLA, which is
# selected from model_type and rejects TRITON_ATTN.
MXFP8_CASES = [
    pytest.param(Case("mgoin/Qwen3-0.6B-MXFP8", "TRITON_ATTN"), id="qwen3-dense"),
    # fp16 activations, the only coverage of them under the mode. The serialized
    # checkpoints reject half (see the module docstring), so this quantizes a BF16
    # Qwen3 to MXFP8 at load instead: same RocmDotScaledMxfp8LinearKernel, and the
    # unquantized logits GEMM -- the one the needle's logprobs come straight out
    # of -- runs through matmul_persistent's fp16 config. Instrumented on gfx950,
    # matmul_persistent is entered once per decode step with torch.float16
    # operands and BLOCK_SIZE_N=256. The needle differs in 6 of 8 trials with
    # VLLM_BATCH_INVARIANT=0 and 0 of 8 with it on -- 2 of 3 and 0 of 3 over the
    # default VLLM_MXFP8_NEEDLE_TRIALS.
    pytest.param(
        Case("Qwen/Qwen3-1.7B", "TRITON_ATTN", dtype="float16", quantization="mxfp8"),
        id="qwen3-online-fp16",
    ),
    # The only coverage of Mxfp8NativeTritonExperts, and of MLA under the mode.
    # It was batch variant until MLA prefill stopped splitting at the scheduler's
    # batch-dependent chunk boundary: 2/3 needle trials differed at this batch
    # and padding before that fix, 0/3 after, with 3/3 still differing when the
    # mode is off.
    pytest.param(
        Case("mawong-amd/JoyAI-LLM-Flash-MXFP8-last-6-BF16-fixed", "TRITON_MLA"),
        marks=large_gpu_mark(min_gb=80),
        id="joyai-moe",
    ),
]

# Long enough that a batch crosses the M band where BLOCK_K changes; see above.
_PROMPT_PADDING = "Some background context for the question that follows. "
_PADDING_REPEATS = int(os.getenv("VLLM_MXFP8_PROMPT_REPEATS", "400"))


def _make_llm(case: Case, max_num_seqs: int, **overrides) -> LLM:
    kwargs = dict(
        model=case.model,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(
            os.getenv("VLLM_MXFP8_TEST_GPU_MEMORY_UTILIZATION", "0.3")
        ),
        max_model_len=int(os.getenv("VLLM_MXFP8_TEST_MAX_MODEL_LEN", "8192")),
        dtype=case.dtype,
        tensor_parallel_size=int(os.getenv("VLLM_MXFP8_TEST_TP_SIZE", "1")),
        enable_prefix_caching=False,
        attention_config={"backend": case.backend},
    )
    if case.quantization is not None:
        kwargs["quantization"] = case.quantization
    kwargs.update(overrides)
    return LLM(**kwargs)


def _assert_needle_is_invariant(llm: LLM, max_batch_size: int) -> None:
    """The needle's per-step logprobs must be bitwise equal at bs=1 and bs=N.

    The needle is never placed at batch index 0: that position keeps its token
    offset between the solo and the batched run, so it can stay invariant even
    when the rest of the batch does not.
    """
    random.seed(int(os.getenv("VLLM_TEST_SEED", "12345")))

    num_trials = int(os.getenv("VLLM_MXFP8_NEEDLE_TRIALS", "3"))
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


@requires_mx
@pytest.mark.parametrize("case", MXFP8_CASES)
def test_mxfp8_generation_is_bitwise_invariant_across_batch_sizes_e2e(case: Case):
    max_batch_size = int(os.getenv("VLLM_MXFP8_NEEDLE_BATCH_SIZE", "16"))
    llm = None
    try:
        llm = _make_llm(case, max_num_seqs=max_batch_size)
        _assert_needle_is_invariant(llm, max_batch_size)
    finally:
        if llm is not None:
            shutdown_llm(llm)


@requires_mx
@large_gpu_mark(min_gb=80)
@create_new_process_for_each_test()
def test_mxfp8_mla_multi_chunk_context_is_batch_invariant():
    """MLA prefill must be invariant when a context is merged over >1 chunk.

    ``build_mla_chunked_context_metadata`` splits a prefill's context into
    ``max_context_chunk``-sized pieces and ``_forward_prefill`` merges one
    attention call per piece; at the case above's max_model_len and max_num_seqs
    the chunk is 4096 and the 3608-token prompts never reach a second one, so
    the merge loop has only ever run with a single chunk. The workspace is
    ``8 * max_model_len`` capped at 64k and, under the mode, the chunk is that
    divided by max_num_seqs -- so max_model_len=4096 with max_num_seqs=32 gives a
    1024-token chunk, and max_num_batched_tokens=1024 caps each prefill step at
    1024 - 32 = 992 tokens, walking the context through 992/1984/2976. Measured
    on gfx950, num_chunks is 1, 2 and 3 in equal numbers (35 metadata builds
    each over a 3-trial run), including 3 in the bs=1 baseline; the assertion
    below keeps that honest rather than trusting the arithmetic.

    Sensitivity: 3 of 3 needle trials differ with VLLM_BATCH_INVARIANT=0 and 0
    of 3 with it on. Note the control does not run the multi-chunk merge itself
    -- with the mode off the chunk is the workspace divided by the live prefill
    count, and with a 1024-token budget only one prefill has context per step,
    so num_chunks collapses to 1 (122 builds, all single-chunk). The mode is
    what creates the multi-chunk path; the control only shows the case is not
    numerically inert.

    The engine runs in-process so the chunk counter is visible; the other e2e
    cases keep the default multiprocessing engine. That is also why the whole
    test runs in a spawned interpreter: an in-process engine's VRAM is *not*
    recoverable within the process that built it. Measured on gfx950 with a
    0.6B model at the same gpu_memory_utilization, ``engine_core.shutdown()``,
    ``del``, ``gc.unfreeze()``, ``gc.collect()``, ``empty_cache()`` and
    ``cleanup_dist_env_and_memory()`` -- in that order and in every other --
    leave the allocator's reported live bytes at 27.91 of 27.92 GiB, because the
    compiled artifacts pin the model and the KV cache from module-level lists
    in Inductor-generated code; ``LLMEngine._cleanup_instance_caches`` only
    unhooks the bytecode hook and does not reach them. The same teardown with
    ``enforce_eager=True`` frees all of it, which is what identifies torch
    compilation as the holder. Left in-process this test therefore parked
    gpu_memory_utilization x total VRAM -- 86 GiB here -- in the pytest process
    for the rest of the session, and every module after it in a full-suite run
    started short of memory. Spawn, not fork: the parent has a live HIP context
    by the time this runs, so ``os.fork`` gives "Cannot re-initialize CUDA in
    forked subprocess".
    """
    import os as _os

    from vllm.model_executor.layers.attention import mla_attention

    # The child interpreter gets no pytest fixtures, so the autouse
    # `enable_batch_invariant_mode` does not apply here; the parent's setenv
    # is inherited through the environment, and this makes that explicit.
    _os.environ["VLLM_BATCH_INVARIANT"] = "1"
    _os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    num_chunks_seen: set[int] = set()
    build = mla_attention.build_mla_chunked_context_metadata

    def counting_build(**kwargs):
        metadata = build(**kwargs)
        if metadata is not None:
            num_chunks_seen.add(int(metadata.seq_lens.shape[0]))
        return metadata

    # Plain assignment rather than the monkeypatch fixture: this body runs in a
    # child interpreter that exits at the end of the test, so there is nothing
    # to restore, and the fixture would have to survive being pickled into it.
    mla_attention.build_mla_chunked_context_metadata = counting_build

    max_batch_size = int(os.getenv("VLLM_MXFP8_NEEDLE_BATCH_SIZE", "16"))
    llm = None
    try:
        llm = _make_llm(
            Case("mawong-amd/JoyAI-LLM-Flash-MXFP8-last-6-BF16-fixed", "TRITON_MLA"),
            max_num_seqs=32,
            max_model_len=4096,
            max_num_batched_tokens=1024,
        )
        _assert_needle_is_invariant(llm, max_batch_size)
    finally:
        if llm is not None:
            shutdown_llm(llm)

    assert max(num_chunks_seen, default=0) > 1, (
        "the chunked-context merge never ran with more than one chunk "
        f"(num_chunks seen: {sorted(num_chunks_seen)}), so this test covers "
        "nothing the single-chunk cases do not"
    )
