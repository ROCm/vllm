# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary CI probe, not for merge.

Measures how far apart the bf16 and int4_per_token_head KV cache runs of
tests/models/quantization/test_per_token_kv_cache.py get on the CI machines.
For every prompt, prints the first diverging position and the rank of each
side's token in the other's top-NUM_LOG_PROBS list, as one JSON line
prefixed with INT4_PROBE.
"""

import json
import os
import socket

import pytest

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_TOKENS = 4
NUM_LOG_PROBS = 64
REPS = int(os.environ.get("INT4_PROBE_REPS", "6"))
ORIGINAL_NUM_LOG_PROBS = 8


def _rank(logprobs: dict, token_id: int) -> int | None:
    ordered = sorted(logprobs.items(), key=lambda kv: kv[1].logprob, reverse=True)
    for i, (tid, _) in enumerate(ordered[:NUM_LOG_PROBS]):
        if tid == token_id:
            return i + 1
    return None


def _top(logprobs: dict, n: int = 3) -> list:
    ordered = sorted(logprobs.items(), key=lambda kv: kv[1].logprob, reverse=True)
    return [[tid, round(lp.logprob, 4), lp.decoded_token] for tid, lp in ordered[:n]]


def _generate(vllm_runner, prompts, kv_cache_dtype):
    with vllm_runner(
        MODEL,
        max_model_len=1024,
        tensor_parallel_size=1,
        enforce_eager=True,
        kv_cache_dtype=kv_cache_dtype,
        attention_config={"backend": "TRITON_ATTN"},
        max_logprobs=NUM_LOG_PROBS,
    ) as vllm_model:
        return vllm_model.generate_greedy_logprobs(prompts, MAX_TOKENS, NUM_LOG_PROBS)


@pytest.mark.parametrize("rep", range(REPS))
def test_int4_kv_rank_probe(vllm_runner, example_prompts, monkeypatch, rep):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    base = _generate(vllm_runner, example_prompts, "auto")
    test = _generate(vllm_runner, example_prompts, "int4_per_token_head")

    node = os.environ.get("VLLM_CI_K8S_NODE_NAME") or socket.gethostname()
    shard = os.environ.get("BUILDKITE_PARALLEL_JOB")
    for prompt_idx, ((ids_0, text_0, lps_0), (ids_1, text_1, lps_1)) in enumerate(
        zip(base, test)
    ):
        record = {
            "node": node,
            "shard": shard,
            "rep": rep,
            "prompt": prompt_idx,
            "bf16_text": text_0,
            "int4_text": text_1,
            "diverged_at": None,
        }
        for pos, (tok_0, tok_1) in enumerate(zip(ids_0, ids_1)):
            if tok_0 == tok_1:
                continue
            rank_0_in_1 = _rank(lps_1[pos], tok_0)
            rank_1_in_0 = _rank(lps_0[pos], tok_1)
            record.update(
                diverged_at=pos,
                bf16_tok_rank_in_int4=rank_0_in_1,
                int4_tok_rank_in_bf16=rank_1_in_0,
                bf16_top=_top(lps_0[pos]),
                int4_top=_top(lps_1[pos]),
                original_check_fails=(
                    rank_0_in_1 is None
                    or rank_1_in_0 is None
                    or rank_0_in_1 > ORIGINAL_NUM_LOG_PROBS
                    or rank_1_in_0 > ORIGINAL_NUM_LOG_PROBS
                ),
            )
            break
        print("INT4_PROBE " + json.dumps(record), flush=True)
