#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture real MoE routing (topk_ids) from live prefills of a W4A16 MoE model.

Feeds the ``captured:<file>`` routing model of
benchmarks/kernels/sweep_hybrid_w4a16_moe_triton.py.

Hook point: ``HybridW4A16MoEExperts.apply`` is wrapped in-process (class
attribute replaced before the engine is built).  ``topk_ids`` is the
exact tensor the hybrid W4A16 experts see, i.e. after the router, top-k
selection and renormalisation, so no model code is touched.  The engine
runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), eager, with prefix
caching off.  One prompt of exactly N tokens is generated with
max_tokens=1 per N, so the first ``--layers`` apply() calls of that
generate() are layers 0..L-1 of the prefill step (assumes every layer is
an MoE layer, true for Qwen3-30B-A3B).

Prompts are natural English prose (repo docs by default, ``--text-file``
to override), tokenized and cut to the exact length, with a different
offset per length.

Output: torch file ``{"meta": {...}, "routing": {num_tokens: [ids_l0, ...]}}``
with each ids a CPU int32 tensor [num_tokens, top_k].

Usage:
    HF_HOME=... HF_HUB_OFFLINE=1 python benchmarks/kernels/capture_moe_routing.py \\
        --out /scratch/$USER/tmp/routing.pt
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

FALLBACK_TEXT = (
    "The history of computing is a story of steadily increasing abstraction. "
    "Early machines were programmed by rewiring plugboards, then by punched "
    "cards, then by assembly language, and finally by high level languages "
    "whose compilers hide the details of the underlying hardware. Each layer "
    "made programmers more productive while making performance harder to "
    "reason about. Today a data scientist can train a neural network in a few "
    "dozen lines, yet the speed of the result depends on memory bandwidth, "
    "cache behaviour, kernel launch overhead and the shape of every matrix "
    "multiplication that the framework dispatches. "
)


def load_corpus(path: str | None) -> str:
    if path:
        return Path(path).read_text()
    chunks = []
    for md in sorted((REPO / "docs").rglob("*.md"))[:200]:
        with contextlib.suppress(OSError):
            chunks.append(md.read_text())
        if sum(map(len, chunks)) > 400_000:
            break
    text = "\n\n".join(chunks)
    return text if len(text) > 50_000 else FALLBACK_TEXT * 400


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
    p.add_argument("--lengths", type=int, nargs="+", default=[6, 128, 512, 1024, 2048])
    p.add_argument("--layers", type=int, default=4, help="MoE layers 0..L-1")
    p.add_argument("--text-file", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.fused_moe.hybrid_w4a16_moe import (
        HybridW4A16MoEExperts,
    )

    state = {"on": False, "calls": 0, "ids": []}
    orig_apply = HybridW4A16MoEExperts.apply

    def hooked_apply(
        self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **kw
    ):
        if state["on"]:
            if state["calls"] < args.layers:
                state["ids"].append(topk_ids.detach().to("cpu", torch.int32).clone())
            state["calls"] += 1
        return orig_apply(
            self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **kw
        )

    HybridW4A16MoEExperts.apply = hooked_apply

    max_len = max(args.lengths)
    llm = LLM(
        args.model,
        dtype=args.dtype,
        max_model_len=max_len + 64,
        max_num_seqs=1,
        max_num_batched_tokens=max(max_len + 64, 2048),
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tok = llm.get_tokenizer()
    corpus_ids = tok(load_corpus(args.text_file), add_special_tokens=False)["input_ids"]
    assert len(corpus_ids) > 2 * max_len, "corpus too short"

    routing: dict[int, list[torch.Tensor]] = {}
    topk = None
    for i, n in enumerate(args.lengths):
        start = (i * 7919) % (len(corpus_ids) - n)
        ids = corpus_ids[start : start + n]
        state.update(on=True, calls=0, ids=[])
        llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0, max_tokens=1),
            use_tqdm=False,
        )
        state["on"] = False
        got = state["ids"]
        assert len(got) == args.layers, (len(got), args.layers)
        assert all(g.size(0) == n for g in got), [tuple(g.shape) for g in got]
        topk = got[0].size(1)
        routing[n] = got
        distinct = [int(torch.unique(g).numel()) for g in got]
        print(f"captured {n} tokens: {args.layers} layers, experts touched {distinct}")

    cfg = llm.llm_engine.model_config.hf_config
    meta = dict(
        model=args.model,
        num_experts=int(cfg.num_experts),
        top_k=int(topk),
        layers=list(range(args.layers)),
        lengths=list(args.lengths),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "routing": routing}, args.out)
    print(f"saved {args.out}: {meta}")


if __name__ == "__main__":
    main()
