#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_ATTN (gfx1151) against TRITON_ATTN on batches: many decodes, and mixed.

`matrix.py` measures one sequence.  This measures what a server runs: batches
of sequences decoding together (one launch, grid.y per sequence) and batches
mixing decodes with a prefill or a chunked-prefill extend (decodes on the
kernel, the rest on Triton).

Decode batches are timed under CUDA graphs, as vLLM replays them.  Mixed
batches are timed eagerly: vLLM does not capture them whole, and the backend
only splits outside a capture.  Decodes are listed first in every mixed spec,
the order vLLM's batch reordering gives the backend.

Each cell names the path the backend took: `kernel` (every call on the kernel),
`split` (decodes on the kernel, the rest on Triton) or `triton` (fallback).

    cd <worktree> && VLLM_KV_CACHE_LAYOUT=HND amd-gpu-lock \\
        <venv>/bin/python benchmarks/kernels/gfx1151_decode_attn/tools/batch.py
"""

import argparse
import pathlib
import statistics
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import shapeset  # noqa: E402

DECODE = ["2q1s4k", "8q1s4k", "8q4s4k", "32q1s1k", "64q1s1k", "32q4s1k"]
MIXED = ["4q1s8k_q512", "16q1s4k_2q1k", "16q1s2k_q64s2k", "32q1s1k_q64s1k"]


def roofline_us(spec: str, hq: int, hkv: int, d: int, window: int, itemsize: int):
    """One dispatch plus every decode sequence's bytes; None if any request
    is a prefill (its floor is compute, not bytes)."""
    from batch_spec import parse_batch_spec

    reqs = parse_batch_spec(spec)
    if any(r.q_len > 8 for r in reqs):
        return None
    moved = sum(
        shapeset.roofline_us(
            hq, hkv, d, r.q_len, r.kv_len, itemsize=itemsize, window=window
        )
        - shapeset.DISPATCH_US
        for r in reqs
    )
    return shapeset.DISPATCH_US + moved


def main() -> None:
    p = argparse.ArgumentParser()
    shapeset.add_arguments(p)
    shapeset.add_dtype_argument(p)
    p.add_argument("--decode", nargs="*", default=DECODE)
    p.add_argument("--mixed", nargs="*", default=MIXED)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--reps", type=int, default=1)
    args = p.parse_args()
    dtype = shapeset.torch_dtype(args.dtype)
    itemsize = 2

    bench = pathlib.Path(__file__).resolve().parents[3] / "attention_benchmarks"
    sys.path.insert(0, str(bench))
    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    import vllm.v1.attention.backends.rocm_attn as backend_mod

    shapes, _ = shapeset.load(args)
    groups: dict[tuple[int, int, int, int], list[str]] = {}
    for sh in shapes:
        if sh.d == 96:  # three elements per lane: never served
            continue
        groups.setdefault((sh.hq, sh.hkv, sh.d, sh.window), []).append(sh.model)

    def run(backend, spec, hq, hkv, d, window, graphs):
        cfg = BenchmarkConfig(
            backend=backend,
            batch_spec=spec,
            num_layers=10,
            min_working_set_mb=96,
            head_dim=d,
            num_q_heads=hq,
            num_kv_heads=hkv,
            block_size=args.block_size,
            device="cuda:0",
            dtype=dtype,
            sliding_window=window or None,
            use_cuda_graphs=graphs,
        )
        rs = [run_attention_benchmark(cfg) for _ in range(args.reps)]
        return statistics.median(r.median_time for r in rs) * 1e6

    def ours(spec, hq, hkv, d, window, graphs):
        impls: list[Any] = []
        orig = backend_mod.RocmAttentionRdna35Impl.__init__

        def spy(self, *a, _init=orig, _seen=impls, **kw):
            _init(self, *a, **kw)
            _seen.append(self)

        backend_mod.RocmAttentionRdna35Impl.__init__ = spy
        try:
            # First call builds whatever variant the batch needs; a build
            # inside the timed run would corrupt its median.
            run("ROCM_ATTN", spec, hq, hkv, d, window, graphs)
            impls.clear()
            us = run("ROCM_ATTN", spec, hq, hkv, d, window, graphs)
        finally:
            backend_mod.RocmAttentionRdna35Impl.__init__ = orig
        k = sum(i.kernel_calls for i in impls)
        f = sum(i.fallback_calls for i in impls)
        split = sum(getattr(i, "split_calls", 0) for i in impls)
        path = "split" if split else ("kernel" if k and not f else "triton")
        return us, path

    print(
        f"<!-- {len(groups)} configurations, {args.dtype}, HND, block size "
        f"{args.block_size}; decode under CUDA graphs, mixed eager -->"
    )
    print(
        "| models | Hq | Hkv | D | window | batch | graphs | path | roofline | "
        "Triton | ours | vs | %roof |"
    )
    print("| --- " * 13 + "|")
    for (hq, hkv, d, window), models in sorted(groups.items()):
        label = models[0] if len(models) == 1 else f"{models[0]} +{len(models) - 1}"
        for spec, graphs in [(s, True) for s in args.decode] + [
            (s, False) for s in args.mixed
        ]:
            roof = roofline_us(spec, hq, hkv, d, window, itemsize)
            us, path = ours(spec, hq, hkv, d, window, graphs)
            tri = run("TRITON_ATTN", spec, hq, hkv, d, window, graphs)
            pct = "-" if roof is None else f"{roof / us * 100:.1f} %"
            rf = "-" if roof is None else f"{roof:.1f}"
            print(
                f"| {label} | {hq} | {hkv} | {d} | {window or '-'} | {spec} | "
                f"{'yes' if graphs else 'no'} | {path} | {rf} | {tri:.1f} | "
                f"{us:.1f} | {tri / us:.2f}x | {pct} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
