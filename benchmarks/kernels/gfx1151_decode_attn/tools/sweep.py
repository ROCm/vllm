#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare RDNA35_HIP_ATTN against TRITON_ATTN through vLLM's own harness.

Both backends go through run_attention_benchmark, so the KV tensors, the
metadata and the timer are identical; only the kernel launch differs.

    cd <worktree> && VLLM_KV_CACHE_LAYOUT=HND \\
        PYTHONPATH=$PWD amd-gpu-lock <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/sweep.py --hq 8 --hkv 4

Pass --nseg/--block/--msplit/--kpw to override what the backend would pick,
which is how the tuning sweeps are run.
"""

import argparse
import contextlib
import itertools
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
sys.path.insert(0, str(_ROOT / "benchmarks" / "attention_benchmarks"))

PEAK_GIBS = 230.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hq", type=int, default=32)
    p.add_argument("--hkv", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--m", type=int, default=4, help="query tokens per sequence")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument(
        "--contexts", type=int, nargs="+", default=[128, 512, 1024, 4096, 8192, 32768]
    )
    p.add_argument("--nseg", type=int, nargs="+", default=[None], help="override NSEG")
    p.add_argument(
        "--block", type=int, nargs="+", default=[None], help="override threads/WG"
    )
    p.add_argument(
        "--kpw", type=int, nargs="+", default=[None], help="override keys/wave/tile"
    )
    p.add_argument(
        "--msplit", type=int, nargs="+", default=[None], help="waves sharing MAXM"
    )
    p.add_argument(
        "--ilv", type=int, nargs="+", default=[None], help="0 = contiguous KV runs"
    )
    p.add_argument(
        "--no-fusedred",
        dest="fusedred",
        action="store_false",
        help="launch reduce_segments instead of finishing in-kernel",
    )
    p.add_argument("--triton", action="store_true", help="also measure Triton")
    args = p.parse_args()

    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    import vllm.v1.attention.backends.rdna35_hip_attn as backend_mod
    from vllm.v1.attention.ops.rdna35_hip_decode import (
        KernelVariant,
        load,
        make_scratch,
        precompile,
    )

    knobs = ("nseg", "block", "kpw", "msplit", "ilv")
    combos = [
        {k: v for k, v in zip(knobs, vals, strict=True) if v is not None}
        for vals in itertools.product(*(getattr(args, k) for k in knobs))
    ]
    if not args.fusedred:
        for c in combos:
            c["fusedred"] = False

    # One row per combination, one column per context.  Passing several values
    # per knob builds the matrix here rather than in a shell loop: a loop pays
    # interpreter startup and a serial build for every cell, while this one
    # compiles the whole product at once and measures in a single process.
    original = backend_mod.Rdna35HipAttentionImpl._prepare
    scratch: dict[Any, Any] = {}
    current: dict[str, Any] = {}

    def cached_scratch(variant, device):
        if variant not in scratch:
            scratch[variant] = make_scratch(variant, device)
        return scratch[variant]

    # An override the backend would not have chosen leaves self._variant
    # permanently disagreeing with what _prepare computes, so it rebuilds every
    # forward; make_scratch zeroes the arrival counters, which is a device
    # memset, i.e. an extra kernel launch inside the timed region.
    backend_mod.make_scratch = cached_scratch

    def patched(self, kv_cache, **kw):
        built = original(self, kv_cache, **kw)
        if built is not None and current:
            want = replace(self._variant, **current)
            if self._variant != want:
                self._variant = want
                self._built = (load(want), cached_scratch(want, kw["q"].device))
            built = self._built
        return built

    backend_mod.Rdna35HipAttentionImpl._prepare = patched

    wanted = []
    for combo in combos:
        for layout in (0, 1):
            with contextlib.suppress(Exception):
                wanted.append(
                    KernelVariant(
                        args.head_dim,
                        args.hq,
                        args.hkv,
                        args.m,
                        args.block_size,
                        layout,
                        **{k: v for k, v in combo.items() if k != "fusedred"},
                        **({"fusedred": False} if not args.fusedred else {}),
                    )
                )
    precompile(wanted)
    time.sleep(5)  # clocks settle after a parallel build; see 00-protocol.md

    kv_per_token = 2 * args.hkv * args.head_dim * 2

    def timeit(backend, s):
        cfg = BenchmarkConfig(
            backend=backend,
            batch_spec=f"q{args.m}s{s}",
            num_layers=10,
            min_working_set_mb=96,
            head_dim=args.head_dim,
            num_q_heads=args.hq,
            num_kv_heads=args.hkv,
            block_size=args.block_size,
            device="cuda:0",
        )
        # Discarded: the build lands inside do_bench's own calibration, which
        # sizes n_repeat from it, so the whole median comes out wrong.
        run_attention_benchmark(cfg)
        return statistics.median(
            run_attention_benchmark(cfg).median_time * 1e6 for _ in range(args.reps)
        )

    label_w = max(24, *(len(_label(c)) for c in combos))
    print(f"Hq={args.hq} Hkv={args.hkv} D={args.head_dim} M={args.m}")
    print(f"{'':<{label_w}} " + " ".join(f"{s:>9}" for s in args.contexts))
    roofs = [s * kv_per_token / (PEAK_GIBS * 1024**3) * 1e6 for s in args.contexts]
    print(f"{'roofline':<{label_w}} " + " ".join(f"{r:>9.2f}" for r in roofs))
    if args.triton:
        current.clear()
        tri = [timeit("TRITON_ATTN", s) for s in args.contexts]
        print(f"{'TRITON_ATTN':<{label_w}} " + " ".join(f"{t:>9.2f}" for t in tri))
    for combo in combos:
        current.clear()
        current.update(combo)
        try:
            us = [timeit("RDNA35_HIP_ATTN", s) for s in args.contexts]
        except Exception as exc:  # noqa: BLE001 - a combination may not build
            print(f"{_label(combo):<{label_w}} {exc.__class__.__name__}")
            continue
        print(f"{_label(combo):<{label_w}} " + " ".join(f"{u:>9.2f}" for u in us))


def _label(combo) -> str:
    if not combo:
        return "backend defaults"
    return " ".join(f"{k}={v}" for k, v in combo.items())


if __name__ == "__main__":
    main()
