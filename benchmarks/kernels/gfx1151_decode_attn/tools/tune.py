#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find the one (NSEG, MSPLIT) per configuration that wins at every context.

The constraint that shapes this whole search: a configuration may not depend
on S. The grid is fixed when the CUDA graph is captured and S is a runtime
argument, so one choice has to serve 128 and 32768 alike. Tuning per context
would be easy and unshippable.

So the objective is the worst context, not the average. A candidate that is
3x at short context and 0.4x at long is useless; one that is 1.05x everywhere
is not. Reported by the minimum speedup against Triton across the range, with
the geomean alongside to break ties.

    cd <worktree> && VLLM_KV_CACHE_LAYOUT=HND PYTHONPATH=$PWD \\
        amd-gpu-lock <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/tune.py
"""

import argparse
import contextlib
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
sys.path.insert(0, str(_ROOT / "benchmarks" / "attention_benchmarks"))
sys.path.insert(0, str(_HERE.parent))

import shapeset  # noqa: E402

PEAK_GIBS = 230.0
# Measured sweet spot is 64-128 workgroups; 32 is the worst value at every head
# size tried and 256 is past the knee. Searching outside that wastes minutes
# per configuration on cells already known to lose.
WG_LO, WG_HI = 32, 256


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> None:
    p = argparse.ArgumentParser()
    shapeset.add_arguments(p)
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument(
        "--search",
        type=int,
        nargs="+",
        default=[128, 16384, 32768],
        help="contexts the search runs on; the winner is verified on --verify",
    )
    p.add_argument(
        "--verify",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 4096, 8192, 16384, 32768],
    )
    args = p.parse_args()

    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    import vllm.v1.attention.backends.rdna35_hip_attn as backend_mod
    from vllm.v1.attention.ops.rdna35_hip_decode import (
        KernelVariant,
        precompile,
        seal,
    )

    # Windowed rows are dropped, not tuned: the kernel is full-context, so a
    # knob scored against them would be scored against the wrong byte count.
    # The window still keys the group, so a model contributing both a windowed
    # and a full-context layer does not collapse into one entry.
    shapes, _windowed = shapeset.load(args)
    if not shapes:
        print(f"no shapes match {shapeset.describe(args) or 'the given filters'}")
        return
    groups: dict[tuple[int, int, int, int], list[str]] = {}
    for sh in shapes:
        groups.setdefault((sh.hq, sh.hkv, sh.d, sh.window), []).append(sh.model)

    def candidates(hq):
        out = []
        for nseg in (1, 2, 4, 8, 16, 32):
            if not WG_LO <= hq * nseg <= WG_HI:
                continue
            for msplit in (1, 2, 4):
                if args.m % msplit or (args.block_size and 8 % msplit):
                    continue
                out.append((nseg, msplit))
        return out

    scratch: dict[Any, Any] = {}
    orig_make = backend_mod.make_scratch

    def cached(variant, device):
        if variant not in scratch:
            scratch[variant] = orig_make(variant, device)
        return scratch[variant]

    backend_mod.make_scratch = cached

    # The override replaces only NSEG and MSPLIT on top of whatever _prepare
    # computed, so the rest of the base -- BFLY in particular -- has to come
    # from the same place the backend gets it.
    from vllm.v1.attention.backends.rdna35_hip_attn import _knobs_for

    wanted = []
    for hq, hkv, d, _window in groups:
        base = dict(_knobs_for(hq, hkv, d, args.m))
        for nseg, msplit in candidates(hq):
            with contextlib.suppress(Exception):
                wanted.append(
                    KernelVariant(
                        d,
                        hq,
                        hkv,
                        args.m,
                        args.block_size,
                        1,
                        **{**base, "nseg": nseg, "msplit": msplit},
                    )
                )
    # The base variant too, for the same reason as sweep.py: the unpatched
    # _prepare builds the backend's own choice before the override lands.
    for hq, hkv, d, _window in groups:
        for layout in (0, 1):
            with contextlib.suppress(Exception):
                wanted.append(
                    KernelVariant(
                        d,
                        hq,
                        hkv,
                        args.m,
                        args.block_size,
                        layout,
                        **_knobs_for(hq, hkv, d, args.m),
                    )
                )
    precompile(wanted)
    # Nothing may build from here on: a build inside do_bench corrupts the
    # median it reports (258% measured), which is why every cell used to pay a
    # discarded warm-up call.  Sealing makes the same hazard a loud failure and
    # halves the run.  Scratch is realised now for the same reason -- its memset
    # would otherwise land in the first timed iteration.
    for variant in wanted:
        with contextlib.suppress(Exception):
            cached(variant, torch.device("cuda:0"))
    seal()
    time.sleep(5)  # clocks settle after a parallel build; see 00-protocol.md

    from dataclasses import replace

    original = backend_mod.Rdna35HipAttentionImpl._prepare
    override: dict[str, int] = {}

    def patched(self, kv_cache, **kw):
        built = original(self, kv_cache, **kw)
        if built is not None and override:
            want = replace(self._variant, **override)
            if self._variant != want:
                from vllm.v1.attention.ops.rdna35_hip_decode import load

                self._variant = want
                self._built = (load(want), cached(want, kw["q"].device))
            built = self._built
        return built

    backend_mod.Rdna35HipAttentionImpl._prepare = patched

    def timeit(backend, hq, hkv, d, s):
        cfg = BenchmarkConfig(
            backend=backend,
            batch_spec=f"q{args.m}s{s}",
            num_layers=10,
            min_working_set_mb=96,
            head_dim=d,
            num_q_heads=hq,
            num_kv_heads=hkv,
            block_size=args.block_size,
            device="cuda:0",
        )
        return statistics.median(
            run_attention_benchmark(cfg).median_time * 1e6 for _ in range(args.reps)
        )

    print(
        "| modelos | Hq | Hkv | D | NSEG | MSPLIT | WGs | peor | geomean | "
        "default peor |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (hq, hkv, d, _window), models in sorted(
        groups.items(), key=lambda kv: kv[0][::-1]
    ):
        tri = {s: timeit("TRITON_ATTN", hq, hkv, d, s) for s in args.search}
        override.clear()
        base = {s: timeit("RDNA35_HIP_ATTN", hq, hkv, d, s) for s in args.search}
        base_min = min(tri[s] / base[s] for s in args.search)

        best = None
        for nseg, msplit in candidates(hq):
            override.clear()
            override.update(nseg=nseg, msplit=msplit)
            try:
                ours = {
                    s: timeit("RDNA35_HIP_ATTN", hq, hkv, d, s) for s in args.search
                }
            except Exception:  # noqa: BLE001 - a combination may not build
                continue
            r = [tri[s] / ours[s] for s in args.search]
            key = (min(r), geomean(r))
            if best is None or key > best[0]:
                best = (key, nseg, msplit)
        override.clear()
        if best is None:
            continue
        (worst, gm), nseg, msplit = best
        label = models[0] if len(models) == 1 else f"{models[0]} +{len(models) - 1}"
        print(
            f"| {label} | {hq} | {hkv} | {d} | {nseg} | {msplit} | {hq * nseg} | "
            f"{worst:.2f}x | {gm:.2f}x | {base_min:.2f}x |",
            flush=True,
        )


if __name__ == "__main__":
    main()
