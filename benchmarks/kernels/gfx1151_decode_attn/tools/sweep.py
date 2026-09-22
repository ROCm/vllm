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
import statistics
import sys
from dataclasses import replace
from pathlib import Path

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
    p.add_argument("--nseg", type=int, default=None, help="override NSEG")
    p.add_argument("--block", type=int, default=None, help="override threads/WG")
    p.add_argument("--kpw", type=int, default=None, help="override keys/wave/tile")
    p.add_argument("--msplit", type=int, default=None, help="waves sharing MAXM")
    p.add_argument("--triton", action="store_true", help="also measure Triton")
    args = p.parse_args()

    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    import vllm.v1.attention.backends.rdna35_hip_attn as backend_mod
    from vllm.v1.attention.ops.rdna35_hip_decode import load, make_scratch

    overrides = {
        k: v
        for k, v in (
            ("nseg", args.nseg),
            ("block", args.block),
            ("kpw", args.kpw),
            ("msplit", args.msplit),
        )
        if v is not None
    }
    if overrides:
        original = backend_mod.Rdna35HipAttentionImpl._prepare

        def patched(self, kv_cache, **kw):
            built = original(self, kv_cache, **kw)
            if built is not None and any(
                getattr(self._variant, k) != v for k, v in overrides.items()
            ):
                self._variant = replace(self._variant, **overrides)
                built = (
                    load(self._variant),
                    make_scratch(self._variant, kw["q"].device),
                )
                self._built = built
            return built

        backend_mod.Rdna35HipAttentionImpl._prepare = patched

    backends = (
        ["TRITON_ATTN", "RDNA35_HIP_ATTN"] if args.triton else ["RDNA35_HIP_ATTN"]
    )
    kv_per_token = 2 * args.hkv * args.head_dim * 2

    print(
        f"Hq={args.hq} Hkv={args.hkv} D={args.head_dim} M={args.m} "
        f"{overrides or 'backend defaults'}"
    )
    print(f"{'S':>7} {'roofline':>9} " + " ".join(f"{b:>24}" for b in backends))
    for s in args.contexts:
        roof = s * kv_per_token / (PEAK_GIBS * 1024**3) * 1e6
        cells = []
        for backend in backends:
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
            # Discarded: the kernel is compiled on its first _prepare, which
            # lands inside do_bench's own calibration.  That does not just add
            # time to a sample -- do_bench sizes n_repeat from it, so a build
            # measuring seconds collapses the run to a single timed iteration
            # and the whole median is wrong.  Seen as a 66.90 us "median" next
            # to a true 7.58.  Only the first context of a config is at risk,
            # which is the short one the tuning cares about most.
            run_attention_benchmark(cfg)
            times = [
                run_attention_benchmark(cfg).median_time * 1e6 for _ in range(args.reps)
            ]
            t = statistics.median(times)
            cells.append(
                f"{t:>9.2f} [{min(times):.2f}-{max(times):.2f}] {roof / t * 100:>4.1f}%"
            )
        print(f"{s:>7} {roof:>9.2f} " + " ".join(f"{c:>24}" for c in cells))


if __name__ == "__main__":
    main()
