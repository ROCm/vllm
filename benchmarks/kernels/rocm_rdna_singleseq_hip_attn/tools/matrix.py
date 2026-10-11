#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time ROCM_RDNA_SINGLESEQ_ATTN on the shapes of shapes.csv, as markdown.

Decode specs run under CUDA graphs, prefill specs without, as in vLLM.
`path` is what ran (`kernel`, `prefill` or `triton`); %roof is against
utils.roofline. Forcing a knob or --ablate builds the variants with jit.py.

    PYTHONPATH=$PWD python \\
        benchmarks/kernels/rocm_rdna_singleseq_hip_attn/tools/matrix.py
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import NamedTuple

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "attention_benchmarks"))
sys.path.insert(0, str(_HERE.parent))

import utils  # noqa: E402

_SHAPES = _HERE.parent / "shapes.csv"
BASELINES = (
    "TRITON_ATTN",
    "ROCM_SEGMENTED_ATTN",
    "ROCM_SEGMENTED_ATTN@autotune",
    "ROCM_ATTN",
)


class Shape(NamedTuple):
    hq: int
    hkv: int
    d: int
    window: int


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shapes", default=str(_SHAPES))
    p.add_argument("--hq", type=int, nargs="+", help="keep only these Hq")
    p.add_argument("--hkv", type=int, nargs="+", help="keep only these Hkv")
    p.add_argument("--head-dim", type=int, nargs="+", help="keep only these D")
    p.add_argument("--gqa", type=int, nargs="+", help="keep only these Hq/Hkv")
    p.add_argument(
        "--windowed",
        action="store_true",
        help="the sliding-window rows instead of the full-attention ones",
    )


def load(args: argparse.Namespace) -> tuple[list[Shape], int]:
    """The shapes left after the filters in `args`.

    Returns:
        (shapes, number of sliding-window rows skipped).

    """
    shapes: list[Shape] = []
    windowed = 0
    with open(args.shapes) as fh:
        for r in csv.DictReader(fh):
            s = Shape(int(r["Hq"]), int(r["Hkv"]), int(r["D"]), int(r["window"]))
            if args.hq and s.hq not in args.hq:
                continue
            if args.hkv and s.hkv not in args.hkv:
                continue
            if args.head_dim and s.d not in args.head_dim:
                continue
            if args.gqa and (s.hq % s.hkv or s.hq // s.hkv not in args.gqa):
                continue
            if bool(s.window) != getattr(args, "windowed", False):
                windowed += bool(s.window)
                continue
            shapes.append(s)
    return shapes, windowed


def describe(args: argparse.Namespace) -> str:
    """The active filters, as text."""
    bits = []
    for name, val in (
        ("Hq", args.hq),
        ("Hkv", args.hkv),
        ("D", args.head_dim),
        ("Hq/Hkv", args.gqa),
    ):
        if val:
            bits.append(f"{name}={','.join(str(v) for v in val)}")
    return " ".join(bits)


def spec_roofline(
    spec: str,
    hq: int,
    hkv: int,
    d: int,
    itemsize: int = 2,
    block_size: int = 16,
    window: int = 0,
    realistic: bool = False,
) -> tuple[float, float, str]:
    """`utils.roofline` of a batch spec: (us, FLOP/byte, bound)."""
    from batch_spec import parse_batch_spec

    reqs = parse_batch_spec(spec)
    args = (itemsize, block_size, window)
    flops = sum(utils.attention_flops(hq, d, r.q_len, r.kv_len, window) for r in reqs)
    moved = sum(
        utils.attention_bytes(hq, hkv, d, r.q_len, r.kv_len, *args) for r in reqs
    )
    memory_us = moved / (utils.PEAK_GIBS * 1024**3) * 1e6
    compute_us = sum(
        utils.attention_compute_us(hq, hkv, d, r.q_len, r.kv_len, window, realistic)
        for r in reqs
    )
    bound = "compute" if compute_us > memory_us else "memory"
    fixed = utils.DISPATCH_US + (3 * utils.DRAM_LATENCY_US if realistic else 0)
    return fixed + max(memory_us, compute_us), flops / moved, bound


def main() -> None:
    p = argparse.ArgumentParser()
    add_arguments(p)
    utils.add_dtype_argument(p)
    p.add_argument(
        "--m",
        type=int,
        nargs="+",
        help="query tokens per sequence (default 1 4; with --prefill "
        "128 256 512 1024 2048 4096 8192)",
    )
    p.add_argument(
        "--prefill",
        action="store_true",
        help="one sequence of --m new tokens after each of --prefixes cached ones",
    )
    p.add_argument("--prefixes", type=int, nargs="+", default=[0, 4096, 16384])
    p.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 4096, 8192, 16384, 32768],
    )
    p.add_argument(
        "--batch",
        nargs="+",
        help="batch specs instead of --m x --contexts",
    )
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="repetitions per cell",
    )
    for knob in dict.fromkeys(utils.KNOBS + utils.PREFILL_KNOBS):
        p.add_argument(
            f"--{knob.replace('_', '-')}",
            type=int,
            help=f"force {knob.upper()} on every configuration (built by jit.py)",
        )
    p.add_argument(
        "--ablate",
        type=int,
        default=0,
        help="profiling only, wrong results; see ABLATE in the kernel",
    )
    p.add_argument(
        "--baselines",
        nargs="+",
        default=[],
        choices=BASELINES,
        help="other backends to time, one column each",
    )
    args = p.parse_args()
    dtype = utils.torch_dtype(args.dtype)

    import jit
    from batch_spec import parse_batch_spec
    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    if args.m is None:
        args.m = [128, 256, 512, 1024, 2048, 4096, 8192] if args.prefill else [1, 4]
    if args.batch:
        specs = args.batch
    elif args.prefill:
        specs = [f"q{m}s{m + p}" for p in args.prefixes for m in args.m]
    else:
        specs = [f"q{m}s{s}" for s in args.contexts for m in args.m]
    parsed = {spec: parse_batch_spec(spec) for spec in specs}
    # One max_model_len for the run, so autotuning runs once per configuration.
    max_len = max(r.kv_len for reqs in parsed.values() for r in reqs)

    shapes, windowed = load(args)
    if not shapes:
        print(f"<!-- no shapes match {describe(args) or 'the filters'} -->")
        return

    names = utils.PREFILL_KNOBS if args.prefill else utils.KNOBS
    forced = {k: getattr(args, k) for k in names if getattr(args, k) is not None}
    if forced or args.ablate:
        jit.install(ablate=args.ablate)
        wanted = []
        for hq, hkv, d, window in shapes:
            for reqs in parsed.values():
                for r in reqs:
                    if r.q_len > 8:
                        with jit.override(forced):
                            v = jit.prefill_variant_for(
                                hq, hkv, d, args.block_size, 1, window, dtype, r.q_len
                            )
                        if v is not None:
                            wanted.append(v)
                if len(reqs) == 1 and reqs[0].q_len <= 8:
                    with jit.override(forced):
                        wanted.append(
                            jit.variant_for(
                                hq,
                                hkv,
                                d,
                                reqs[0].q_len,
                                args.block_size,
                                1,
                                window,
                                dtype,
                            )
                        )
        jit.precompile(wanted)
        jit.seal()
        time.sleep(5)  # let clocks settle after the build

    def timeit(backend, spec, hq, hkv, d, window):
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
            use_cuda_graphs=all(r.q_len <= 8 for r in parsed[spec]),
            max_model_len=max_len,
        )
        rs = [run_attention_benchmark(cfg) for _ in range(args.reps)]
        spread = max(r.std_time / r.median_time for r in rs) * 100
        return statistics.median(r.median_time for r in rs) * 1e6, spread

    active = describe(args)
    knobs = " ".join(f"{k}={v}" for k, v in forced.items())
    print(
        f"<!-- {len(shapes)} configurations; "
        f"{windowed} sliding-window rows skipped; {args.dtype}, block size "
        f"{args.block_size}"
        f"{'; filters: ' + active if active else ''}"
        f"{'; forced: ' + knobs if knobs else ''}"
        f"{f'; ABLATE={args.ablate}, wrong numbers' if args.ablate else ''} -->"
    )
    # "vs" is against the fastest baseline.
    columns = [
        "spec",
        "Hq",
        "Hkv",
        "D",
        "window",
        "path",
        "AI",
        "bound",
        "roofline",
        *args.baselines,
        "ours",
        *(["vs"] if args.baselines else []),
        "%roof",
        "spread",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| --- " * len(columns) + "|")
    for spec in specs:
        for hq, hkv, d, window in shapes:
            roof, intensity, bound = spec_roofline(
                spec,
                hq,
                hkv,
                d,
                block_size=args.block_size,
                window=window,
                realistic=args.prefill,
            )
            head = [spec, hq, hkv, d, window or "-"]
            try:
                with jit.paths() as path, jit.override(forced):
                    ours, spread = timeit(
                        "ROCM_RDNA_SINGLESEQ_ATTN", spec, hq, hkv, d, window
                    )
            except Exception as exc:  # noqa: BLE001 - a forced knob may not build
                print(
                    f"| {' | '.join(map(str, head))} | {type(exc).__name__}: "
                    f"{str(exc).splitlines()[0]} |",
                    flush=True,
                )
                continue
            base = []
            for b in args.baselines:
                try:
                    base.append(timeit(b, spec, hq, hkv, d, window)[0])
                except Exception:  # noqa: BLE001 - the backend refuses the shape
                    base.append(None)
            served = [t for t in base if t is not None]
            cells = [
                *head,
                path(),
                f"{intensity:.0f}",
                bound,
                f"{roof:.2f}",
                *("-" if t is None else f"{t:.2f}" for t in base),
                f"{ours:.2f}",
                *([f"{min(served) / ours:.2f}x" if served else "-"] if base else []),
                f"{roof / ours * 100:.1f} %",
                f"{spread:.1f} %",
            ]
            print("| " + " | ".join(map(str, cells)) + " |", flush=True)


if __name__ == "__main__":
    main()
