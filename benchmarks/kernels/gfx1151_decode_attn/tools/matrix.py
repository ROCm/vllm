#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_ATTN's RDNA3.5 decode kernel on every shape, as markdown.

By default one sequence of --m tokens at each of --contexts, per configuration
of shapes.csv; --batch measures batch specs instead (attention_benchmarks
grammar, or `decode` / `mixed` for the lists below): batches of sequences
decoding together, and decodes mixed with a prefill or a chunked-prefill
extend (decodes on the kernel, the rest on Triton).  Specs whose requests are
all decodes are timed under CUDA graphs, as vLLM replays them; mixed ones
eagerly, as vLLM runs them.

Only our kernel runs unless --baselines names other backends.  It is the one
_rocm_C carries; forcing a knob (--max-segments ...) or --ablate builds variants with
jit.py instead.  The path column says what ran: `kernel`, `split` (decodes on
the kernel, the rest on Triton) or `triton` (fallback).  Every backend runs
in the KV cache layout vLLM gives it: ours LBHNC, TRITON_ATTN LBNHC,
ROCM_SEGMENTED_ATTN LHBNC.

    cd <worktree> && PYTHONPATH=$PWD <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py > table.md
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
BASELINES = ("TRITON_ATTN", "ROCM_SEGMENTED_ATTN", "ROCM_SEGMENTED_ATTN@autotune")
SPECS = {
    "decode": ["2q1s4k", "8q1s4k", "8q4s4k", "32q1s1k", "64q1s1k", "32q4s1k"],
    "mixed": ["4q1s8k_q512", "16q1s4k_2q1k", "16q1s2k_q64s2k", "32q1s1k_q64s1k"],
}


class Shape(NamedTuple):
    model: str
    hq: int
    hkv: int
    d: int
    window: int


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shapes", default=str(_SHAPES))
    p.add_argument("--filter", default="", help="substring match on the model name")
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
    """The shapes the kernel can serve, after the filters in `args`.

    Returns the surviving shapes and how many sliding-window rows were dropped.
    The window is checked last so that the count describes the selection rather
    than the whole file.

    Args:
        args: Namespace populated by `add_arguments`.

    Returns:
        A `(shapes, windowed)` pair.

    """
    shapes: list[Shape] = []
    windowed = 0
    with open(args.shapes) as fh:
        for r in csv.DictReader(fh):
            s = Shape(
                r["model"],
                int(r["Hq"]),
                int(r["Hkv"]),
                int(r["D"]),
                int(r["window"]),
            )
            if args.filter and args.filter not in s.model:
                continue
            if args.hq and s.hq not in args.hq:
                continue
            if args.hkv and s.hkv not in args.hkv:
                continue
            if args.head_dim and s.d not in args.head_dim:
                continue
            # A ratio that is not a whole number matches no integer --gqa, which
            # is the wanted answer rather than a rounding decision.
            if args.gqa and (s.hq % s.hkv or s.hq // s.hkv not in args.gqa):
                continue
            if bool(s.window) != getattr(args, "windowed", False):
                windowed += bool(s.window)
                continue
            shapes.append(s)
    return shapes, windowed


def describe(args: argparse.Namespace) -> str:
    """The active filters, for tools that print what they measured."""
    bits = []
    for name, val in (
        ("model~", args.filter),
        ("Hq", args.hq),
        ("Hkv", args.hkv),
        ("D", args.head_dim),
        ("Hq/Hkv", args.gqa),
    ):
        if val:
            joined = val if isinstance(val, str) else ",".join(str(v) for v in val)
            bits.append(f"{name}={joined}")
    return " ".join(bits)


def spec_roofline_us(
    spec: str,
    hq: int,
    hkv: int,
    d: int,
    itemsize: int = 2,
    block_size: int = 16,
    window: int = 0,
) -> float | None:
    """`utils.roofline_us` of a batch spec: one dispatch plus every sequence's bytes.

    None if a request is a prefill, whose floor is compute rather than bytes.
    """
    from batch_spec import parse_batch_spec

    reqs = parse_batch_spec(spec)
    if any(r.q_len > 8 for r in reqs):
        return None
    moved = sum(
        utils.roofline_us(hq, hkv, d, r.q_len, r.kv_len, itemsize, block_size, window)
        - utils.DISPATCH_US
        for r in reqs
    )
    return utils.DISPATCH_US + moved


def main() -> None:
    p = argparse.ArgumentParser()
    add_arguments(p)
    utils.add_dtype_argument(p)
    p.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1, 4],
        help="query tokens per sequence; 1 is plain decode, 4 is speculative",
    )
    p.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 4096, 8192, 16384, 32768],
    )
    p.add_argument(
        "--batch",
        nargs="+",
        help="batch specs instead of --m x --contexts; `decode` and `mixed` "
        "stand for the lists in SPECS",
    )
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="whole re-setups per cell; do_bench medians many iterations "
        "inside one already",
    )
    for knob in utils.KNOBS:
        p.add_argument(
            f"--{knob.replace('_', '-')}",
            type=int,
            help=f"force {knob.upper()} on every configuration (built by jit.py)",
        )
    p.add_argument(
        "--ablate",
        type=int,
        default=0,
        help="MEASUREMENT ONLY, wrong numbers; see ABLATE in the kernel",
    )
    p.add_argument(
        "--baselines",
        nargs="+",
        default=[],
        choices=BASELINES,
        help="backends to measure alongside ours, one column each; none by "
        "default, %%roof does not need them and each one is another full run",
    )
    args = p.parse_args()
    dtype = utils.torch_dtype(args.dtype)

    import jit
    from batch_spec import parse_batch_spec
    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    specs = (
        [s for b in args.batch for s in SPECS.get(b, [b])]
        if args.batch
        else [f"q{m}s{s}" for s in args.contexts for m in args.m]
    )
    parsed = {spec: parse_batch_spec(spec) for spec in specs}
    # One limit for the whole run, so that a backend's startup autotuning
    # (keyed on it) runs once per configuration rather than once per cell.
    max_len = max(r.kv_len for reqs in parsed.values() for r in reqs)

    # Grouped by configuration, not listed by model: the kernel cannot tell
    # two models with the same (Hq, Hkv, D, window) apart.
    shapes, windowed = load(args)
    if not shapes:
        print(f"<!-- no shapes match {describe(args) or 'the filters'} -->")
        return
    groups: dict[tuple[int, int, int, int], list[str]] = {}
    for sh in shapes:
        groups.setdefault((sh.hq, sh.hkv, sh.d, sh.window), []).append(sh.model)

    forced = {k: getattr(args, k) for k in utils.KNOBS if getattr(args, k) is not None}
    if forced or args.ablate:
        jit.install(ablate=args.ablate)
        wanted = []
        for hq, hkv, d, window in groups:
            for reqs in parsed.values():
                dec = [r for r in reqs if r.q_len <= 8]
                if dec and all(r.q_len == dec[0].q_len for r in dec):
                    with jit.override(forced):
                        wanted.append(
                            jit.variant_for(
                                hq,
                                hkv,
                                d,
                                dec[0].q_len,
                                args.block_size,
                                1,
                                window,
                                dtype,
                                len(dec) > 1,
                            )
                        )
        jit.precompile(wanted)
        jit.seal()
        time.sleep(5)  # a parallel build moves the SoC clock

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
        f"<!-- {len(groups)} configurations cover {len(shapes)} models; "
        f"{windowed} sliding-window rows skipped; {args.dtype}, block size "
        f"{args.block_size}"
        f"{'; filters: ' + active if active else ''}"
        f"{'; forced: ' + knobs if knobs else ''}"
        f"{f'; ABLATE={args.ablate}, wrong numbers' if args.ablate else ''} -->"
    )
    # "vs" is against the fastest baseline of the cell, so with several it
    # answers whether we beat everything else, not just one of them.
    columns = [
        "models",
        "spec",
        "Hq",
        "Hkv",
        "D",
        "window",
        "path",
        "roofline",
        *args.baselines,
        "ours",
        *(["vs"] if args.baselines else []),
        "%roof",
        "spread",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| --- " * len(columns) + "|")
    # Spec-major, configuration last: every configuration side by side at the
    # same context.  The contexts also go upward across the whole run, which
    # warms the allocator: a pass starting at a large context reads its first
    # cell per configuration up to 10x slow.
    for spec in specs:
        for (hq, hkv, d, window), models in groups.items():
            label = models[0] if len(models) == 1 else f"{models[0]} +{len(models) - 1}"
            roof = spec_roofline_us(
                spec, hq, hkv, d, block_size=args.block_size, window=window
            )
            head = [label, spec, hq, hkv, d, window or "-"]
            try:
                with jit.paths() as path, jit.override(forced):
                    ours, spread = timeit("ROCM_ATTN", spec, hq, hkv, d, window)
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
                "-" if roof is None else f"{roof:.2f}",
                *("-" if t is None else f"{t:.2f}" for t in base),
                f"{ours:.2f}",
                *([f"{min(served) / ours:.2f}x" if served else "-"] if base else []),
                "-" if roof is None else f"{roof / ours * 100:.1f} %",
                f"{spread:.1f} %",
            ]
            print("| " + " | ".join(map(str, cells)) + " |", flush=True)


if __name__ == "__main__":
    main()
