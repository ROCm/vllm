#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading and filtering shapes.csv.

Shared by roofline.py, matrix.py and tune.py so that `--hkv 8` selects the same
rows whichever tool is asked. Selecting by hand instead -- editing the csv, or
passing a model substring that happens to cover the shapes you meant -- is how
a pass ends up measuring a different set than the one it reports.
"""

import argparse
import csv
from pathlib import Path
from typing import NamedTuple

_SHAPES = Path(__file__).resolve().parent / "shapes.csv"

# Measured peak DRAM bandwidth, LPDDR5X-8000 on Strix Halo: 256 GiB/s
# theoretical, 230 GiB/s achieved by a tuned streaming kernel.
PEAK_GIBS = 230.0

# Per-kernel dispatch overhead on the 8060S, measured under a HIP graph with a
# single-wave spin, so it is the floor even when the launch is already graphed
# and not an artifact of eager dispatch.  Attention cannot be computed without
# dispatching at least one kernel, so it belongs in an algorithmic floor.
DISPATCH_US = 1.48


def roofline_us(
    hq: int,
    hkv: int,
    d: int,
    m: int,
    s: int,
    itemsize: int = 2,
    block_size: int = 16,
) -> float:
    """The floor for one decode-attention call, in microseconds.

    Deliberately algorithmic. It counts what the problem requires -- one kernel
    dispatch, the query in, the KV cache in, the output out, and the block
    table that addresses the pages -- and nothing an implementation chose. Our
    own split-KV partials go through global memory and can exceed the KV
    traffic at short context, but they are excluded on purpose: a denominator
    that grew with a kernel's own overhead would let a wasteful kernel report
    itself near the roof. One dispatch is counted rather than NSEG of them, for
    the same reason -- the decomposition is ours, the single launch is not.

    The dispatch term dominates at short context: at S=128 it is more than the
    bytes for every shape in the table, which is why %roof there used to read
    absurdly low. So this is a floor for any implementation, not a model of
    ours.

    Args:
        hq: Query heads.
        hkv: KV heads.
        d: Head dimension.
        m: Query tokens per sequence.
        s: Context length.
        itemsize: Bytes per element of Q, K, V and the output.
        block_size: KV cache page size, for the block table.

    Returns:
        One dispatch plus the time those bytes take at peak bandwidth, in
        microseconds.
    """
    query = m * hq * d * itemsize
    kv = s * hkv * 2 * d * itemsize
    out = m * hq * d * itemsize
    table = -(-s // block_size) * 4
    moved = (query + kv + out + table) / (PEAK_GIBS * 1024**3) * 1e6
    return DISPATCH_US + moved


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


def add_dtype_argument(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="element type of Q, the KV cache and the output",
    )


def torch_dtype(name: str):
    """The torch dtype for a `--dtype` value."""
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


# Relative error bound check.py applies per dtype; bf16 output rounding alone
# is up to 2^-8 relative.
RTOL = {"fp16": 1e-3, "bf16": 8e-3}


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
            if s.window:
                windowed += 1
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
