#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The roofline the tools report against, the knobs they force, and --dtype.

Shared by matrix.py and tune.py.
"""

import argparse

# Assumed peak DRAM bandwidth
PEAK_GIBS = 230.0

# Assumed per-kernel dispatch overhead measured under a HIP graph
DISPATCH_US = 1.48

# The launch knobs of a variant that tune.py searches and matrix.py can force.
KNOBS = (
    "max_segments",
    "row_groups",
    "min_segment_blocks",
    "waves",
    "head_dim_split",
    "prefetch",
    "v_in_lds",
    "dot_product",
)


def roofline_us(
    hq: int,
    hkv: int,
    d: int,
    m: int,
    s: int,
    itemsize: int = 2,
    block_size: int = 16,
    window: int = 0,
) -> float:
    """The floor for one decode-attention call, in microseconds.

    Deliberately algorithmic. It counts what the problem requires -- one kernel
    dispatch, the query in, the KV cache in, the output out, and the block
    table that addresses the pages -- and nothing an implementation chose.

    Args:
        hq: Query heads.
        hkv: KV heads.
        d: Head dimension.
        m: Query tokens per sequence.
        s: Context length.
        itemsize: Bytes per element of Q, K, V and the output.
        block_size: KV cache page size, for the block table.
        window: Sliding window in keys, 0 for full attention.  A windowed layer
            needs only the keys some query can see: min(s, window + m - 1).

    Returns:
        One dispatch plus the time those bytes take at peak bandwidth, in
        microseconds.

    """
    keys = s if not window else min(s, window + m - 1)
    query = m * hq * d * itemsize
    kv = keys * hkv * 2 * d * itemsize
    out = m * hq * d * itemsize
    table = -(-keys // block_size) * 4
    moved = (query + kv + out + table) / (PEAK_GIBS * 1024**3) * 1e6
    return DISPATCH_US + moved


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
