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

# Sustained WMMA 16x16x16 f16/bf16 throughput achieved on gfx1151, TFLOPS
PEAK_TFLOPS = 43.0

# Arithmetic intensity (FLOP/byte) where the two roofs meet: below it a call is
# memory-bound, above it compute-bound.  ~174.
RIDGE = PEAK_TFLOPS * 1e12 / (PEAK_GIBS * 1024**3)

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

# The prefill kernel's knobs (rdna35_prefill_variants.csv), likewise.
PREFILL_KNOBS = (
    "waves",
    "key_tile",
    "head_dim_split",
    "v_t",
    "double_buffer",
    "key_waves",
    "max_segments",
    "head_group",
)


def _visible_keys(m: int, s: int, window: int) -> int:
    """Keys the m queries of a sequence of length s see, summed over them:
    causal, and only the last `window` keys of each when window > 0."""
    prefix = s - m
    if not window:
        return m * prefix + m * (m + 1) // 2
    return sum(min(prefix + t + 1, window) for t in range(m))


def attention_flops(hq: int, d: int, m: int, s: int, window: int = 0) -> float:
    """FLOPs the call must do: Q@K and P@V over the keys each query sees."""
    return 4.0 * hq * d * _visible_keys(m, s, window)


def attention_bytes(
    hq: int,
    hkv: int,
    d: int,
    m: int,
    s: int,
    itemsize: int = 2,
    block_size: int = 16,
    window: int = 0,
) -> float:
    """Bytes the call must move: the query in, every key some query sees in
    once, the output out, and the block table that addresses the pages."""
    keys = s if not window else min(s, window + m - 1)
    query = m * hq * d * itemsize
    kv = keys * hkv * 2 * d * itemsize
    out = m * hq * d * itemsize
    table = -(-keys // block_size) * 4
    return query + kv + out + table


def roofline(
    hq: int,
    hkv: int,
    d: int,
    m: int,
    s: int,
    itemsize: int = 2,
    block_size: int = 16,
    window: int = 0,
) -> tuple[float, float, str]:
    """The floor for one attention call of one sequence.

    Deliberately algorithmic: it counts what the problem requires and nothing
    an implementation chose.  The call takes the longer of its bytes at peak
    bandwidth and its FLOPs at peak throughput, plus one dispatch: decode sits
    far below RIDGE, a prefill of a few hundred tokens above it.

    Args:
        hq: Query heads.
        hkv: KV heads.
        d: Head dimension.
        m: Query tokens.
        s: Sequence length, the m tokens included.
        itemsize: Bytes per element of Q, K, V and the output.
        block_size: KV cache page size, for the block table.
        window: Sliding window in keys, 0 for full attention.

    Returns:
        (microseconds, arithmetic intensity in FLOP/byte, "memory" or
        "compute": the roof that sets the floor).

    """
    flops = attention_flops(hq, d, m, s, window)
    moved = attention_bytes(hq, hkv, d, m, s, itemsize, block_size, window)
    memory_us = moved / (PEAK_GIBS * 1024**3) * 1e6
    compute_us = flops / (PEAK_TFLOPS * 1e6)
    bound = "compute" if compute_us > memory_us else "memory"
    return DISPATCH_US + max(memory_us, compute_us), flops / moved, bound


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
    """`roofline`'s floor alone, in microseconds."""
    return roofline(hq, hkv, d, m, s, itemsize, block_size, window)[0]


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
