#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic MoE routing-distribution generator + padding/fill space-map.

Covers the space of token->expert distributions and batch sizes that real
inputs can hit, so the kernel-vs-Triton benchmarks (``bench_archetype_map.py``,
``bench_gemm2.py``) can probe the win/loss surface keyed by the routing shape
and the number of valid tokens (num_valid = T*top_k).

``synth_topk(name, T)`` is importable: the benchmarks call it to build routing
in-process, so there is no on-disk dataset step. Running this file as a script
prints a per-archetype padding/fill space-map.

Archetypes (per-expert popularity over E experts):
  - balanced: deterministic round-robin, every expert gets an equal share.
  - uniform:  near-balanced random (Poisson variance around the mean).
  - zipf1:    moderate power-law skew (1/r).
  - zipf2:    heavy power-law skew (1/r^2), a few hot experts.
  - hot16:    near-collapse, ~16 hot experts take almost all tokens.

Block-fill model matches moe_align_block_size: an expert with c tokens occupies
ceil(c/BM) blocks; the first ceil-1 are full (BM rows), the last holds c%BM rows.
"""

import argparse  # plain argparse: this file is numpy-only, no torch/vllm import
from enum import Enum

import numpy as np

E, TOPK = 256, 8
TS = [16, 64, 128, 256, 512, 768, 994, 1024, 1536, 2048, 4096]


class Archetype(str, Enum):
    """Token->expert routing distributions, from flat to heavily skewed."""

    BALANCED = "balanced"  # deterministic round-robin, every expert equal
    UNIFORM = "uniform"  # near-balanced random (Poisson variance)
    ZIPF1 = "zipf1"  # moderate power-law skew (1/r)
    ZIPF2 = "zipf2"  # heavy power-law skew (1/r^2), a few hot experts
    HOT16 = "hot16"  # near-collapse, ~16 hot experts take almost all tokens


def popularity(arch):
    """Per-expert popularity weights over the E experts (len E), or None for
    the deterministic ``BALANCED`` archetype."""
    r = np.arange(1, E + 1)
    if arch is Archetype.UNIFORM:
        return np.ones(E)
    if arch is Archetype.ZIPF1:
        return 1.0 / r**1.0
    if arch is Archetype.ZIPF2:
        return 1.0 / r**2.0
    if arch is Archetype.HOT16:
        return np.where(r <= 16, 1.0, 1e-6)
    if arch is Archetype.BALANCED:
        return None
    raise ValueError(arch)


def synth_topk(arch, T, rng=None):
    """Per-token top_k distinct experts drawn from the archetype popularity.

    Returns an int32 ``[T, TOPK]`` array. ``rng`` lets callers fix the seed.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if arch is Archetype.BALANCED:
        # round-robin so every expert gets as equal a share as possible
        flat = np.tile(np.arange(E), (T * TOPK + E - 1) // E)[: T * TOPK]
        return flat.reshape(T, TOPK).astype(np.int32)
    logp = np.log(popularity(arch) + 1e-12)
    toks = np.empty((T, TOPK), dtype=np.int32)
    for t in range(T):  # gumbel-top-k = sample top_k w/o replacement ~ popularity
        g = rng.gumbel(size=E)
        toks[t] = np.argpartition(-(logp + g), TOPK)[:TOPK]
    return toks


def counts(toks):
    return np.bincount(toks.reshape(-1), minlength=E)


def pad_stats(cnt, BM):
    """useful% = real rows / padded rows; sk8% = fraction of blocks with <=8 fill."""
    used = cnt[cnt > 0]
    if used.size == 0:
        return dict(useful=0.0, sk8=0.0)
    blocks = -(-used // BM)  # ceil
    fills = []
    for c, b in zip(used, blocks):
        fills += [BM] * (b - 1) + [int(c - (b - 1) * BM)]
    fills = np.array(fills)
    padr = int(blocks.sum()) * BM
    return dict(useful=100 * int(cnt.sum()) / padr, sk8=100 * (fills <= 8).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arches", default=",".join(a.value for a in Archetype))
    ap.add_argument("--ts", default=",".join(str(t) for t in TS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arches = [Archetype(x) for x in args.arches.split(",")]
    ts = [int(x) for x in args.ts.split(",")]
    rng = np.random.default_rng(args.seed)

    hdr = (
        f"{'archetype':10s} {'T':>5s} {'nvalid':>7s} | "
        f"{'mean':>5s} {'med':>4s} {'max':>5s} {'dead':>4s} | "
        f"{'bm16 use%':>9s} {'sk8%':>5s} | {'bm32 use%':>9s} {'sk8%':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name in arches:
        for T in ts:
            cnt = counts(synth_topk(name, T, rng=rng))
            u = cnt[cnt > 0]
            s16, s32 = pad_stats(cnt, 16), pad_stats(cnt, 32)
            print(
                f"{name.value:10s} {T:5d} {T * TOPK:7d} | {u.mean():5.1f} "
                f"{int(np.median(u)):4d} {int(u.max()):5d} "
                f"{int((cnt == 0).sum()):4d} | "
                f"{s16['useful']:8.1f} {s16['sk8']:5.0f} | "
                f"{s32['useful']:8.1f} {s32['sk8']:5.0f}"
            )
        print()


if __name__ == "__main__":
    main()
