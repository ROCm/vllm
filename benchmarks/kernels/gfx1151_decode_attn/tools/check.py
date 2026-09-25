#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate a kernel variant against a float reference.

Run this before trusting any timing: several bugs this session were silent,
producing finite but wrong numbers (an empty workgroup whose gmax was -inf, a
wave reaching the LDS stores with exec = 0, bf16 read as fp16).

    cd <worktree> && PYTHONPATH=$PWD amd-gpu-lock <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/check.py --hq 8 --hkv 4

S=48 matters: a causal off-by-one moves the softmax by ~1/S while the tolerance
is fixed, so long contexts hide exactly the bug most likely to be present.
"""

import argparse
import itertools
import sys

import torch

RTOL = 1e-3


def reference(q, kv, s, hq, hkv, head_dim, m):
    flat = kv.transpose(1, 2).flatten(0, 1)[:s]
    k, v = flat[..., :head_dim], flat[..., head_dim:]
    gqa = hq // hkv
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    vf = v.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * (head_dim**-0.5)
    pos = torch.arange(s, device=q.device).view(1, s)
    lim = (s - m + torch.arange(m, device=q.device)).view(m, 1)
    scores = scores.masked_fill((pos > lim).view(1, m, s), float("-inf"))
    return torch.bmm(torch.softmax(scores, -1), vf).permute(1, 0, 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hq", type=int, default=32)
    p.add_argument("--hkv", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--contexts", type=int, nargs="+", default=[48, 1024])
    p.add_argument("--layouts", type=int, nargs="+", default=[0, 1])
    p.add_argument("--nseg", type=int, nargs="+", default=[None])
    p.add_argument("--rg", type=int, nargs="+", default=[None])
    p.add_argument("--minb", type=int, nargs="+", default=[None])
    p.add_argument("--nw", type=int, default=None)
    p.add_argument("--dspl", type=int, default=None)
    p.add_argument("--ablate", type=int, default=None)
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="invoke N times and check every result; catches state a kernel "
        "leaves behind between launches, such as an arrival counter that is "
        "not reset",
    )
    p.add_argument(
        "--mutate",
        type=int,
        default=0,
        help="1 admits one key too many; the check MUST then fail",
    )
    args = p.parse_args()

    from vllm.v1.attention.ops.rdna35_hip_decode import (
        KernelVariant,
        load,
        make_scratch,
        precompile,
    )

    dev = torch.device("cuda")
    failures = 0

    # Build every variant the matrix needs before checking any of them.  The
    # variant does not depend on S, so the product below is usually a handful
    # of builds at ~19.6 s each -- serially that is minutes of nothing before
    # the first result.  Failures are left for the loop, which reports them
    # against the shape that asked for them.
    def variant_for(layout, nseg, rg, minb):
        from vllm.v1.attention.backends.rdna35_hip_attn import _knobs_for

        kw = dict(_knobs_for(args.hq, args.hkv, args.head_dim, args.m))
        kw["mutate"] = args.mutate
        for name, val in (
            ("nseg", nseg),
            ("rg", rg),
            ("minb", minb),
            ("nw", args.nw),
            ("dspl", args.dspl),
            ("ablate", args.ablate),
        ):
            if val is not None:
                kw[name] = val
        return KernelVariant(
            args.head_dim, args.hq, args.hkv, args.m, args.block_size, layout, **kw
        )

    precompile(
        [
            variant_for(*combo)
            for combo in itertools.product(args.layouts, args.nseg, args.rg, args.minb)
        ]
    )
    for s, layout, nseg, rg, minb in itertools.product(
        args.contexts, args.layouts, args.nseg, args.rg, args.minb
    ):
        torch.manual_seed(0)
        # Round up, so S need not be a multiple of the page: a tile that runs
        # off the end of the sequence is exactly the case where a kernel may
        # read uninitialised slots, and an exact-division harness never builds
        # one.  The trailing slots are filled with NaN below for that reason.
        blocks = -(-s // args.block_size)
        shape = (
            (blocks, args.block_size, args.hkv, 2 * args.head_dim)
            if layout == 0
            else (blocks, args.hkv, args.block_size, 2 * args.head_dim)
        )
        kv = torch.randn(shape, device=dev, dtype=torch.float16) * 0.5
        if layout == 0:
            kv = kv.transpose(1, 2)
        # Slots past the sequence carry NaN, not plausible data.  A kernel that
        # addresses them is fine; one that lets them reach the accumulator is
        # not, and 0 * NaN = NaN makes that failure total rather than subtle.
        tail = blocks * args.block_size - s
        if tail:
            kv[-1, :, args.block_size - tail :, :] = float("nan")
        q = (
            torch.randn(args.m, args.hq, args.head_dim, device=dev, dtype=torch.float16)
            * 0.5
        )
        bt = torch.arange(blocks, device=dev, dtype=torch.int32)

        variant = variant_for(layout, nseg, rg, minb)
        module = load(variant)
        acc, smax, ssum, arrivals = make_scratch(variant, dev)
        out = torch.empty_like(q)
        ref = reference(q, kv, s, args.hq, args.hkv, args.head_dim, args.m)
        max_rel = 0.0
        ok = True
        for _ in range(args.repeat):
            out.zero_()
            module.decode_attn(
                q, kv, bt, out, acc, smax, ssum, arrivals, s, args.head_dim**-0.5
            )
            torch.accelerator.synchronize()
            got = out.float()
            rel = ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
            max_rel = max(max_rel, rel)
            ok = ok and rel <= RTOL and torch.isfinite(got).all()
        if args.mutate:
            ok = not ok  # the negative control must be detected
        failures += not ok
        label = f"S={s} l={layout} nseg={nseg} rg={rg} minb={minb}"
        print(f"{label:<44} max_rel={max_rel:.3e}  {'PASS' if ok else 'FAIL'}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
