#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where every shipped shape sits against its KV roofline, worst first.

This is both halves of the optimisation loop in one command: it is the
regression gate, because it checks every shape against a float reference, and
it is the opportunity finder, because it sorts by distance from the roofline.

    cd <worktree> && VLLM_KV_CACHE_LAYOUT=HND PYTHONPATH=$PWD \\
        amd-gpu-lock <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/roofline.py

S=128 only, deliberately. The fixed cost is the same at every context and the
roofline budget is smallest here, so this is where a shape's distance from the
bus is most visible -- and one context keeps a full 50-shape pass to minutes.

`ran` is the column to read first. The backend falls back to Triton silently
for any shape it cannot serve, and a harness that does not check reports
Triton's number as if it were ours; that has happened twice on this project.
"""

import argparse
import contextlib
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


def correctness(hq, hkv, d, m, block_size, dtype, s=48):
    """max_rel against a float reference, or None if the shape does not build.

    S=48 rather than 128: a causal off-by-one moves the softmax by ~1/S while
    the tolerance is fixed, so a short context is the one that catches it.
    """
    from vllm.v1.attention.backends.rdna35_hip_attn import _knobs_for
    from vllm.v1.attention.ops.rdna35_hip_decode import (
        KernelVariant,
        UnexpectedBuildError,
        load,
        make_scratch,
    )

    dev = torch.device("cuda")
    torch.manual_seed(0)
    blocks = -(-s // block_size)
    kv = torch.randn(blocks, hkv, block_size, 2 * d, device=dev, dtype=dtype)
    kv *= 0.5
    q = torch.randn(m, hq, d, device=dev, dtype=dtype) * 0.5
    bt = torch.arange(blocks, device=dev, dtype=torch.int32)
    try:
        variant = KernelVariant(
            d,
            hq,
            hkv,
            m,
            block_size,
            1,
            **_knobs_for(hq, hkv, d, m),
            dtype=dtype,
        )
        module = load(variant)
        acc, smax, ssum, arrivals = make_scratch(variant, dev)
    except UnexpectedBuildError:
        # Never swallowed: the seal exists to be heard.
        raise
    except Exception:
        return None
    out = torch.empty_like(q)
    seq_lens = torch.tensor([s], device=dev, dtype=torch.int32)
    module.decode_attn(q, kv, bt, out, acc, smax, ssum, arrivals, seq_lens, d**-0.5)
    torch.accelerator.synchronize()
    ref = reference(q, kv, s, hq, hkv, d, m)
    got = out.float()
    if not torch.isfinite(got).all():
        return float("inf")
    return ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()


def main() -> None:
    p = argparse.ArgumentParser()
    shapeset.add_arguments(p)
    p.add_argument("--s", type=int, default=128)
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="whole re-setups per cell; do_bench medians many iterations "
        "inside one already",
    )
    p.add_argument("--triton", action="store_true", help="also time TRITON_ATTN")
    p.add_argument("--skip-check", action="store_true", help="timings only")
    shapeset.add_dtype_argument(p)
    args = p.parse_args()
    dtype = shapeset.torch_dtype(args.dtype)

    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    import vllm.v1.attention.backends.rdna35_hip_attn as backend_mod

    rows, windowed = shapeset.load(args)
    if not rows:
        print(f"no shapes match {shapeset.describe(args) or 'the given filters'}")
        return

    # Scratch is memoised for the same reason sweep.py memoises it: a rebuild
    # per forward drags a device memset into the timed region.
    scratch: dict[Any, Any] = {}
    orig_make = backend_mod.make_scratch

    def cached(variant, device):
        if variant not in scratch:
            scratch[variant] = orig_make(variant, device)
        return scratch[variant]

    backend_mod.make_scratch = cached

    # Build every variant the pass needs before timing anything. A cold build
    # is ~19.6 s and they are independent, so serially the 27 distinct variants
    # of this table cost nine minutes before the first measurement; spread over
    # the machine it is under a minute. Nothing here is timed, so a build that
    # lands late cannot contaminate a number.
    from vllm.v1.attention.backends.rdna35_hip_attn import _knobs_for
    from vllm.v1.attention.ops.rdna35_hip_decode import (
        KernelVariant,
        precompile,
        seal,
    )

    wanted = []
    for _, hq, hkv, d, _window in rows:
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
                        dtype=dtype,
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
    # Let the machine settle before timing anything.  Compiling saturates the
    # cores and moves the SoC clock, which is the contamination 00-protocol.md
    # invented quiet-lock to avoid; doing it in-process moments before the
    # first measurement is the same mistake by another route.  A pass taken
    # straight after a 50-way rebuild reported three shapes falling back that
    # were served on every quiet run before and after.
    time.sleep(5)

    active = shapeset.describe(args)
    if active:
        print(f"filtered to {len(rows)} shapes: {active}")
    hdr = f"{'model':<36} {'Hq':>3} {'Hkv':>4} {'D':>4} {'ran':>4} {'max_rel':>9}"
    hdr += f" {'us':>9} {'%roof':>6} {'spread':>7}"
    if args.triton:
        hdr += f" {'triton':>9} {'vs':>6}"
    print(hdr)

    out_rows = []
    for model, hq, hkv, d, _window in rows:
        rel = (
            None
            if args.skip_check
            else correctness(hq, hkv, d, args.m, args.block_size, dtype, s=48)
        )
        roof = shapeset.roofline_us(
            hq, hkv, d, args.m, args.s, block_size=args.block_size
        )

        def timeit(backend, d=d, hq=hq, hkv=hkv):
            cfg = BenchmarkConfig(
                backend=backend,
                batch_spec=f"q{args.m}s{args.s}",
                num_layers=10,
                min_working_set_mb=96,
                head_dim=d,
                num_q_heads=hq,
                num_kv_heads=hkv,
                block_size=args.block_size,
                device="cuda:0",
                dtype=dtype,
            )
            rs = [run_attention_benchmark(cfg) for _ in range(args.reps)]
            spread = max(r.std_time / r.median_time for r in rs) * 100
            return statistics.median(r.median_time for r in rs) * 1e6, spread

        impls: list[Any] = []
        orig_init = backend_mod.Rdna35HipAttentionImpl.__init__

        def spy(self, *a, _init=orig_init, _seen=impls, **kw):
            _init(self, *a, **kw)
            _seen.append(self)

        backend_mod.Rdna35HipAttentionImpl.__init__ = spy
        try:
            us, spread = timeit("RDNA35_HIP_ATTN")
        finally:
            backend_mod.Rdna35HipAttentionImpl.__init__ = orig_init
        ran = any(i.kernel_calls for i in impls) and not any(
            i.fallback_calls for i in impls
        )
        # Why, not just whether.  warning_once suppresses the backend's own
        # message after the first shape hits a given reason, so a pass over
        # many shapes hides every fallback but the first.
        why = next((i._rejected for i in impls if i._rejected), None)

        line = f"{model:<36} {hq:>3} {hkv:>4} {d:>4} {'yes' if ran else 'NO':>4}"
        line += f" {'-' if rel is None else f'{rel:.2e}':>9}"
        line += f" {us:>9.2f} {roof / us * 100:>5.1f}% {spread:>6.1f}%"
        if args.triton:
            t, _ = timeit("TRITON_ATTN")
            line += f" {t:>9.2f} {t / us:>5.2f}x"
        if not ran and why:
            line += f"  <- {why}"
        print(line, flush=True)
        out_rows.append((roof / us * 100, ran, model, hq, hkv, d, us))

    print("\nworst first (only shapes the kernel actually serves):")
    served = [r for r in out_rows if r[1]]
    for pct, _, model, hq, hkv, d, us in sorted(served):
        print(f"  {pct:>5.1f}%  {model:<36} Hq={hq} Hkv={hkv} D={d}  {us:.2f} us")
    missing = [r for r in out_rows if not r[1]]
    if missing:
        by_d: dict[int, int] = {}
        for _, _, _, _, _, d, _ in missing:
            by_d[d] = by_d.get(d, 0) + 1
        spread = ", ".join(f"D={k}: {v}" for k, v in sorted(by_d.items()))
        print(f"\nnot served, {len(missing)} of {len(out_rows)} shapes -> {spread}")
    if windowed:
        print(f"skipped {windowed} sliding-window shapes: the kernel is full-context")


if __name__ == "__main__":
    main()
