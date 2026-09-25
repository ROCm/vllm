#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every shipped shape against Triton across the context range.

roofline.py answers "where should I look next" in minutes by measuring one
context. This answers "what do we ship" and takes much longer, so it writes
markdown to stdout rather than pretending to be a gate.

    cd <worktree> && VLLM_KV_CACHE_LAYOUT=HND PYTHONPATH=$PWD \\
        amd-gpu-lock <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/matrix.py > table.md
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


def main() -> None:
    p = argparse.ArgumentParser()
    shapeset.add_arguments(p)
    p.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1, 4],
        help="query tokens per sequence; 1 is plain decode, 4 is speculative",
    )
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="whole re-setups per cell; do_bench medians many iterations "
        "inside one already",
    )
    p.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 4096, 8192, 16384, 32768],
    )
    p.add_argument(
        "--nseg",
        type=int,
        default=None,
        help="force NSEG on every configuration, overriding the heuristics",
    )
    p.add_argument(
        "--rg",
        type=int,
        default=None,
        help="force RG on every configuration, overriding the heuristics",
    )
    p.add_argument(
        "--minb",
        type=int,
        default=None,
        help="force MINB on every configuration, overriding the heuristics",
    )
    p.add_argument(
        "--nw",
        type=int,
        default=None,
        help="force NW on every configuration, overriding the heuristics",
    )
    p.add_argument(
        "--dspl",
        type=int,
        default=None,
        help="force DSPL on every configuration, overriding the heuristics",
    )
    p.add_argument(
        "--rspl",
        type=int,
        default=None,
        help="force RSPL on every configuration, overriding the heuristics",
    )
    p.add_argument(
        "--no-triton",
        dest="triton",
        action="store_false",
        help="skip the Triton column; %%roof does not need it and it is half the run",
    )
    shapeset.add_dtype_argument(p)
    args = p.parse_args()
    dtype = shapeset.torch_dtype(args.dtype)

    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    import vllm.v1.attention.backends.rdna35_hip_attn as backend_mod
    from vllm.v1.attention.backends.rdna35_hip_attn import _knobs_for
    from vllm.v1.attention.ops.rdna35_hip_decode import (
        KernelVariant,
        precompile,
        seal,
    )

    # Patch the module's own binding too: _prepare resolves _knobs_for through
    # the backend namespace, so rebinding only the local name here would
    # precompile one set of variants and measure another.
    forced = {
        k: v
        for k, v in (
            ("nseg", args.nseg),
            ("rg", args.rg),
            ("minb", args.minb),
            ("nw", args.nw),
            ("dspl", args.dspl),
            ("rspl", args.rspl),
        )
        if v is not None
    }
    if forced:
        _heuristic = _knobs_for

        def _knobs_for(hq, hkv, d, m):  # noqa: F811
            return {**_heuristic(hq, hkv, d, m), **forced}

        backend_mod._knobs_for = _knobs_for

    # Grouped by configuration, not listed by model.  The table names 50
    # models but holds only 27 distinct (Hq, Hkv, D) tuples, and the kernel
    # cannot tell two models with the same tuple apart -- measuring each is
    # measuring the same thing twice.  Ungrouped, 46% of the run was redundant.
    #
    # That redundancy did buy one thing before it was removed: 161 repeated
    # cells put the harness noise floor at 0.2% median, 1.4% p90, 3.3% worst.
    # The window is part of the key, not just the shape: gemma interleaves
    # windowed and full-context layers, so one model contributes two rows with
    # identical Hq/Hkv/D.  Keyed on shape alone they would collapse into one
    # group and the table would claim a windowed measurement it never took.
    shapes, windowed = shapeset.load(args)
    if not shapes:
        print(f"<!-- no shapes match {shapeset.describe(args) or 'the filters'} -->")
        return
    groups: dict[tuple[int, int, int, int], list[str]] = {}
    for sh in shapes:
        groups.setdefault((sh.hq, sh.hkv, sh.d, sh.window), []).append(sh.model)
    seen = set(groups)
    rows = [(models, *key) for key, models in groups.items()]

    scratch: dict[Any, Any] = {}
    orig_make = backend_mod.make_scratch

    def cached(variant, device):
        if variant not in scratch:
            scratch[variant] = orig_make(variant, device)
        return scratch[variant]

    backend_mod.make_scratch = cached

    wanted = []
    for hq, hkv, d, _window in seen:
        for m in args.m:
            for layout in (0, 1):
                with contextlib.suppress(Exception):
                    wanted.append(
                        KernelVariant(
                            d,
                            hq,
                            hkv,
                            m,
                            args.block_size,
                            layout,
                            **_knobs_for(hq, hkv, d, m),
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
    time.sleep(5)  # clocks settle after a parallel build; see 00-protocol.md

    def timeit(backend, hq, hkv, d, s, m):
        cfg = BenchmarkConfig(
            backend=backend,
            batch_spec=f"q{m}s{s}",
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

    active = shapeset.describe(args)
    print(
        f"<!-- {len(rows)} configuraciones cubren "
        f"{sum(len(m) for m, *_ in rows)} modelos; "
        f"{windowed} filas con ventana deslizante omitidas; {args.dtype}"
        f"{'; filtros: ' + active if active else ''} -->"
    )
    print(
        "| modelos | S | M | Hq | Hkv | D | roofline | Triton | nuestro | vs "
        "| %roof | spread |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    # Ordered by (S, M) first and configuration last, so every configuration
    # sits side by side at the same context and token count -- the comparison
    # the table exists to support.  It also walks the contexts upward across
    # the whole run, which is what warms the allocator: a pass that starts at a
    # large context reads its first cell per configuration up to 10x slow.
    for s in args.contexts:
        for m in args.m:
            for models, hq, hkv, d, _window in rows:
                roof = shapeset.roofline_us(
                    hq, hkv, d, m, s, block_size=args.block_size
                )
                impls: list[Any] = []
                orig_init = backend_mod.Rdna35HipAttentionImpl.__init__

                def spy(self, *a, _init=orig_init, _seen=impls, **kw):
                    _init(self, *a, **kw)
                    _seen.append(self)

                backend_mod.Rdna35HipAttentionImpl.__init__ = spy
                try:
                    ours, spread = timeit("RDNA35_HIP_ATTN", hq, hkv, d, s, m)
                finally:
                    backend_mod.Rdna35HipAttentionImpl.__init__ = orig_init
                ran = any(i.kernel_calls for i in impls) and not any(
                    i.fallback_calls for i in impls
                )
                tri = None
                if args.triton:
                    tri, _ = timeit("TRITON_ATTN", hq, hkv, d, s, m)
                mark = "" if ran else " (fallback)"
                label = (
                    models[0] if len(models) == 1 else f"{models[0]} +{len(models) - 1}"
                )
                print(
                    f"| {label}{mark} | {s} | {m} | {hq} | {hkv} | {d} | "
                    f"{roof:.2f} | {'-' if tri is None else f'{tri:.2f}'} | "
                    f"{ours:.2f} | {'-' if tri is None else f'{tri / ours:.2f}x'} | "
                    f"{roof / ours * 100:.1f} % | {spread:.1f} % |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
