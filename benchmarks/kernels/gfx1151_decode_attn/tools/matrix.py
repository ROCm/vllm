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
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
sys.path.insert(0, str(_ROOT / "benchmarks" / "attention_benchmarks"))

PEAK_GIBS = 230.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default=str(_HERE.parent / "shapes.csv"))
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 4096, 8192, 16384, 32768],
    )
    args = p.parse_args()

    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    import vllm.v1.attention.backends.rdna35_hip_attn as backend_mod
    from vllm.v1.attention.backends.rdna35_hip_attn import _msplit_for, _segments_for
    from vllm.v1.attention.ops.rdna35_hip_decode import KernelVariant, precompile

    # Grouped by configuration, not listed by model.  The table names 50
    # models but holds only 27 distinct (Hq, Hkv, D) tuples, and the kernel
    # cannot tell two models with the same tuple apart -- measuring each is
    # measuring the same thing twice.  Ungrouped, 46% of the run was redundant.
    #
    # That redundancy did buy one thing before it was removed: 161 repeated
    # cells put the harness noise floor at 0.2% median, 1.4% p90, 3.3% worst.
    groups: dict[tuple[int, int, int], list[str]] = {}
    with open(args.shapes) as fh:
        for r in csv.DictReader(fh):
            key = (int(r["Hq"]), int(r["Hkv"]), int(r["D"]))
            groups.setdefault(key, []).append(r["model"])
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
    for hq, hkv, d in seen:
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
                        nseg=_segments_for(hq),
                        msplit=_msplit_for(hkv, args.m),
                    )
                )
    precompile(wanted)
    time.sleep(5)  # clocks settle after a parallel build; see 00-protocol.md

    def timeit(backend, hq, hkv, d, s):
        cfg = BenchmarkConfig(
            backend=backend,
            batch_spec=f"q{args.m}s{s}",
            num_layers=10,
            min_working_set_mb=96,
            head_dim=d,
            num_q_heads=hq,
            num_kv_heads=hkv,
            block_size=args.block_size,
            device="cuda:0",
        )
        run_attention_benchmark(cfg)
        return statistics.median(
            run_attention_benchmark(cfg).median_time * 1e6 for _ in range(args.reps)
        )

    print(
        f"<!-- {len(rows)} configuraciones cubren "
        f"{sum(len(m) for m, *_ in rows)} modelos -->"
    )
    print("| modelos | Hq | Hkv | D | S | roofline | Triton | nuestro | vs | %roof |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for models, hq, hkv, d in rows:
        for s in args.contexts:
            roof = s * 2 * hkv * d * 2 / (PEAK_GIBS * 1024**3) * 1e6
            impls: list[Any] = []
            orig_init = backend_mod.Rdna35HipAttentionImpl.__init__

            def spy(self, *a, _init=orig_init, _seen=impls, **kw):
                _init(self, *a, **kw)
                _seen.append(self)

            backend_mod.Rdna35HipAttentionImpl.__init__ = spy
            try:
                ours = timeit("RDNA35_HIP_ATTN", hq, hkv, d, s)
            finally:
                backend_mod.Rdna35HipAttentionImpl.__init__ = orig_init
            ran = any(i.kernel_calls for i in impls) and not any(
                i.fallback_calls for i in impls
            )
            tri = timeit("TRITON_ATTN", hq, hkv, d, s)
            mark = "" if ran else " (fallback)"
            label = models[0] if len(models) == 1 else f"{models[0]} +{len(models) - 1}"
            print(
                f"| {label}{mark} | {hq} | {hkv} | {d} | {s} | {roof:.2f} | "
                f"{tri:.2f} | {ours:.2f} | {tri / ours:.2f}x | "
                f"{roof / ours * 100:.1f} % |",
                flush=True,
            )


if __name__ == "__main__":
    main()
