#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find the one set of launch knobs per configuration that serves every context.

The constraint that shapes this whole search: a configuration may not depend
on S. The grid is fixed when the CUDA graph is captured and S is a runtime
argument, so one choice has to serve 128 and 32768 alike. Tuning per context
would be easy and unshippable.

So the objective is the whole range: the geomean of the fraction of roofline
over the seven contexts matrix.py reports, with the worst context alongside.
A candidate that wins the long contexts by starving the short ones scores
badly on both.

The search is coordinate descent -- one knob at a time over its values, the
rest held, keeping the best -- starting from the backend's own choice.  The
knobs interact, which is why --rounds exists; an exhaustive grid is ~100
builds per configuration.

    cd <worktree> && VLLM_KV_CACHE_LAYOUT=HND PYTHONPATH=$PWD \\
        amd-gpu-lock <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/tune.py --hq 32 --hkv 8 \\
        --head-dim 128 --m 1

Prints one `_TUNED` row per configuration, ready to paste.
"""

import argparse
import contextlib
import math
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
sys.path.insert(0, str(_ROOT / "benchmarks" / "attention_benchmarks"))
sys.path.insert(0, str(_HERE.parent))

import shapeset  # noqa: E402

CONTEXTS = [128, 512, 1024, 4096, 8192, 16384, 32768]
# Long-context workgroup counts worth trying; see _TARGET_WORKGROUPS in the
# backend for why the answer is small.
TARGETS = (8, 16, 32)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def space(hq, hkv, d, start):
    """The values each knob is searched over, for one configuration."""
    gqa = hq // hkv
    rg0 = start["rg"]
    default_dspl = d // 128 if d >= 256 else 1
    rgs = {1, rg0, 2 * rg0, 4 * rg0, max(1, rg0 // 2)}
    return {
        "nw": [2, 4, 8],
        "dspl": [0, 2 * default_dspl] if d >= 128 else [0],
        "rg": sorted(r for r in rgs if gqa % r == 0),
        "target": list(TARGETS),
        "minb": [1, 2, 4],
    }


def knobs_of(hkv, cand):
    """KernelVariant keyword arguments for a search point."""
    k = {
        "nw": cand["nw"],
        "rg": cand["rg"],
        "minb": cand["minb"],
        "nseg": max(1, cand["target"] // (hkv * cand["rg"])),
    }
    if cand["dspl"]:
        k["dspl"] = cand["dspl"]
    return k


def main() -> None:
    p = argparse.ArgumentParser()
    shapeset.add_arguments(p)
    p.add_argument("--m", type=int, nargs="+", default=[1, 4])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    shapeset.add_dtype_argument(p)
    args = p.parse_args()
    dtype = shapeset.torch_dtype(args.dtype)

    from common import BenchmarkConfig
    from runner import run_attention_benchmark

    import vllm.v1.attention.backends.rdna35_hip_attn as backend_mod
    from vllm.v1.attention.backends.rdna35_hip_attn import _knobs_for
    from vllm.v1.attention.ops.rdna35_hip_decode import (
        KernelVariant,
        load,
        precompile,
        seal,
    )

    shapes, _windowed = shapeset.load(args)
    configs = sorted({(s.hq, s.hkv, s.d) for s in shapes if s.d != 96})
    if not configs:
        print(f"no shapes match {shapeset.describe(args) or 'the given filters'}")
        return

    scratch: dict[Any, Any] = {}
    orig_make = backend_mod.make_scratch

    def cached(variant, device):
        if variant not in scratch:
            scratch[variant] = orig_make(variant, device)
        return scratch[variant]

    backend_mod.make_scratch = cached

    original = backend_mod.Rdna35HipAttentionImpl._prepare
    override: dict[str, int] = {}

    # The override replaces the whole knob set rather than patching what
    # _prepare chose, so nothing of the base leaks into a point that did not
    # ask for it -- dspl in particular, where absent means the kernel's rule.
    def patched(self, kv_cache, **kw):
        built = original(self, kv_cache, **kw)
        if built is not None and override:
            v = self._variant
            want = KernelVariant(
                v.head_size,
                v.num_q_heads,
                v.num_kv_heads,
                v.max_m,
                v.block_size,
                v.layout,
                **override,
                dtype=v.dtype,
            )
            if self._variant != want:
                self._variant = want
                self._built = (load(want), cached(want, kw["q"].device))
            built = self._built
        return built

    backend_mod.Rdna35HipAttentionImpl._prepare = patched

    def timeit(hq, hkv, d, m, s):
        cfg = BenchmarkConfig(
            backend="RDNA35_HIP_ATTN",
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
        return run_attention_benchmark(cfg).median_time * 1e6

    def key(c):
        return tuple(sorted(c.items()))

    for hq, hkv, d in configs:
        for m in args.m:
            base = dict(_knobs_for(hq, hkv, d, m))
            start = {
                "nw": base.get("nw", 8),
                "dspl": base.get("dspl", 0),
                "rg": base.get("rg", 1),
                "target": 16,
                "minb": base.get("minb", 1),
            }
            values = space(hq, hkv, d, start)
            roof = {
                s: shapeset.roofline_us(hq, hkv, d, m, s, block_size=args.block_size)
                for s in args.contexts
            }
            scores: dict[tuple, tuple[float, list[float]]] = {}

            def evaluate(
                cands, hq=hq, hkv=hkv, d=d, m=m, base=base, roof=roof, scores=scores
            ):
                todo = [c for c in cands if key(c) not in scores]
                variants = {}
                for c in todo:
                    with contextlib.suppress(Exception):
                        variants[key(c)] = KernelVariant(
                            d,
                            hq,
                            hkv,
                            m,
                            args.block_size,
                            1,
                            **knobs_of(hkv, c),
                            dtype=dtype,
                        )
                # The unpatched _prepare builds the backend's own choice before
                # the override replaces it, so that variant is needed too.
                seal(False)
                precompile(
                    [
                        *variants.values(),
                        KernelVariant(
                            d, hq, hkv, m, args.block_size, 1, **base, dtype=dtype
                        ),
                    ]
                )
                seal()
                time.sleep(5)  # clocks settle after a parallel build
                for c in todo:
                    if key(c) not in variants:
                        scores[key(c)] = (0.0, [])
                        continue
                    override.clear()
                    override.update(knobs_of(hkv, c))
                    try:
                        r = [roof[s] / timeit(hq, hkv, d, m, s) for s in args.contexts]
                    except Exception:  # noqa: BLE001 - a point may not build
                        scores[key(c)] = (0.0, [])
                        continue
                    scores[key(c)] = (geomean(r), r)
                override.clear()

            best = dict(start)
            evaluate([best])
            for _ in range(args.rounds):
                for knob in ("nw", "dspl", "rg", "target", "minb"):
                    cands = [dict(best, **{knob: v}) for v in values[knob]]
                    evaluate(cands)
                    for c in cands:
                        if scores[key(c)][0] > scores[key(best)][0]:
                            best = c
            g, cells = scores[key(best)]
            row = ", ".join(f'"{k}": {v}' for k, v in knobs_of(hkv, best).items())
            worst = min(cells) * 100 if cells else 0.0
            print(
                f"    ({hq}, {hkv}, {d}, {m}): {{{row}}},  "
                f"# {g * 100:.1f} % geomean, worst {worst:.1f} %",
                flush=True,
            )


if __name__ == "__main__":
    main()
