# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sweep wvSplitK_int4_hf_sml_ tuples at batch N=1..5 for the production shapes.

Every tuple is checked against the production op's output and timed with the
perf-harness methodology (cold rotated weights, cudagraph replay); the
production op is timed the same way in the same pass as the baseline.
Needs a build with VLLM_SKINNY_GEMM_SWEEP=1 (op wvSplitK_int4g_sweep_skinny).

    python benchmarks/kernels/sweep_int4g_skinny.py [--only 8960x1536 ...]
        [--batch 1 2 ...] [--peak-gibs 230] [--out CSV]
"""

import argparse
import csv
import itertools
import pathlib
import sys
import time

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tests.kernels.quantization.test_hybrid_w4a16_perf as perf  # noqa: E402
import vllm._custom_ops as ops  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402
from vllm.utils.platform_utils import num_compute_units  # noqa: E402

# M(out) x K(in), dtype, group_size, modes used in production
SHAPES = """
64x2048 bf16 128 sym
128x2048 f16 128 sym
2560x256 bf16 32 sym
896x896 f16 128 asym
256x2560 bf16 32 sym
1152x896 f16 128 asym
2048x1536 f16 32 asym
2560x1536 f16 32 asym
4096x1536 f16 32 asym
1536x2048 f16 32 asym
9728x896 f16 128 asym
2560x2048 f16 128 asym
2048x2048 f16 128 asym
5120x1536 f16 32 asym
2048x2560 f16 32 asym
8960x1536 f16 32 asym
896x4864 f16 128 asym
2048x2560 bf16 32 sym
12288x1536 f16 32 asym
2560x2048 bf16 32 sym
2560x2048 f16 32 asym
2048x4096 bf16 128 sym
1536x4096 f16 32 asym
2048x4096 f16 128 sym
24576x1536 f16 32 asym
3072x2560 bf16 32 sym
3072x2560 f16 32 asym
4096x2048 f16 128 asym
4096x2560 f16 32 asym
4608x3584 f16 128 sym,asym
4096x2560 bf16 32 sym
4608x3584 bf16 128 sym
2048x16384 f16 128 asym
2560x4096 f16 32 sym,asym
1536x6144 f16 32 asym
2048x6144 f16 128 asym
5120x2048 f16 128 sym
2560x4096 bf16 32 sym
"""

YTILES = [1, 2, 4]
WVPRGRPS = [8, 12, 16, 32]
# Batch 1 also sweeps UNRL=8 and A_CHUNK=8; more activation rows multiply the
# register footprint, so N>=2 sweeps the smaller grid the op instantiates.
UNRLS = {1: [1, 2, 4, 8]}
ACHUNKS = {1: [8, 16, 32]}
UNRLS_N, ACHUNKS_N = [1, 2, 4], [16, 32]
LDS_ELEMS = 64 * 1024 // 2
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16}


def bench(fn, bufs):
    n = len(bufs)
    idx = [0]

    def run():
        b = bufs[idx[0] % n]
        idx[0] += 1
        return fn(b)

    return triton.testing.do_bench_cudagraph(run, quantiles=[0.5]) * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="MxK filters")
    p.add_argument("--batch", nargs="*", type=int, default=[1], help="N values")
    p.add_argument("--dtypes", nargs="*", help="dtype filter (f16, bf16)")
    p.add_argument("--modes", nargs="*", help="override the listed modes (sym, asym)")
    p.add_argument("--as-dtype", choices=DTYPES, help="override the listed dtypes")
    p.add_argument("--peak-gibs", type=float, default=230.0)
    p.add_argument("--out", default="sweep_int4g_skinny.csv")
    args = p.parse_args()
    peak = args.peak_gibs * 2**30

    cu = num_compute_units()
    out = open(args.out, "w", newline="")  # noqa: SIM115
    writer = csv.writer(out)
    writer.writerow(
        [
            "M",
            "K",
            "dtype",
            "g",
            "mode",
            "batch",
            "ytile",
            "unrl",
            "achunk",
            "wvprgrp",
            "time_us",
            "pct_peak",
            "max_rel_err",
        ]
    )
    seen = set()
    for line in SHAPES.strip().splitlines():
        mk, dt, g, modes = line.split()
        if args.only and mk not in args.only:
            continue
        if args.dtypes and dt not in args.dtypes:
            continue
        if args.as_dtype:
            dt = args.as_dtype
        if args.modes:
            modes = ",".join(args.modes)
        if (mk, dt, g, modes) in seen:
            continue
        seen.add((mk, dt, g, modes))
        M, K = map(int, mk.split("x"))
        g = int(g)
        dtype = DTYPES[dt]
        w = perf.prepare_hybrid_weights(K, M, g, dtype=dtype)
        w_bytes = w["w_q_skinny"].numel()
        n_buf = max(2, min(perf._ROTATE_MAX, -(-perf._ROTATE_TARGET_BYTES // w_bytes)))
        for mode, n in itertools.product(modes.split(","), args.batch):
            if K * n > LDS_ELEMS:
                continue  # the chunked kernel serves it, not swept here
            zp = mode == "asym"
            x = torch.randn(n, K, dtype=dtype, device="cuda")
            nbytes = (
                M * K // 2
                + M * (K // g) * 2
                + (M * (K // g) // 2 if zp else 0)
                + n * K * 2
                + n * M * 2
            )
            bufs = [
                (
                    perf._clone_strided(w["w_q_skinny"]),
                    perf._clone_strided(w["w_s_skinny"]),
                    perf._clone_strided(w["w_zp"]) if zp else None,
                )
                for _ in range(n_buf)
            ]

            def prod(b, x=x, g=g):
                return ops.wvSplitK_int4_g(b[0], x, b[1], cu, g, b[2])

            def row(tup, t, err, key=(M, K, dt, g, mode, n), nbytes=nbytes):
                pct = f"{nbytes / (t * 1e-6) / peak * 100:.1f}" if t else ""
                writer.writerow([*key, *tup, f"{t:.3f}" if t else "", pct, err])

            ref = prod(bufs[0])
            scale = ref.float().abs().max().item() or 1.0
            t_prod = bench(prod, bufs)
            row(("prod", "", "", ""), t_prod, 0)
            best = (t_prod, "prod")
            unrls, achunks = UNRLS.get(n, UNRLS_N), ACHUNKS.get(n, ACHUNKS_N)
            for yt, un, ac, wv in itertools.product(YTILES, unrls, achunks, WVPRGRPS):
                if K % ac or M % yt:
                    continue

                def tuned(b, x=x, g=g, yt=yt, un=un, ac=ac, wv=wv):
                    return ops.wvSplitK_int4g_sweep_skinny(
                        b[0], x, b[1], b[2], cu, g, yt, un, ac, wv
                    )

                try:
                    o = tuned(bufs[0])
                    torch.accelerator.synchronize()
                    err = (o.float() - ref.float()).abs().max().item() / scale
                    if err > 2e-2:
                        row((yt, un, ac, wv), None, f"{err:.2e}")
                        continue
                    t = bench(tuned, bufs)
                except Exception as e:  # noqa: BLE001
                    print(f"  {mk} {mode} N={n} {(yt, un, ac, wv)}: {e}")
                    continue
                row((yt, un, ac, wv), t, f"{err:.2e}")
                if t < best[0]:
                    best = (t, (yt, un, ac, wv))
            out.flush()
            print(
                f"{mk:>11} {dt:>4} g{g:<3} {mode:>4} N={n}  prod {t_prod:7.2f} us"
                f"  best {best[0]:7.2f} us {best[1]}  "
                f"({(t_prod / best[0] - 1) * 100:+.1f}%)",
                flush=True,
            )
            del bufs
        del w
        torch.accelerator.empty_cache()
        time.sleep(1.0)
    out.close()


if __name__ == "__main__":
    main()
