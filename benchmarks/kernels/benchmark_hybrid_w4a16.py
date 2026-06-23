#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI benchmark for the dense hybrid_w4a16 GEMM kernel.

Drives the same measurement rig as the perf regression test
(``tests/kernels/quantization/test_hybrid_w4a16_perf.py``): packed int4
weights via the production ``pack_skinny_int4`` helper, cold-weight MALL
rotation, and ``do_bench_cudagraph`` median timing -- so the reported
time/call matches what a real prefill forward pass sees (and what shows up
in a torch profiler trace).

Defaults reproduce the Qwen3-VL-4B-Instruct-AWQ-4bit fused gate_up_proj
GEMM at the 1322-token prefill shape from the gfx1151 roofline analysis:
M=1322, N=19456, K=2560, group_size=32, symmetric (no zero-point), bf16.

Usage::

    # Default: the up_proj shape at M=1322
    python benchmarks/kernels/benchmark_hybrid_w4a16.py

    # All Qwen3-VL-4B AWQ decoder GEMMs across a batch sweep
    python benchmarks/kernels/benchmark_hybrid_w4a16.py --model qwen3vl-4b-awq

    # A custom shape / quant config
    python benchmarks/kernels/benchmark_hybrid_w4a16.py \\
        --k 2560 --n 19456 --group-size 32 --provider hybrid-w4a16-bf16 \\
        --batches 1024 1322 2048

The provider name follows the test convention
``hybrid-w4a16[-zp][-bf16]``: ``-zp`` selects the asymmetric per-group
zero-point dequant path; ``-bf16`` runs with bfloat16 activations/scales
instead of float16. Symmetric fp16 is the bare ``hybrid-w4a16``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the in-tree tests helper importable from a stock checkout, mirroring
# bench_hybrid_w4a16_moe.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.kernels.quantization.test_hybrid_w4a16_perf import (  # noqa: E402
    PROVIDERS,
    _provider_dtype,
    measure_tflops,
    prepare_hybrid_weights,
)

# Named (K, N) decoder GEMM presets. K=in_features, N=out_features.
# qwen3vl-4b-awq: group_size=32, symmetric, bf16 (see model config).
MODEL_SHAPES: dict[str, list[dict]] = {
    "qwen3vl-4b-awq": [
        {"k": 2560, "n": 19456, "group_size": 32, "comment": "gate_up_proj"},
        {"k": 9728, "n": 2560, "group_size": 32, "comment": "down_proj"},
        {"k": 2560, "n": 6144, "group_size": 32, "comment": "qkv_proj"},
        {"k": 2560, "n": 4096, "group_size": 32, "comment": "o_proj"},
    ],
}


def _bench_one(
    m: int, k: int, n: int, group_size: int, provider: str
) -> tuple[str, float, float]:
    """Return (kernel, tflops, ms) for one shape/provider/batch."""
    weights = prepare_hybrid_weights(k, n, group_size, dtype=_provider_dtype(provider))
    measure_tflops(m, weights, k, n, group_size, provider)  # warmup
    kernel, tflops = measure_tflops(m, weights, k, n, group_size, provider)
    ms = (2 * m * n * k) * 1e-12 / (tflops * 1e-3)
    return kernel, tflops, ms


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=sorted(MODEL_SHAPES),
        help="Benchmark all decoder GEMMs for a named model preset.",
    )
    p.add_argument("--k", type=int, default=2560, help="in_features (default 2560)")
    p.add_argument("--n", type=int, default=19456, help="out_features (default 19456)")
    p.add_argument("--group-size", type=int, default=32, help="quant group size")
    p.add_argument(
        "--provider",
        default="hybrid-w4a16-bf16",
        choices=PROVIDERS,
        help="quant variant (default hybrid-w4a16-bf16: bf16, symmetric)",
    )
    p.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1322],
        help="M (token) batch sizes to sweep (default: 1322)",
    )
    args = p.parse_args()

    if args.model:
        shapes = MODEL_SHAPES[args.model]
    else:
        shapes = [
            {"k": args.k, "n": args.n, "group_size": args.group_size, "comment": ""}
        ]

    hdr = f"{'shape (MxNxK)':>22} {'g':>4} {'kernel':>22} {'TFLOP/s':>9} {'ms/call':>9}"
    print(hdr)
    print("-" * len(hdr))
    for s in shapes:
        k, n, gs = s["k"], s["n"], s["group_size"]
        for m in args.batches:
            kernel, tflops, ms = _bench_one(m, k, n, gs, args.provider)
            tag = f"  # {s['comment']}" if s["comment"] else ""
            shape_str = f"{m}x{n}x{k}"
            print(
                f"{shape_str:>22} {gs:>4} {kernel:>22} {tflops:>9.2f} {ms:>9.3f}{tag}"
            )
    torch.accelerator.synchronize()


if __name__ == "__main__":
    main()
