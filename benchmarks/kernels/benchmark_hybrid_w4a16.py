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
import os
import sys
from pathlib import Path

import torch

# Make the in-tree tests helper importable from a stock checkout, mirroring
# bench_hybrid_w4a16_moe.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.kernels.quantization.test_hybrid_w4a16_perf import (  # noqa: E402
    PROVIDERS,
    _compute_packed_scale_zp,
    _provider_dtype,
    _provider_use_zp,
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


def _apply(a, weights, group_size, provider):
    """Run the production apply path once (respects VLLM_W4A16_HANDASM)."""
    from vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16 import (
        _hybrid_w4a16_apply_impl,
    )
    from vllm.utils.platform_utils import num_compute_units

    use_zp = _provider_use_zp(provider)
    packed_scale_zp = _compute_packed_scale_zp(
        weights["w_s_skinny"], weights["w_zp"] if use_zp else None, a.dtype
    )
    return _hybrid_w4a16_apply_impl(
        a,
        weights["w_q_skinny"],
        weights["w_s_skinny"],
        weights["w_q_skinny_i32"],
        weights["w_zp"] if use_zp else None,
        None,  # bias
        num_compute_units(),
        group_size,
        packed_scale_zp,
    )


def _check_one(
    m: int, k: int, n: int, group_size: int, provider: str
) -> tuple[bool, float]:
    """Compare the active (hand-asm) kernel against the Triton reference on the
    SAME inputs. Returns (handasm_actually_used, cosine_similarity).

    Uses Triton as the trusted reference: run once with VLLM_W4A16_HANDASM=0
    (force Triton) and once with =1 (hand-asm if its config matches this shape),
    then compare. When the hand-asm config does not match, both runs are Triton
    and the check is trivially exact (reported as handasm_used=False).
    """
    from vllm.model_executor.kernels.linear.mixed_precision import asm_w4a16
    from vllm.platforms.rocm import on_gfx1151

    dtype = _provider_dtype(provider)
    weights = prepare_hybrid_weights(k, n, group_size, dtype=dtype)
    a = torch.randn((m, k), device="cuda", dtype=dtype)

    # Did the dispatcher's pinned config match this launch?
    block = _select_skinny_config(m, n, k, group_size, dtype)
    handasm_used = (
        on_gfx1151()
        and block is not None
        and asm_w4a16.config_matches(
            dtype, _provider_use_zp(provider), group_size, *block
        )
    )

    os.environ["VLLM_W4A16_HANDASM"] = "0"
    c_ref = _apply(a, weights, group_size, provider).float()
    os.environ["VLLM_W4A16_HANDASM"] = "1"
    c = _apply(a, weights, group_size, provider).float()

    cos = torch.nn.functional.cosine_similarity(
        c.flatten(), c_ref.flatten(), dim=0
    ).item()
    return handasm_used, cos


def _select_skinny_config(m, n, k, group_size, dtype):
    """The tile (block_m, block_n, block_k, num_warps) the dispatcher would pick,
    or None off gfx11."""
    from vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16 import (
        _select_skinny_gfx11_config,
    )
    from vllm.platforms.rocm import on_gfx1x

    if not on_gfx1x():
        return None
    return _select_skinny_gfx11_config(m, n, k, group_size, dtype)


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
    p.add_argument(
        "--check",
        action="store_true",
        help="Verify the active kernel matches the Triton reference (cos > "
        "0.999) on identical inputs before/instead of timing. Exits non-zero "
        "on mismatch. Use this after editing the hand-asm .amdgcn.",
    )
    p.add_argument(
        "--check-tol",
        type=float,
        default=0.999,
        help="Minimum cosine similarity for --check (default 0.999)",
    )
    args = p.parse_args()

    if args.model:
        shapes = MODEL_SHAPES[args.model]
    else:
        shapes = [
            {"k": args.k, "n": args.n, "group_size": args.group_size, "comment": ""}
        ]

    if args.check:
        failures = 0
        for s in shapes:
            k, n, gs = s["k"], s["n"], s["group_size"]
            for m in args.batches:
                used, cos = _check_one(m, k, n, gs, args.provider)
                ok = cos >= args.check_tol
                failures += not ok
                where = "hand-asm" if used else "triton-only (no hand-asm)"
                print(
                    f"  CHECK {m}x{n}x{k} g{gs} {args.provider}: "
                    f"cos={cos:.6f} [{where}] -> {'PASS' if ok else 'FAIL'}"
                )
        torch.accelerator.synchronize()
        if failures:
            raise SystemExit(f"{failures} correctness check(s) FAILED")
        return

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
