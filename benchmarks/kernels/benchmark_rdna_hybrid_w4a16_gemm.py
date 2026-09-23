# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the RDNAHybridW4A16LinearKernel across decode and prefill shapes.

Usage:
    python benchmark_int4_gemm.py
    python benchmark_int4_gemm.py --models Qwen/Qwen3-4B
    python benchmark_int4_gemm.py --group-size 128
"""

import argparse
import copy
import itertools
import os

import torch

from vllm.triton_utils import triton

# ---------------------------------------------------------------------------
# Weight shapes: [K, N], TP_SPLIT_DIM
# ---------------------------------------------------------------------------
WEIGHT_SHAPES = {
    "Qwen/Qwen3-4B": [
        ([2560, 3840], 1),  # qkv_proj
        ([2560, 2560], 0),  # o_proj
        ([2560, 19456], 1),  # gate_up_proj
        ([9728, 2560], 0),  # down_proj
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        ([3584, 4608], 1),
        ([3584, 3584], 0),
        ([3584, 37888], 1),
        ([18944, 3584], 0),
    ],
    "trymirai/SmolLM2-1.7B-Instruct-AWQ": [
        ([2048, 6144], 1),  # qkv_proj
        ([2048, 2048], 0),  # o_proj
        ([2048, 16384], 1),  # gate_up_proj
        ([8192, 2048], 0),  # down_proj
    ],
    "RedHatAI/Qwen3-8B-quantized.w4a16": [
        ([4096, 6144], 1),  # qkv_proj
        ([4096, 4096], 0),  # o_proj
        ([4096, 24576], 1),  # gate_up_proj
        ([12288, 4096], 0),  # down_proj
    ],
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": [
        ([4096, 6144], 1),  # qkv_proj
        ([4096, 4096], 0),  # o_proj
        ([4096, 28672], 1),  # gate_up_proj
        ([14336, 4096], 0),  # down_proj
    ],
}


# ---------------------------------------------------------------------------
# Weight packing
# ---------------------------------------------------------------------------
def prepare_hybrid_weights(K, N, group_size, device="cuda"):
    """Create random weights for benchmarking, in the layer's own layout.

    Packing goes through ``pack_skinny_int4`` rather than a plain contiguous
    buffer so the row stride (and the gfx1151 cliff pad) matches production --
    throughput is a period-512-byte function of that stride, so a benchmark
    that packs its own weights measures a different kernel.
    """
    from vllm.model_executor.kernels.linear.mixed_precision import (
        rdna_hybrid_w4a16 as _k,
    )

    num_groups = K // group_size

    # Actual weight values don't matter for throughput, but the zero-points
    # feed the fp16 carrier's arithmetic, so keep them in their real 0..15 range.
    unpacked = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)
    w_q_skinny, w_q_skinny_i32 = _k.pack_skinny_int4(unpacked)
    w_s_skinny = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.01
    zp_unpacked = torch.randint(
        0, 16, (N, num_groups), dtype=torch.int32, device=device
    )

    return {
        "w_q_skinny": w_q_skinny,
        "w_q_skinny_i32": w_q_skinny_i32,
        "w_s_skinny": w_s_skinny,
        # Zero-points packed 8 rows per int32 word, as the kernels read them.
        "w_zp": _pack_zp_along_n(zp_unpacked),
        "packed_scale_zp": _k.pack_scale_zp_carrier(
            w_s_skinny, zp_unpacked, torch.float16
        ),
        # FP16 baseline for F.linear
        "w_fp16": torch.randn(N, K, dtype=torch.float16, device=device) * 0.01,
    }


def _pack_zp_along_n(zp_nkg):
    """[N, K//G] raw nibbles -> [N//8, K//G] int32, row n at bits 4*(n%8)."""
    N, G = zp_nkg.shape
    shifts = (torch.arange(8, device=zp_nkg.device, dtype=torch.int32) * 4)[:, None]
    return torch.sum(
        (zp_nkg.view(N // 8, 8, G) & 0xF) << shifts, dim=1, dtype=torch.int32
    ).contiguous()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
PROVIDERS = ["torch-fp16", "hybrid-w4a16", "hybrid-w4a16-zp"]
# Force one path regardless of the dispatch heuristic, so all of them can be
# timed on the same shape (used to check where the HIP skinny kernel stops
# winning, and to compare hipBLASLt against both of the in-tree kernels).
FORCED_PROVIDERS = [
    "hip-w4a16",
    "triton-w4a16",
    "hipblaslt-w4a16",
    "hip-w4a16-zp",
    "triton-w4a16-zp",
    "hipblaslt-w4a16-zp",
]

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def _make_runner(provider, a, weights, group_size):
    import vllm._custom_ops as ops
    from vllm.model_executor.kernels.linear.mixed_precision import (
        rdna_hybrid_w4a16 as _k,
    )
    from vllm.utils.platform_utils import num_compute_units

    w = weights
    cu_count = num_compute_units()
    asym = provider.endswith("-zp")
    zp = w["w_zp"] if asym else None
    carrier = w["packed_scale_zp"] if asym else None

    if provider == "torch-fp16":
        w_fp16 = w["w_fp16"]
        return lambda: torch.nn.functional.linear(a, w_fp16)

    if provider.startswith("hybrid-w4a16"):
        return lambda: _k._rdna_hybrid_w4a16_apply_impl(
            a,
            w["w_q_skinny"],
            w["w_s_skinny"],
            w["w_q_skinny_i32"],
            zp,
            None,  # bias
            cu_count,
            group_size,
            carrier,
        )

    if provider.startswith("hip-w4a16"):
        return lambda: ops.wvSplitK_int4_g(
            w["w_q_skinny"], a, w["w_s_skinny"], cu_count, group_size, zp, None
        )

    if provider.startswith("triton-w4a16"):
        return lambda: _k.triton_w4a16_skinny_fmt_gemm(
            a,
            w["w_q_skinny_i32"],
            w["w_s_skinny"],
            group_size,
            packed_scale_zp=carrier,
        )

    if provider.startswith("hipblaslt-w4a16"):
        from vllm.model_executor.kernels.linear.mixed_precision import (
            hipblaslt_w4a16 as _h,
        )

        # hipBLASLt takes the scales and zero-points as one allocation; the
        # layer builds the same buffer once at load time.
        scale, _ = _h.build_scale_buffer(w["w_s_skinny"], zp, group_size)
        return lambda: _h.hipblaslt_w4a16_gemm(
            a, w["w_q_skinny"], scale, group_size, asym
        )

    return None


def benchmark(batch_size, provider, N, K, group_size, weights, hot):
    M = batch_size
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]
    run = _make_runner(provider, a, weights, group_size)
    if run is None:
        return 0.0, 0.0, 0.0

    # do_bench flushes a 256 MiB buffer between reps, so the weights are read
    # from DRAM as they are in a real decode step. do_bench_cudagraph does not,
    # which for these weight sizes means measuring out of the 32 MiB MALL.
    bench = triton.testing.do_bench_cudagraph if hot else triton.testing.do_bench
    try:
        ms, min_ms, max_ms = bench(run, quantiles=quantiles)
    except RuntimeError as e:
        # The HIP op rejects shapes outside its supported range (N_in > 5 or
        # K*N over the medium LDS limit); report those as 0 rather than abort.
        print(f"  {provider} M={M}: {e}")
        return 0.0, 0.0, 0.0

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


def make_report(providers, batch_sizes):
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=batch_sizes,
            x_log=False,
            line_arg="provider",
            line_vals=providers,
            line_names=providers,
            ylabel="TFLOP/s (larger is better)",
            plot_name="FP16 vs Hybrid W4A16",
            args={},
        )
    )(benchmark)


def prepare_shapes(args):
    KN_model_names = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            KN.append(model)
            KN_model_names.append(KN)
    return KN_model_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RDNAHybridW4A16LinearKernel"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["Qwen/Qwen3-4B"],
        choices=list(WEIGHT_SHAPES.keys()),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        type=str,
        default=PROVIDERS,
        choices=PROVIDERS + FORCED_PROVIDERS,
    )
    parser.add_argument(
        "--hot",
        action="store_true",
        help="time with do_bench_cudagraph (no cache flush) instead of do_bench",
    )
    args = parser.parse_args()

    report = make_report(args.providers, args.batch_sizes)

    for K, N, model in prepare_shapes(args):
        group_size = args.group_size
        print(f"\n{'=' * 70}")
        print(f"{model}, N={N} K={K}, group_size={group_size}")
        print(f"{'=' * 70}")

        weights = prepare_hybrid_weights(K, N, group_size)

        save_path = args.save_path or f"bench_int4_res_n{N}_k{K}"
        os.makedirs(save_path, exist_ok=True)
        report.run(
            print_data=True,
            show_plots=False,
            save_path=save_path,
            N=N,
            K=K,
            group_size=group_size,
            weights=weights,
            hot=args.hot,
        )

    print("\nBenchmark finished!")
