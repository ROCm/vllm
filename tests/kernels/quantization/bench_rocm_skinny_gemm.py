#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the bf16/fp16 wvSplitK skinny GEMM kernel.

Measures throughput and weight bandwidth for representative model shapes
at batch sizes 1-4. Validates accuracy against torch.mm. Dynamically
determines iteration count per shape based on IQR convergence.

Per-kernel timestamps are recorded inside a captured CUDA graph via a small
ctypes shim that calls hipEventRecordWithFlags(hipEventRecordExternal) —
PyTorch's high-level torch.cuda.Event blocks this path on ROCm (AIESW-34641).

Usage:
    python tests/kernels/quantization/bench_rocm_skinny_gemm.py
    python tests/kernels/quantization/bench_rocm_skinny_gemm.py --dtype bf16
    python tests/kernels/quantization/bench_rocm_skinny_gemm.py --batch-sizes 1 4
    python tests/kernels/quantization/bench_rocm_skinny_gemm.py --shapes 4096x4096
"""

import argparse
import ctypes
import math
import os
import time

import torch

import vllm._custom_ops as ops
from vllm.utils.platform_utils import num_compute_units as get_cu_count

# Infinity Cache on Strix Halo is 32-64MB depending on SKU.
# Use a conservative estimate to ensure we bust L3.
CACHE_SIZE_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# HIP ctypes shim — workaround for PyTorch's blanket disable of
# cudaEventRecordExternal on ROCm (see AIESW-34641). Lets us record per-kernel
# events inside a captured CUDA graph and read back queryable timestamps.
# Remove once PyTorch upstream lifts the TORCH_CHECK in c10/cuda/CUDAEvent.h.
# ---------------------------------------------------------------------------
HIP_EVENT_RECORD_EXTERNAL = 0x01


def _load_hip():
    site = os.path.dirname(os.path.dirname(torch.__file__))
    for sub in ("_rocm_sdk_core/lib", "_rocm_sdk_devel/lib"):
        for name in ("libamdhip64.so.7", "libamdhip64.so"):
            p = os.path.join(site, sub, name)
            if os.path.exists(p):
                lib = ctypes.CDLL(p)
                lib.hipEventRecordWithFlags.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_uint,
                ]
                lib.hipEventRecordWithFlags.restype = ctypes.c_int
                lib.hipEventElapsedTime.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
                lib.hipEventElapsedTime.restype = ctypes.c_int
                return lib
    raise RuntimeError("libamdhip64 not found under torch site-packages")


_HIP = _load_hip()


def _record_external(ev: torch.cuda.Event, stream) -> None:
    """Record `ev` on `stream` with hipEventRecordExternal (graph-safe)."""
    err = _HIP.hipEventRecordWithFlags(
        int(ev.cuda_event), int(stream.cuda_stream), HIP_EVENT_RECORD_EXTERNAL
    )
    if err != 0:
        raise RuntimeError(f"hipEventRecordWithFlags returned {err}")


def _elapsed_ms(start_ev: torch.cuda.Event, end_ev: torch.cuda.Event) -> float:
    ms = ctypes.c_float(-1.0)
    err = _HIP.hipEventElapsedTime(
        ctypes.byref(ms), int(start_ev.cuda_event), int(end_ev.cuda_event)
    )
    if err != 0:
        raise RuntimeError(f"hipEventElapsedTime returned {err}")
    return ms.value


def _make_event():
    """Create a timing event and force lazy hipEventCreate by recording once."""
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


SHAPES = [
    # Qwen3-4B / Qwen3-VL-4B (identical backbone)
    (6144, 2560, "Qwen3-4B qkv"),
    (2560, 4096, "Qwen3-4B o_proj"),
    (19456, 2560, "Qwen3-4B gate_up"),
    (2560, 9728, "Qwen3-4B down"),
    (151936, 2560, "Qwen3-4B lm_head"),
    # Qwen2.5-VL-7B
    (4608, 3584, "Qwen2.5VL-7B qkv"),
    (3584, 3584, "Qwen2.5VL-7B o_proj"),
    (37888, 3584, "Qwen2.5VL-7B gate_up"),
    (3584, 18944, "Qwen2.5VL-7B down"),
    (152064, 3584, "Qwen2.5VL-7B lm_head"),
    # Qwen3.5-35B-A3B (vocab=248320, hidden=2048).  The (M, K=2048, N=1)
    # shapes here are the ones the gfx11 dispatcher routes through the
    # tuned (W=32, AC=16, YT=1, UN=8) branch added by this PR.
    (256, 2048, "Qwen3.5-35B-A3B router gate"),
    (1024, 2048, "Qwen3.5-35B-A3B shared gate_up"),
    (2048, 512, "Qwen3.5-35B-A3B shared down"),
    (248320, 2048, "Qwen3.5-35B-A3B lm_head"),
    # Llama-3.1-8B (hidden=4096, intermediate=14336, vocab=128256)
    (4096, 4096, "Llama-8B q/o_proj"),
    (6144, 4096, "Llama-8B qkv"),
    (28672, 4096, "Llama-8B gate_up"),
    (4096, 14336, "Llama-8B down"),
    (128256, 4096, "Llama-8B lm_head"),
]


def _median_se(times_sorted):
    """Standard error of the median as % of median, using MAD estimator."""
    n = len(times_sorted)
    med = times_sorted[n // 2]
    if med == 0 or n < 3:
        return med, 0.0
    mad = sorted(abs(t - med) for t in times_sorted)[n // 2]
    # SE_median ≈ 1.253 * σ / √n, with σ ≈ 1.4826 * MAD
    se = 1.253 * 1.4826 * mad / math.sqrt(n)
    return med, se / med * 100


def bench_dynamic(
    fn,
    target_se_pct=0.2,
    min_replays=4,
    max_replays=40,
    max_time_s=1.0,
    target_replay_ms=20.0,
):
    """Benchmark fn with per-kernel timing inside a captured CUDA graph.

    Probes the kernel time, sizes one capture so a replay runs ~target_replay_ms
    (so the GPU stays continuously busy and DVFS doesn't drop the clock between
    launches), captures `iters_per_replay` calls of fn(0..iters-1), each
    bracketed by hipEventRecord(EXTERNAL) so per-kernel timestamps are queryable
    on replay. fn(i) lets callers rotate weight buffers.

    Returns (median_ms_per_kernel, num_samples, se_pct).
    """
    # 1) Probe one kernel to size the graph.
    fn(0)
    torch.accelerator.synchronize()
    probe_start = torch.Event(enable_timing=True)
    probe_end = torch.Event(enable_timing=True)
    probe_start.record()
    fn(0)
    probe_end.record()
    torch.accelerator.synchronize()
    probe_ms = max(probe_start.elapsed_time(probe_end), 1e-3)
    iters_per_replay = max(2, min(2000, int(target_replay_ms / probe_ms)))

    # 2) Allocate a chain of iters_per_replay+1 events. The i-th per-kernel
    #    time is events[i].elapsed_time(events[i+1]). Force handle creation
    #    on the default stream so the underlying hipEvent_t exists before
    #    the stream-capture region.
    events = [_make_event() for _ in range(iters_per_replay + 1)]
    torch.accelerator.synchronize()

    # 3) Capture on a side stream, recording the event chain with EXTERNAL.
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        _record_external(events[0], s)
        for i in range(iters_per_replay):
            fn(i)
            _record_external(events[i + 1], s)

    # Warm one replay to absorb first-launch cost.
    g.replay()
    torch.accelerator.synchronize()

    # 4) Time replays adaptively. Each replay yields iters_per_replay samples.
    samples = []
    wall_start = time.monotonic()
    for r in range(max_replays):
        g.replay()
        torch.accelerator.synchronize()
        for i in range(iters_per_replay):
            samples.append(_elapsed_ms(events[i], events[i + 1]))

        if r + 1 >= min_replays:
            med, se_pct = _median_se(sorted(samples))
            if se_pct < target_se_pct:
                return med, len(samples), se_pct
            if time.monotonic() - wall_start > max_time_s:
                return med, len(samples), se_pct

    med, se_pct = _median_se(sorted(samples))
    return med, len(samples), se_pct


def parse_shape(s):
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be MxK, got '{s}'")
    return (int(parts[0]), int(parts[1]), s)


def run_bench(shapes, batch_sizes, dtype, target_se_pct):
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"

    print(f"GPU: {gpu_name}, CU count: {cu_count}")
    print(f"dtype: {dtype_name}, target SE: {target_se_pct}%")
    print(f"Shapes: {len(shapes)}, Batch sizes: {batch_sizes}")
    print()

    print(f"{'N':>2} {'M':>6}x{'K':<6} {'Label':<22} {'med_us':>9} {'med_GiB/s':>10}")
    print("-" * 60)

    t0 = time.time()
    for M, K, label in shapes:
        for N in batch_sizes:
            xavier = math.sqrt(2 / K)
            weight = (torch.rand(M, K, dtype=dtype, device="cuda") * 2 - 1) * xavier
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") * 2 - 1) * xavier

            ref_out = torch.mm(activation, weight.t())
            out = ops.wvSplitK(weight, activation, cu_count)
            atol = max(1e-3, torch.finfo(dtype).eps * math.sqrt(K))
            torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-2)

            weight_bytes = M * K * dtype.itemsize
            n_bufs = max(1, CACHE_SIZE_BYTES // weight_bytes + 1)
            weights = [
                (torch.rand(M, K, dtype=dtype, device="cuda") * 2 - 1) * xavier
                for _ in range(n_bufs)
            ]

            fn = lambda i, ws=weights, a=activation: ops.wvSplitK(
                ws[i % len(ws)], a, cu_count
            )
            med_ms, _, _ = bench_dynamic(
                fn,
                target_se_pct=target_se_pct,
            )
            med_us = med_ms * 1000
            med_bw = weight_bytes / (med_ms * 1e-3) / (1 << 30)

            print(f"{N:>2} {M:>6}x{K:<6} {label:<22} {med_us:>8.1f} {med_bw:>9.1f}")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark bf16/fp16 wvSplitK skinny GEMM"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Batch sizes (N) to test (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=None,
        help="Shapes as MxK (default: all representative shapes)",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="bf16",
        help="Data type (default: bf16)",
    )
    parser.add_argument(
        "--target-se",
        type=float,
        default=0.1,
        help="Stop when SE of median < this %% of median (default: 0.1)",
    )
    args = parser.parse_args()

    shapes = args.shapes if args.shapes else SHAPES
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    run_bench(shapes, args.batch_sizes, dtype, args.target_se)


if __name__ == "__main__":
    main()
