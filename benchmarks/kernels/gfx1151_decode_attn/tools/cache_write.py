#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV cache write of a decode step: Triton's writer against the HIP one.

ROCM_ATTN on gfx1151 uses TritonAttentionImpl.do_kv_cache_update, which calls
triton_reshape_and_cache_flash.  This times it against
torch.ops._C_cache_ops.reshape_and_cache_flash on the cache exactly as the
backend hands it over: packed HND, (blocks, Hkv, block, 2*D), K and V as the
two halves of the last dimension, and key/value as views of one qkv tensor.
Both writers must leave the cache bit-identical, or the row is not printed.

Per (Hq, Hkv, D) and token count it reports the device time under a HIP graph
(as matrix.py times attention) and the host time of one eager call (what a
step outside a full CUDA graph pays per layer).

    cd <worktree> && amd-gpu-lock <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/cache_write.py
"""

import argparse
import math
import pathlib
import sys
import time

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import shapeset  # noqa: E402

BLOCK_SIZE = 16
CACHE_MIB = 256


def setup(hq, hkv, d, tokens, dtype):
    qkv = torch.randn(tokens, (hq + 2 * hkv) * d, dtype=dtype, device="cuda")
    _, key, value = qkv.split([hq * d, hkv * d, hkv * d], dim=-1)
    key = key.view(tokens, hkv, d)
    value = value.view(tokens, hkv, d)
    block_bytes = hkv * BLOCK_SIZE * 2 * d * qkv.element_size()
    blocks = max(tokens, CACHE_MIB * 1024**2 // block_bytes)
    kv_cache = torch.zeros(blocks, hkv, BLOCK_SIZE, 2 * d, dtype=dtype, device="cuda")
    # One token per block, in random blocks: a batch of decodes.
    slots = torch.randperm(blocks, device="cuda")[:tokens] * BLOCK_SIZE
    slots += torch.randint(0, BLOCK_SIZE, (tokens,), device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    return key, value, kv_cache, slots, scale


def writers(key, value, kv_cache, slots, scale):
    # As TritonAttentionImpl.do_kv_cache_update views the cache.
    d = key.shape[-1]
    key_cache, value_cache = kv_cache.transpose(1, 2).split(d, dim=-1)

    def tri():
        triton_reshape_and_cache_flash(
            key, value, key_cache, value_cache, slots, "auto", scale, scale
        )

    def hip():
        ops.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slots, "auto", scale, scale
        )

    return {"triton": tri, "hip": hip}


def host_us(fn, calls=2000):
    for _ in range(50):
        fn()
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    t1 = time.perf_counter()
    torch.accelerator.synchronize()
    return (t1 - t0) / calls * 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    shapeset.add_arguments(p)
    shapeset.add_dtype_argument(p)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 32, 128])
    args = p.parse_args()
    dtype = shapeset.torch_dtype(args.dtype)

    shapes, _ = shapeset.load(args)
    args.windowed = not args.windowed
    shapes += shapeset.load(args)[0]
    configs = sorted({(s.hq, s.hkv, s.d) for s in shapes})

    print(f"# {args.dtype}, block {BLOCK_SIZE}, packed HND, {shapeset.describe(args)}")
    print(
        "| Hq/Hkv/D | tokens | triton us | hip us | dev x "
        "| triton host us | hip host us |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    ratios = []
    for hq, hkv, d in configs:
        for t in args.tokens:
            key, value, kv_cache, slots, scale = setup(hq, hkv, d, t, dtype)
            fns = writers(key, value, kv_cache, slots, scale)
            fns["triton"]()
            ref = kv_cache.clone()
            kv_cache.zero_()
            fns["hip"]()
            if not torch.equal(ref, kv_cache):
                print(f"| {hq}/{hkv}/{d} | {t} | MISMATCH |")
                continue
            dev = {
                k: triton.testing.do_bench_cudagraph(f, return_mode="median") * 1e3
                for k, f in fns.items()
            }
            host = {k: host_us(f) for k, f in fns.items()}
            ratios.append(dev["triton"] / dev["hip"])
            print(
                f"| {hq}/{hkv}/{d} | {t} | {dev['triton']:.2f} | {dev['hip']:.2f} "
                f"| {ratios[-1]:.2f} | {host['triton']:.1f} | {host['hip']:.1f} |"
            )
            del kv_cache
    geo = math.exp(sum(map(math.log, ratios)) / len(ratios))
    print(f"\ndevice time triton/hip, geomean over {len(ratios)} cells: {geo:.3f}x")


if __name__ == "__main__":
    main()
