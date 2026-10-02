#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoPE and the KV cache write of a decode step, for models without a Q/K
norm: our fused kernel against what runs today and against AITER's.

On gfx1151 a decode step rotates Q and K in an Inductor kernel compiled from
RotaryEmbedding.forward_native (custom ops are off under VLLM_COMPILE) and
then writes K and V with triton_reshape_and_cache_flash.  This
times, for nseq sequences decoding M tokens each, on the packed HND cache the
backend hands over:

    now     Inductor RoPE from qkv, then the Triton writer
    aiter   AITER's Triton fused_qk_rope_reshape_and_cache, as vLLM's
            fuse_rope_kvcache calls it (in place, flash layout)
    rdna35  _rocm_C.rdna35_rope_cache, in place

Each with triton's do_bench (eager: includes the host launch) and
do_bench_cudagraph (device only, as in a graphed decode step).  rdna35's Q and
cache rows are checked against `now`.

    cd <worktree> && amd-gpu-lock <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/rope_cache.py
"""

import argparse
import pathlib
import sys

import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.triton_utils import triton
from vllm.v1.attention.ops.rdna35_rope_cache import rdna35_rope_cache
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import shapeset  # noqa: E402

BLOCK_SIZE = 16
CACHE_MIB = 256
MAX_POS = 8192


def cos_sin_cache(rot, dtype):
    inv = 1.0 / (1e6 ** (torch.arange(0, rot, 2, dtype=torch.float) / rot))
    f = torch.outer(torch.arange(MAX_POS, dtype=torch.float), inv)
    return torch.cat((f.cos(), f.sin()), dim=-1).to(dtype).cuda()


def compile_like_vllm(f):
    """torch.compile with the Inductor options of vLLM's compilation config:
    combo kernels merge the independent Q and K kernels as in the model."""
    return torch.compile(
        f,
        dynamic=False,
        fullgraph=True,
        options={"combo_kernels": True, "benchmark_combo_kernel": True},
    )


def inductor_rope(hq, hkv, d, cs):
    """Q and K out of qkv, rotated by the native RoPE, compiled by Inductor."""

    def f(pos, qkv):
        q, k, _ = qkv.split([hq * d, hkv * d, hkv * d], dim=-1)
        return RotaryEmbedding.forward_static(pos, q, k, d, d, cs, True)

    return compile_like_vllm(f)


def setup(hq, hkv, d, ctx, nseq, m, dtype):
    """nseq sequences decoding their last m tokens of a ctx-token context."""
    t = nseq * m
    qkv = torch.randn(t, (hq + 2 * hkv) * d, dtype=dtype, device="cuda")
    block_bytes = hkv * BLOCK_SIZE * 2 * d * qkv.element_size()
    blocks = max(t, CACHE_MIB * 1024**2 // block_bytes)
    kv_cache = torch.zeros(blocks, hkv, BLOCK_SIZE, 2 * d, dtype=dtype, device="cuda")
    pos = (ctx - m + torch.arange(m, device="cuda")).repeat(nseq)
    # Each sequence starts at a block of its own; its tokens follow their
    # positions from there.
    span = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    first = torch.randperm(blocks // span, device="cuda")[:nseq] * span
    slots = first.repeat_interleave(m) * BLOCK_SIZE + pos
    return qkv, kv_cache, pos, slots


def pipelines(rope, hq, hkv, d, qkv, kv_cache, pos, slots, cs, aiter_rope):
    t = qkv.shape[0]
    key_cache, value_cache = kv_cache.transpose(1, 2).split(d, dim=-1)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    q, k, v = (x.view(t, -1, d) for x in qkv.split([hq * d, hkv * d, hkv * d], dim=-1))

    def now():
        _, key = rope(pos, qkv)
        triton_reshape_and_cache_flash(
            key.view(t, hkv, d), v, key_cache, value_cache, slots, "auto", one, one
        )

    def rdna35():
        rdna35_rope_cache(pos, q, k, v, kv_cache, slots, cs)

    fns = {"now": now, "rdna35": rdna35}
    # AITER's kernel takes a bf16 cache only, and power-of-two head sizes.
    if aiter_rope is not None and qkv.dtype == torch.bfloat16 and d & (d - 1) == 0:

        def aiter():
            aiter_rope(
                q, k, v, pos, cs, True, key_cache, value_cache, slots, one, one,
                True, False,
            )  # fmt: skip

        fns["aiter"] = aiter
    return fns


def check(rope, fns, hkv, d, qkv, kv_cache, pos, slots):
    """Max abs difference of rdna35's Q and K cache rows from `now`."""
    q_ref, k_ref = rope(pos, qkv)
    t = qkv.shape[0]
    x = qkv.clone()
    fns["rdna35"]()
    q_got = qkv[:, : q_ref.shape[1]]
    b, o = slots // BLOCK_SIZE, slots % BLOCK_SIZE
    cached = kv_cache[b, :, o, :d]
    diff = max(
        (q_got.float() - q_ref.float()).abs().max().item(),
        (cached.float() - k_ref.view(t, hkv, d).float()).abs().max().item(),
    )
    qkv.copy_(x)
    return diff


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    shapeset.add_arguments(p)
    shapeset.add_dtype_argument(p)
    p.add_argument("--ctx", type=int, nargs="+", default=[128, 1024])
    p.add_argument("--nseq", type=int, nargs="+", default=[1, 4])
    p.add_argument("--m", type=int, nargs="+", default=[1, 8])
    p.add_argument("--no-aiter", action="store_true")
    args = p.parse_args()
    dtype = shapeset.torch_dtype(args.dtype)

    aiter_rope = None
    if not args.no_aiter:
        from vllm._aiter_ops import rocm_aiter_ops

        aiter_rope = rocm_aiter_ops.triton_rope_and_cache

    shapes, _ = shapeset.load(args)
    args.windowed = not args.windowed
    shapes += shapeset.load(args)[0]
    configs = sorted({(s.hq, s.hkv, s.d) for s in shapes})

    names = ["now", "aiter", "rdna35"]
    print(f"# {args.dtype}, block {BLOCK_SIZE}, us, eager / graph")
    print(
        "| Hq/Hkv/D | ctx | nseq | M | "
        + " | ".join(names)
        + " | now/rdna35 graph | aiter/rdna35 graph | aiter/rdna35 eager "
        "| max diff |"
    )
    print("|" + " --- |" * (len(names) + 8))
    for hq, hkv, d in configs:
        torch._dynamo.reset()
        cs = cos_sin_cache(d, dtype)
        rope = inductor_rope(hq, hkv, d, cs)
        for nseq in args.nseq:
            for m in args.m:
                for ctx in args.ctx:
                    qkv, kv_cache, pos, slots = setup(hq, hkv, d, ctx, nseq, m, dtype)
                    fns = pipelines(
                        rope, hq, hkv, d, qkv, kv_cache, pos, slots, cs, aiter_rope
                    )
                    diff = check(rope, fns, hkv, d, qkv, kv_cache, pos, slots)
                    eager, graph = {}, {}
                    for k, f in fns.items():
                        f()
                        eager[k] = (
                            triton.testing.do_bench(f, return_mode="median") * 1e3
                        )
                        graph[k] = (
                            triton.testing.do_bench_cudagraph(f, return_mode="median")
                            * 1e3
                        )
                        torch.accelerator.synchronize()
                    cells = " | ".join(
                        f"{eager[k]:.1f} / {graph[k]:.2f}" if k in eager else "-"
                        for k in names
                    )
                    vs_aiter = (
                        f"{graph['aiter'] / graph['rdna35']:.2f} "
                        f"| {eager['aiter'] / eager['rdna35']:.2f}"
                        if "aiter" in graph
                        else "- | -"
                    )
                    print(
                        f"| {hq}/{hkv}/{d} | {ctx} | {nseq} | {m} | {cells} "
                        f"| {graph['now'] / graph['rdna35']:.2f} | {vs_aiter} "
                        f"| {diff:.2g} |",
                        flush=True,
                    )
                    del kv_cache


if __name__ == "__main__":
    main()
