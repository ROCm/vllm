#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Q/K RMSNorm, RoPE and the KV cache write of a decode step: our fused
kernel against what runs today on gfx1151 and against AITER's.

Models with a q/k norm before RoPE (Qwen3, Qwen3.5/3.6, Gemma 3/4) run it in
Inductor kernels compiled from the native ops, then the Triton cache writer.
This times, for nseq sequences decoding M tokens each:

    now     Inductor q/k norm + RoPE from qkv, then the Triton writer on the
            packed HND cache ROCM_ATTN uses on gfx1151
    aiter   fused_qk_norm_rope_cache_pts_quant_shuffle, called as vLLM calls
            it (q_out, k_out), on the cache it writes: NHD within a block,
            (blocks, 2, block, H, D); not built for D=96 or D=512
    rdna35  _rocm_C.rdna35_rope_cache on the packed HND cache, writing the
            same q_out and k_out

Each with triton's do_bench (eager: includes the host launch) and
do_bench_cudagraph (device only, as in a graphed decode step).  rdna35's Q, K
and cache rows are checked against `now`.

    cd <worktree> && amd-gpu-lock <venv>/bin/python \\
        benchmarks/kernels/gfx1151_decode_attn/tools/qknorm_rope_cache.py
"""

import argparse
import pathlib
import sys

import torch
from aiter.ops.fused_qk_norm_rope_cache_quant import (
    fused_qk_norm_rope_cache_pts_quant_shuffle,
)

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.triton_utils import triton
from vllm.v1.attention.ops.rdna35_rope_cache import rdna35_rope_cache
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import shapeset  # noqa: E402
from rope_cache import (  # noqa: E402
    BLOCK_SIZE,
    CACHE_MIB,
    compile_like_vllm,
    cos_sin_cache,
)

EPS = 1e-6
AITER_HEAD_SIZES = (64, 128, 256)


def rms_norm(x, weight, eps):
    """The native body of vllm.ir.ops.rms_norm, which is what Inductor fuses
    in the model; the IR op itself does not lower that way outside vLLM."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.to(weight.dtype) * weight).to(orig_dtype)


def inductor_qknorm_rope(hq, hkv, d, cs, qw, kw):
    """As Qwen3Attention.forward: split qkv, norm q and k per head, RoPE."""

    def f(pos, qkv):
        t = qkv.shape[0]
        q, k, _ = qkv.split([hq * d, hkv * d, hkv * d], dim=-1)
        q = rms_norm(q.view(t, hq, d), qw, EPS).view(t, hq * d)
        k = rms_norm(k.view(t, hkv, d), kw, EPS).view(t, hkv * d)
        return RotaryEmbedding.forward_static(pos, q, k, d, d, cs, True)

    return compile_like_vllm(f)


def setup(hq, hkv, d, ctx, nseq, m, dtype):
    """nseq sequences decoding their last m tokens of a ctx-token context."""
    t = nseq * m
    qkv = torch.randn(t, (hq + 2 * hkv) * d, dtype=dtype, device="cuda")
    block_bytes = hkv * BLOCK_SIZE * 2 * d * qkv.element_size()
    blocks = max(t, CACHE_MIB * 1024**2 // block_bytes)
    hnd = torch.zeros(blocks, hkv, BLOCK_SIZE, 2 * d, dtype=dtype, device="cuda")
    nhd = torch.zeros(blocks, 2, BLOCK_SIZE, hkv, d, dtype=dtype, device="cuda")
    pos = (ctx - m + torch.arange(m, device="cuda")).repeat(nseq)
    # Each sequence starts at a block of its own; its tokens follow their
    # positions from there.
    span = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    first = torch.randperm(blocks // span, device="cuda")[:nseq] * span
    slots = first.repeat_interleave(m) * BLOCK_SIZE + pos
    return qkv, hnd, nhd, pos, slots


def pipelines(rope, hq, hkv, d, qkv, hnd, nhd, pos, slots, cs, qw, kw):
    t = qkv.shape[0]
    key_cache, value_cache = hnd.transpose(1, 2).split(d, dim=-1)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    one_cpu = torch.ones(1, dtype=torch.float32)
    k_nhd, v_nhd = nhd.unbind(1)
    q_out = torch.empty(t, hq, d, dtype=qkv.dtype, device="cuda")
    k_out = torch.empty(t, hkv, d, dtype=qkv.dtype, device="cuda")
    q, k, v = (x.view(t, -1, d) for x in qkv.split([hq * d, hkv * d, hkv * d], dim=-1))

    def write(key):
        triton_reshape_and_cache_flash(
            key, v, key_cache, value_cache, slots, "auto", one, one
        )

    def now():
        _, key = rope(pos, qkv)
        write(key.view(t, hkv, d))

    def aiter():
        fused_qk_norm_rope_cache_pts_quant_shuffle(
            qkv, qw, kw, cs, pos, t, hq, hkv, hkv, d, True, EPS,
            q_out, k_nhd, v_nhd, slots, one_cpu, one_cpu, k_out, None,
            True, False, BLOCK_SIZE, 16 // qkv.element_size(), 0,
        )  # fmt: skip

    def rdna35():
        rdna35_rope_cache(
            pos, q, k, v, hnd, slots, cs, q_weight=qw, k_weight=kw, eps=EPS,
            q_out=q_out, k_out=k_out,
        )  # fmt: skip

    fns = {"now": now, "rdna35": rdna35}
    if d in AITER_HEAD_SIZES:
        fns["aiter"] = aiter
    return fns, (q_out, k_out)


def check(rope, fns, outs, hkv, d, qkv, hnd, pos, slots):
    """Max abs difference of rdna35's Q, K and cache rows from `now`."""
    q_ref, k_ref = rope(pos, qkv)
    t = qkv.shape[0]
    fns["rdna35"]()
    q_out, k_out = outs
    b, o = slots // BLOCK_SIZE, slots % BLOCK_SIZE
    cached = hnd[b, :, o, :d]
    return max(
        (q_out.flatten(1).float() - q_ref.float()).abs().max().item(),
        (k_out.flatten(1).float() - k_ref.float()).abs().max().item(),
        (cached.float() - k_ref.view(t, hkv, d).float()).abs().max().item(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    shapeset.add_arguments(p)
    shapeset.add_dtype_argument(p)
    p.add_argument("--ctx", type=int, nargs="+", default=[128, 1024])
    p.add_argument("--nseq", type=int, nargs="+", default=[1, 4])
    p.add_argument("--m", type=int, nargs="+", default=[1, 8])
    args = p.parse_args()
    dtype = shapeset.torch_dtype(args.dtype)

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
        qw = torch.rand(d, dtype=dtype, device="cuda") + 0.5
        kw = torch.rand(d, dtype=dtype, device="cuda") + 0.5
        rope = inductor_qknorm_rope(hq, hkv, d, cs, qw, kw)
        for nseq in args.nseq:
            for m in args.m:
                for ctx in args.ctx:
                    qkv, hnd, nhd, pos, slots = setup(hq, hkv, d, ctx, nseq, m, dtype)
                    fns, outs = pipelines(
                        rope, hq, hkv, d, qkv, hnd, nhd, pos, slots, cs, qw, kw
                    )
                    diff = check(rope, fns, outs, hkv, d, qkv, hnd, pos, slots)
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
                        # Alternating AITER's graphs and ours on
                        # do_bench_cudagraph's side stream without a sync
                        # faulted intermittently.
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
                    del hnd, nhd


if __name__ == "__main__":
    main()
