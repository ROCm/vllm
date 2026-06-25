#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""rdna_moe_gemm-vs-Triton across routing archetypes, for both MoE prefill GEMMs.

gemm1 (up/gate, top_k=8, K=hidden=2048 N=intermediate=1024) and gemm2 (down,
top_k=1, K=intermediate=512 N=hidden=2048) of the W4A16 MoE share a single
moe_align@32 layout. For each (archetype, T) from ``gen_distributions.synth_topk``
this times ``torch.ops._rocm_C.moe_gemm_w4a16`` against the Triton reference
(``invoke_fused_moe_kernel_hybrid_triton``) on identical inputs, so the kernel's
win/loss can be correlated with the per-block fill of the routing. Triton is the
correctness oracle (``rel`` = kernel-vs-Triton).

The W4A16 weights/scales are routing-independent, so they are built once per gemm
and reused; only the routing and A change per case. Requires gfx11 (RDNA3 WMMA);
off gfx11 the compiled op is a stub.

  python bench_moe_gemm.py                       # both gemms, all archetypes
  python bench_moe_gemm.py --gemm 1              # gemm1 only
  python bench_moe_gemm.py --arches zipf2,hot16 --ts 256,994
"""

import os
import sys

import numpy as np
import torch

# Import the sibling routing generator (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_distributions import Archetype, synth_topk  # noqa: E402

from vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16 import (  # noqa: E402
    pack_int4_exllama_shuffle,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    invoke_fused_moe_kernel_hybrid_triton,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

# Shared routing/quant params (must match gen_distributions: E=256, top_k=8).
E, ROUTING_TOPK, G = 256, 8, 128
BLOCK_M = 32  # the only block_m the rdna_moe_gemm op supports
DEV, DT = "cuda", torch.bfloat16

# Per-gemm shape and tuned Triton reference config (block_m=32). ``op_top_k`` is
# the op's top_k arg; ``a_rows(T)`` is the number of activation rows the gemm
# consumes (gemm1 reads per-token, gemm2 reads per-routed-slot).
GEMM_SPECS = {
    1: dict(K=2048, N=1024, op_top_k=ROUTING_TOPK, tri_k=128, tri_gm=4, a_rows=1),
    2: dict(K=512, N=2048, op_top_k=1, tri_k=64, tri_gm=1, a_rows=ROUTING_TOPK),
}


def time_us(fn):
    # do_bench flushes the L2 cache between iterations, so each launch starts
    # cold -- it measures one isolated launch (the production pattern, where the
    # same gemm does not run back-to-back) rather than rewarding cross-launch
    # weight cache-reuse that real inference never sees.
    return triton.testing.do_bench(fn, return_mode="median") * 1000.0


def ramp_clocks():
    # This APU starts at low clocks; without a ramp the first do_bench call reads
    # inflated and biases whichever op runs first.
    w = torch.randn(4096, 4096, dtype=DT, device=DEV)
    for _ in range(60):
        w = w @ w.t() * 1e-4 + 0.1
    torch.accelerator.synchronize()


def run_gemm(gemm, arches, ts, seed):
    spec = GEMM_SPECS[gemm]
    K, N = spec["K"], spec["N"]
    op_top_k = spec["op_top_k"]
    triton_cfg = dict(
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=spec["tri_k"],
        GROUP_SIZE_M=spec["tri_gm"],
        num_warps=2,
        num_stages=1,
    )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)  # routing rng (so --seed varies routing too)
    print(f"\n### gemm{gemm}: building weights E={E} N={N} K={K} (packed int4)...")
    w_uint4 = torch.randint(0, 16, (E, N, K), dtype=torch.int32, device=DEV)
    w_packed = torch.stack([pack_int4_exllama_shuffle(w_uint4[e]) for e in range(E)])
    w_scale = torch.randn(E, N, K // G, dtype=DT, device=DEV).abs() * 0.01
    del w_uint4

    hdr = (
        f"{'arch':9s} {'T':>5s} {'nvt':>6s} {'blocks':>6s} {'use%':>5s} | "
        f"{'triton_us':>9s} {'kernel_us':>9s} {'speedup%':>8s} "
        f"{'k_TFLOP/s':>9s} {'rel':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name in arches:
        for T in ts:
            topk_ids = (
                torch.from_numpy(synth_topk(name, T, rng=rng)).to(torch.int32).to(DEV)
            )
            sti, eid, ntpp_t = moe_align_block_size(
                topk_ids, BLOCK_M, E, ignore_invalid_experts=True
            )
            ntpp = int(ntpp_t.item())
            num_slots = sti.size(0)
            nvt = T * ROUTING_TOPK  # useful routed rows (== op's n_valid_tokens)
            # valid_blocks = num_tokens_post_padded // block_m; the kernel computes
            # every row of these blocks, so padded_rows is the real GEMM work.
            valid_blocks = ntpp // BLOCK_M
            padded_rows = valid_blocks * BLOCK_M
            use = 100.0 * nvt / padded_rows if padded_rows else 0.0
            A = torch.randn(spec["a_rows"] * T, K, dtype=DT, device=DEV) * 0.1

            C_ref = torch.zeros(num_slots, N, dtype=DT, device=DEV)

            def run_triton(C=C_ref, sti=sti, eid=eid, ntpp_t=ntpp_t, A=A):
                invoke_fused_moe_kernel_hybrid_triton(
                    A=A,
                    B=w_packed,
                    C=C,
                    B_scale=w_scale,
                    topk_weights=None,
                    sorted_token_ids=sti,
                    expert_ids=eid,
                    num_tokens_post_padded=ntpp_t,
                    mul_routed_weight=False,
                    top_k=op_top_k,
                    config=triton_cfg,
                    compute_type=tl.bfloat16,
                    group_size=G,
                    align_block_size_m=BLOCK_M,
                )

            run_triton()
            torch.accelerator.synchronize()
            C_ref = C_ref.clone()  # freeze reference

            C_k = torch.zeros(num_slots, N, dtype=DT, device=DEV)

            def run_kernel(C=C_k, sti=sti, eid=eid, nb=valid_blocks, A=A, nvt=nvt):
                torch.ops._rocm_C.moe_gemm_w4a16(
                    A, w_packed, w_scale, sti, eid, C, nvt, op_top_k, BLOCK_M, nb
                )

            run_kernel()
            torch.accelerator.synchronize()
            rel = (
                (C_k.float() - C_ref.float()).abs().sum()
                / (C_ref.float().abs().sum() + 1e-9)
            ).item()

            t_us = time_us(run_triton)
            k_us = time_us(run_kernel)
            spd = (t_us / k_us - 1) * 100.0
            # Real GEMM work = padded_rows (valid_blocks * block_m), not just nvt:
            # the kernel computes the padded rows too. FLOPs = 2 * M * N * K.
            k_tflops = 2 * padded_rows * N * K / (k_us * 1e6)
            flag = "" if rel < 0.01 else " BADREL"
            print(
                f"{name.value:9s} {T:5d} {nvt:6d} {valid_blocks:6d} {use:5.0f} | "
                f"{t_us:9.1f} {k_us:9.1f} {spd:+8.1f} {k_tflops:9.1f} "
                f"{rel:7.4f}{flag}",
                flush=True,
            )
            rows.append((spd, name, T, use, t_us, k_us, rel))

    print(f"\n==== gemm{gemm} kernel vs Triton, worst-first ====")
    for spd, name, T, use, t_us, k_us, rel in sorted(rows):
        print(
            f"  {spd:+6.1f}%  {name.value:9s} T={T:<5d} use={use:3.0f}% "
            f"(triton {t_us:.0f} / kernel {k_us:.0f} us) rel={rel:.4f}"
        )


def main():
    ap = FlexibleArgumentParser(description=__doc__)
    ap.add_argument("--gemm", default="1,2", help="which gemms to bench: 1, 2, or 1,2")
    ap.add_argument("--arches", default=",".join(a.value for a in Archetype))
    ap.add_argument("--ts", default="16,64,128,256,512,768,994,1024,1536,2048,4096")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("HIP/CUDA not available")
    if not hasattr(torch.ops._rocm_C, "moe_gemm_w4a16"):
        raise SystemExit("torch.ops._rocm_C.moe_gemm_w4a16 not available (need gfx11)")

    gemms = [int(g) for g in args.gemm.split(",")]
    arches = [Archetype(x) for x in args.arches.split(",")]
    ts = [int(x) for x in args.ts.split(",")]
    ramp_clocks()
    for gemm in gemms:
        run_gemm(gemm, arches, ts, args.seed)


if __name__ == "__main__":
    main()
