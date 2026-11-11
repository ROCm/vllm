#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-only sweep for the hybrid_w4a16 MoE prefill Triton kernel.

Calls ``invoke_fused_moe_kernel_hybrid_triton`` directly (skipping the
HybridW4A16MoEExperts.apply host setup so kernel time is not drowned
out, unlike benchmarks/kernels/bench_hybrid_w4a16_moe.py).  Launches
mimic HybridW4A16MoEExperts.apply: tokens are aligned with
``moe_align_block_size(topk_ids, ALIGN, ...)`` and, when a tile's
BLOCK_SIZE_M is smaller than ALIGN, ``expert_ids`` is repeat-interleaved
by ALIGN // BLOCK_SIZE_M and ALIGN is passed as ``align_block_size_m``
(EM cap).  The alignment is therefore decoupled from BLOCK_SIZE_M;
configs whose BLOCK_SIZE_M does not divide ALIGN are skipped.

Default shapes target Qwen3-30B-A3B / Qwen3-Omni-30B-A3B (E=128,
top_k=8, group_size=32, hidden=2048, moe_intermediate=768).  Override
with --n / --k / --e / --group-size to retune for other shapes.

Modes
-----
--mode tiles   Per-gemm tile sweep (BM must divide the alignment; configs
               valid at only some alignments are ranked after full-coverage
               ones, see the npts column).  Every config is timed at every
               (M, routing, align) point; configs are ranked by the
               geomean over points of (us / best us at that point), with
               the worst ratio.  Production tiles (from _triton_config)
               are marked with 'P'.
--mode align   For each (M, routing, align) time gemm1+gemm2
               back-to-back using the production _triton_config tiles
               (override with --gemm1-tile/--gemm2-tile), print the
               joint-us table with the best alignment per row, plus the
               num_tokens_post_padded (npp) and routed rows per point.

Routing models (--routing): ``random`` (fused_topk on randn scores),
``uniform`` (token t takes experts (t*topk + j) % E: perfectly even) and
``captured:<file>`` (output of capture_moe_routing.py; the captured
topk_ids for each M, timings averaged over captured layers).  M values
absent from a captured file are skipped for that routing.

Align values: integers, ``prod`` (production _select_block_size_m for
that M) or ``tied`` (alignment == the tile's BLOCK_SIZE_M, the legacy
behaviour; tiles mode only).

Usage (how the group_size <= SMALL_GROUP_SIZE_THRESHOLD alignments and
tiles in HybridW4A16MoEExperts were chosen; tuned on gfx1151):
    # Routing capture (once; ~minutes, needs the model in the HF cache)
    python benchmarks/kernels/capture_moe_routing.py \\
        --out /scratch/$USER/tmp/routing.problem

    # Alignment x M table (joint gemm1+gemm2, us)
    python benchmarks/kernels/sweep_hybrid_w4a16_moe_triton.py \\
        --mode align --m 6 128 512 1024 2048 --align 32 64 128 \\
        --routing random uniform captured:/scratch/$USER/tmp/routing.problem \\
        --csv /scratch/$USER/tmp/align.csv

    # gemm1 tile ranking at alignment 64 (production: < 2048 tokens)
    python benchmarks/kernels/sweep_hybrid_w4a16_moe_triton.py \\
        --mode tiles --gemms gemm1 --align 64 \\
        --m 6 16 32 64 128 256 512 1024 2048 3056 \\
        --routing random uniform captured:/scratch/$USER/tmp/routing.problem \\
        --block-m 16 32 64 --block-n 32 64 128 --group-m 1 --top 15

    # gemm1 tile ranking at alignment 128 (production: >= 2048 tokens)
    python benchmarks/kernels/sweep_hybrid_w4a16_moe_triton.py \\
        --mode tiles --gemms gemm1 --align 128 --m 2048 4096 8192 \\
        --routing random uniform \\
        --block-m 128 --block-n 32 64 128 --group-m 1 --top 12

    # gemm2 tile ranking at the production alignment for each M
    python benchmarks/kernels/sweep_hybrid_w4a16_moe_triton.py \\
        --mode tiles --gemms gemm2 --align prod \\
        --m 6 16 32 64 128 256 512 1024 2048 3056 \\
        --routing random uniform captured:/scratch/$USER/tmp/routing.problem \\
        --block-m 32 64 --block-n 32 64 128 --group-m 1 --top 15

    # Legacy single-M sweep (alignment tied to BLOCK_SIZE_M, random routing)
    python benchmarks/kernels/sweep_hybrid_w4a16_moe_triton.py --m 2048 \\
        --n 512 --k 2048 --e 256 --topk 8 --group-size 128
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.kernels.moe.test_hybrid_w4a16_moe import (  # noqa: E402
    _make_hybrid_moe_weights,
)
from vllm.model_executor.layers.fused_moe import fused_topk  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    invoke_fused_moe_kernel_hybrid_triton,
)
from vllm.model_executor.layers.fused_moe.hybrid_w4a16_moe import (  # noqa: E402
    HybridW4A16MoEExperts,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton  # noqa: E402

TILE_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
)
TILE_ALIASES = {
    "BM": "BLOCK_SIZE_M",
    "BN": "BLOCK_SIZE_N",
    "BK": "BLOCK_SIZE_K",
    "GM": "GROUP_SIZE_M",
    "nw": "num_warps",
    "ns": "num_stages",
}


def time_us(fn, warmup_ms: int = 25, rep_ms: int = 80) -> float:
    return (
        triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
        * 1000.0
    )


# --------------------------------------------------------------------------
# Production helpers (only the tile/alignment policy is reused; no weights).
# --------------------------------------------------------------------------
def make_policy(group_size: int) -> HybridW4A16MoEExperts:
    """HybridW4A16MoEExperts shell exposing _triton_config/_select_block_size_m."""
    obj = object.__new__(HybridW4A16MoEExperts)
    obj._group_size = group_size
    return obj


def prod_tile(policy, which: str, M: int, topk: int, K: int, N_inter: int, align: int):
    """Tile production would use: gemm1 contraction K, gemm2 contraction N_inter."""
    return policy._triton_config(K if which == "gemm1" else N_inter, M * topk, align)


def parse_tile_override(spec: str | None) -> dict:
    out: dict = {}
    if not spec:
        return out
    for item in spec.split(","):
        k, v = item.split("=")
        out[TILE_ALIASES.get(k.strip(), k.strip())] = int(v)
    return out


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def routing_topk_ids(
    name: str, M: int, E: int, topk: int, captured, device: str, dtype
) -> list[torch.Tensor] | None:
    """Return a list of topk_ids tensors ([M, topk] int32), one per layer."""
    if name == "random":
        hidden = torch.zeros(M, 1, device=device, dtype=dtype)
        scores = torch.randn(M, E, device=device, dtype=dtype)
        _, ids, _ = fused_topk(hidden, scores, topk, False)
        return [ids]
    if name == "uniform":
        t = torch.arange(M, device=device).unsqueeze(1)
        j = torch.arange(topk, device=device).unsqueeze(0)
        return [((t * topk + j) % E).to(torch.int32)]
    if name.startswith("captured:"):
        layers = captured["routing"].get(M)
        if not layers:
            return None
        return [ids.to(device=device, dtype=torch.int32) for ids in layers]
    raise ValueError(f"unknown routing {name!r}")


def load_captured(path: str, E: int, topk: int) -> dict:
    data = torch.load(path, map_location="cpu")
    meta = data.get("meta", {})
    if meta.get("num_experts", E) != E or meta.get("top_k", topk) != topk:
        print(f"WARNING: {path} captured with {meta}, sweep E={E} topk={topk}")
    data["routing"] = {int(k): v for k, v in data["routing"].items()}
    return data


# --------------------------------------------------------------------------
# Aligned MoE problems and kernel launches
# --------------------------------------------------------------------------
class AlignedMoEProblem:
    """Routing for one (M, routing, layer), aligned to one block size, plus the
    gemm1/gemm2 buffers; every tile config is timed against the same instance."""

    def __init__(self, W, M, topk_ids, align, E):
        self.M = M
        self.align = align
        self.topk = topk_ids.size(1)
        self.topk_ids = topk_ids
        dev, dtype = W["hidden"].device, W["hidden"].dtype
        self.hidden = W["hidden"][:M] if W["hidden"].size(0) >= M else None
        if self.hidden is None:
            self.hidden = torch.randn(M, W["K"], device=dev, dtype=dtype) / 10.0
        self.sorted_ids, self.expert_ids, self.npp = moe_align_block_size(
            topk_ids, align, E, None, ignore_invalid_experts=True
        )
        self.num_slots = self.sorted_ids.size(0)
        self.npp_val = int(self.npp.item())
        self.routed = M * self.topk
        self.touched = int(torch.unique(topk_ids).numel())
        N_w = W["w1"].size(1)
        self.gemm1_out = torch.empty(self.num_slots, N_w, device=dev, dtype=dtype)
        self.act = (
            torch.randn(self.num_slots, W["N_inter"], device=dev, dtype=dtype) / 10.0
        )
        self.gemm2_out = torch.empty(self.num_slots, W["K"], device=dev, dtype=dtype)
        self._eid_cache: dict[int, torch.Tensor] = {}

    def expert_ids_for(self, bm: int) -> torch.Tensor:
        """Same as apply()'s _expert_ids_for."""
        if bm == self.align:
            return self.expert_ids
        assert self.align % bm == 0
        if bm not in self._eid_cache:
            self._eid_cache[bm] = self.expert_ids.repeat_interleave(self.align // bm)
        return self._eid_cache[bm]


def launch_gemm1(W, problem: AlignedMoEProblem, cfg: dict) -> Callable[[], None]:
    eids = problem.expert_ids_for(cfg["BLOCK_SIZE_M"])

    def run():
        invoke_fused_moe_kernel_hybrid_triton(
            A=problem.hidden,
            B=W["w1"],
            C=problem.gemm1_out,
            B_scale=W["w1_s"],
            topk_weights=None,
            sorted_token_ids=problem.sorted_ids,
            expert_ids=eids,
            num_tokens_post_padded=problem.npp,
            mul_routed_weight=False,
            top_k=problem.topk,
            config=cfg,
            compute_type=W["compute_type"],
            group_size=W["group_size"],
            align_block_size_m=problem.align,
        )

    return run


def launch_gemm2(W, problem: AlignedMoEProblem, cfg: dict) -> Callable[[], None]:
    """apply()'s second call: slot-space A, top_k=1."""
    eids = problem.expert_ids_for(cfg["BLOCK_SIZE_M"])

    def run():
        invoke_fused_moe_kernel_hybrid_triton(
            A=problem.act,
            B=W["w2"],
            C=problem.gemm2_out,
            B_scale=W["w2_s"],
            topk_weights=None,
            sorted_token_ids=problem.sorted_ids,
            expert_ids=eids,
            num_tokens_post_padded=problem.npp,
            mul_routed_weight=False,
            top_k=1,
            config=cfg,
            compute_type=W["compute_type"],
            group_size=W["group_size"],
            align_block_size_m=problem.align,
        )

    return run


def build_weights(args, dtype=torch.float16, device="cuda") -> dict:
    gs = args.group_size
    w1, w1_s, _ = _make_hybrid_moe_weights(args.e, args.k, 2 * args.n, gs, device)
    w2, w2_s, _ = _make_hybrid_moe_weights(args.e, args.n, args.k, gs, device)
    return dict(
        w1=w1,
        w1_s=w1_s,
        w2=w2,
        w2_s=w2_s,
        K=args.k,
        N_inter=args.n,
        group_size=gs,
        hidden=torch.randn(max(args.m), args.k, device=device, dtype=dtype) / 10.0,
        compute_type=tl.float16 if dtype == torch.float16 else tl.bfloat16,
    )


def resolve_align(spec: str, policy, M, topk, E, bm=None) -> int:
    if spec == "prod":
        return policy._select_block_size_m(M, topk, E)
    if spec == "tied":
        assert bm is not None
        return bm
    return int(spec)


# --------------------------------------------------------------------------
# Tiles mode
# --------------------------------------------------------------------------
def tiles_mode(args, W, policy, captured, rows):
    skipped: set[int] = set()
    bm_list = [int(x) for x in args.block_m]
    bn_list = [int(x) for x in args.block_n]
    # BK is capped to group_size inside the wrapper; dedupe the equivalents.
    bk_list = sorted({min(int(x), args.group_size) for x in args.block_k})
    gm_list = [int(x) for x in args.group_m]
    nw_list = [int(x) for x in args.num_warps]
    ns_list = [int(x) for x in args.num_stages]
    cfgs = [
        dict(zip(TILE_KEYS, v))
        for v in itertools.product(bm_list, bn_list, bk_list, gm_list, nw_list, ns_list)
    ]
    tied = "tied" in args.align
    for gname in args.gemms:
        launcher = launch_gemm1 if gname == "gemm1" else launch_gemm2
        print(
            f"\n=== {gname} tile sweep: {len(cfgs)} configs, M={args.m}, "
            f"routing={args.routing}, align={args.align} "
            f"(N={args.n}, K={args.k}, E={args.e}, topk={args.topk}, "
            f"gs={args.group_size}) ==="
        )
        # time[cfg_idx][point_key] = us (mean over captured layers)
        times: dict[int, dict[tuple, float]] = defaultdict(dict)
        prod_cfg: dict[tuple, dict] = {}
        for M in args.m:
            for routing in args.routing:
                ids_list = routing_topk_ids(
                    routing, M, args.e, args.topk, captured, "cuda", torch.float16
                )
                if ids_list is None:
                    print(f"  (skip M={M} {routing}: not captured)")
                    continue
                # Alignments in play for this (M, routing); tied is per-cfg.
                aligns = sorted(
                    {
                        resolve_align(a, policy, M, args.topk, args.e)
                        for a in args.align
                        if a != "tied"
                    }
                )
                if tied:
                    aligns.append(0)  # 0 == tied-to-BM marker
                for align in aligns:
                    pkey = (M, routing, align)
                    problems_cache: dict[int, list[AlignedMoEProblem]] = {}

                    def get_problems(a, ids_list=ids_list, M=M, cache=problems_cache):
                        if a not in cache:
                            cache[a] = [
                                AlignedMoEProblem(W, M, ids, a, args.e)
                                for ids in ids_list
                            ]
                        return cache[a]

                    if align:
                        prod_cfg[pkey] = prod_tile(
                            policy,
                            gname,
                            M,
                            args.topk,
                            args.k,
                            args.n,
                            align,
                        )
                    for ci, cfg in enumerate(cfgs):
                        a = align or cfg["BLOCK_SIZE_M"]
                        if a % cfg["BLOCK_SIZE_M"]:
                            continue
                        try:
                            ts = []
                            for problem in get_problems(a):
                                fn = launcher(W, problem, cfg)
                                fn()
                                torch.accelerator.synchronize()
                                ts.append(time_us(fn))
                        except Exception as e:
                            if ci not in skipped:
                                skipped.add(ci)
                                print(f"  SKIP {cfg}: {type(e).__name__}: {e}")
                            continue
                        t = sum(ts) / len(ts)
                        times[ci][pkey] = t
                        rows.append(
                            dict(
                                mode="tiles",
                                gemm=gname,
                                M=M,
                                routing=routing,
                                align=a,
                                us=t,
                                npp=sum(p.npp_val for p in get_problems(a))
                                / len(problems_cache[a]),
                                routed=M * args.topk,
                                **cfg,
                            )
                        )
                    print(f"  done M={M} routing={routing} align={align or 'tied'}")
        _report_tiles(args, gname, cfgs, times, prod_cfg)


def _report_tiles(args, gname, cfgs, times, prod_cfg):
    best: dict[tuple, float] = {}
    for per in times.values():
        for pk, t in per.items():
            best[pk] = min(best.get(pk, t), t)
    n_points = len(best)
    ranked = []
    for ci, per in times.items():
        ratios = [t / best[pk] for pk, t in per.items()]
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        ranked.append((gm, max(ratios), len(ratios), ci))
    # Full-coverage configs first, then by geomean.
    ranked.sort(key=lambda r: (-(r[2] == n_points), r[0]))
    prod_keys = {
        tuple(c[k] for k in TILE_KEYS) for c in prod_cfg.values() if c is not None
    }
    print(f"\n--- {gname}: {len(ranked)} configs over {n_points} points ---")
    print(
        f"  {'rank':>4}   {'BM':>4} {'BN':>4} {'BK':>4} {'GM':>3} {'nw':>3} {'ns':>3}"
        f"  {'geomean':>8} {'worst':>7} {'npts':>5}"
    )
    shown = 0
    for rank, (gm, worst, n, ci) in enumerate(ranked, 1):
        cfg = cfgs[ci]
        key = tuple(
            min(cfg[k], args.group_size) if k == "BLOCK_SIZE_K" else cfg[k]
            for k in TILE_KEYS
        )
        is_prod = key in prod_keys
        if shown < args.top or (args.mark_prod and is_prod):
            shown += 1
            print(
                f"  {rank:>4}{'P' if is_prod else ' '}  {cfg['BLOCK_SIZE_M']:>4} "
                f"{cfg['BLOCK_SIZE_N']:>4} {cfg['BLOCK_SIZE_K']:>4} "
                f"{cfg['GROUP_SIZE_M']:>3} {cfg['num_warps']:>3} "
                f"{cfg['num_stages']:>3}  {gm:>8.3f} {worst:>7.2f} {n:>5}"
            )
    if args.mark_prod:
        print("  P = production config from _triton_config at one of the points")


# --------------------------------------------------------------------------
# Align mode
# --------------------------------------------------------------------------
def align_mode(args, W, policy, captured, rows):
    ov1 = parse_tile_override(args.gemm1_tile)
    ov2 = parse_tile_override(args.gemm2_tile)
    aligns = [int(a) for a in args.align]
    print(
        f"\n=== alignment sweep, joint gemm1+gemm2 us (N={args.n}, K={args.k}, "
        f"E={args.e}, topk={args.topk}, gs={args.group_size}) ==="
    )
    for routing in args.routing:
        table = []
        for M in args.m:
            ids_list = routing_topk_ids(
                routing, M, args.e, args.topk, captured, "cuda", torch.float16
            )
            if ids_list is None:
                print(f"  (skip M={M} {routing}: not captured)")
                continue
            res = {}
            for a in aligns:
                cfg1 = prod_tile(policy, "gemm1", M, args.topk, args.k, args.n, a)
                cfg2 = prod_tile(policy, "gemm2", M, args.topk, args.k, args.n, a)
                # Alignments below production's BLOCK_SIZE_M (e.g. A=32 with
                # BM=64 tiles) clamp BM to the alignment; explicit
                # --gemm*-tile overrides are applied afterwards and win.
                for c, ov in ((cfg1, ov1), (cfg2, ov2)):
                    if "BLOCK_SIZE_M" not in ov:
                        c["BLOCK_SIZE_M"] = min(c["BLOCK_SIZE_M"], a)
                    c.update(ov)
                if a % cfg1["BLOCK_SIZE_M"] or a % cfg2["BLOCK_SIZE_M"]:
                    continue
                ts, npps, touched = [], [], []
                for ids in ids_list:
                    problem = AlignedMoEProblem(W, M, ids, a, args.e)
                    f1, f2 = (
                        launch_gemm1(W, problem, cfg1),
                        launch_gemm2(W, problem, cfg2),
                    )

                    def joint(f1=f1, f2=f2):
                        f1()
                        f2()

                    joint()
                    torch.accelerator.synchronize()
                    ts.append(time_us(joint))
                    npps.append(problem.npp_val)
                    touched.append(problem.touched)
                us = sum(ts) / len(ts)
                npp = sum(npps) / len(npps)
                res[a] = (us, npp, sum(touched) / len(touched))
                rows.append(
                    dict(
                        mode="align",
                        gemm="joint",
                        M=M,
                        routing=routing,
                        align=a,
                        us=us,
                        npp=npp,
                        routed=M * args.topk,
                        experts=res[a][2],
                    )
                )
            table.append((M, res))
        print(f"\n--- routing: {routing} ---")
        hdr = f"  {'tokens':>6} {'routed':>7} {'experts':>7}"
        for a in aligns:
            hdr += f" {'A=' + str(a):>9} {'npp@' + str(a):>8}"
        print(hdr + "   best")
        for M, res in table:
            line = f"  {M:>6} {M * args.topk:>7} "
            exp = next(iter(res.values()))[2] if res else float("nan")
            line += f"{exp:>7.0f}"
            for a in aligns:
                if a in res:
                    line += f" {res[a][0]:>9.1f} {res[a][1]:>8.0f}"
                else:
                    line += f" {'-':>9} {'-':>8}"
            if res:
                ba = min(res, key=lambda k: res[k][0])
                line += f"   A={ba}"
            print(line)


# --------------------------------------------------------------------------
def sweep(args):
    policy = make_policy(args.group_size)
    captured = None
    for r in args.routing:
        if r.startswith("captured:"):
            captured = load_captured(r.split(":", 1)[1], args.e, args.topk)
            if len({x for x in args.routing if x.startswith("captured:")}) > 1:
                raise SystemExit("only one captured:<path> routing is supported")
    W = build_weights(args)
    torch.accelerator.synchronize()
    rows: list[dict] = []
    if args.mode == "tiles":
        tiles_mode(args, W, policy, captured, rows)
    else:
        align_mode(args, W, policy, captured, rows)
    if args.csv:
        fields = [
            "mode", "gemm", "M", "routing", "align", "routed", "npp", "experts",
            *TILE_KEYS, "us",
        ]  # fmt: skip
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["tiles", "align"], default="tiles")
    p.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[2048],
        help="Token counts (prefill batch rows)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=768,
        help="moe_intermediate_size (single, not 2x).  768 for Qwen3-30B-A3B, "
        "512 for Qwen3.5-A3B.",
    )
    p.add_argument("--k", type=int, default=2048, help="hidden_size")
    p.add_argument("--e", type=int, default=128, help="num_experts")
    p.add_argument("--topk", type=int, default=8, help="experts per token")
    p.add_argument("--group-size", type=int, default=32, help="W4A16 group size")
    p.add_argument(
        "--gemms", nargs="+", default=["gemm1", "gemm2"], choices=["gemm1", "gemm2"]
    )
    p.add_argument(
        "--align",
        nargs="+",
        default=None,
        help="moe_align alignments: ints, 'prod' or 'tied'.  Default: 'tied' "
        "(tiles mode, legacy) / 32 64 128 (align mode).",
    )
    p.add_argument(
        "--routing",
        nargs="+",
        default=["random"],
        help="random | uniform | captured:<path>",
    )
    p.add_argument("--block-m", nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--block-n", nargs="+", default=[16, 32, 64, 128])
    p.add_argument(
        "--block-k",
        nargs="+",
        default=[32, 64, 128],
        help="Capped to group_size inside the wrapper (duplicates are merged)",
    )
    p.add_argument("--group-m", nargs="+", default=[1, 4, 8])
    p.add_argument("--num-warps", nargs="+", default=[2, 4, 8])
    p.add_argument("--num-stages", nargs="+", default=[1, 2, 4])
    p.add_argument("--top", type=int, default=20, help="tiles mode: rows to print")
    p.add_argument(
        "--no-mark-prod",
        dest="mark_prod",
        action="store_false",
        help="tiles mode: do not mark/force-print production configs",
    )
    p.add_argument(
        "--gemm1-tile",
        default=None,
        help="align mode: override production gemm1 tile, e.g. BM=64,BN=32,nw=2,ns=4",
    )
    p.add_argument("--gemm2-tile", default=None, help="align mode: as --gemm1-tile")
    p.add_argument("--csv", type=str, default=None, help="Write all raw points")
    args = p.parse_args()
    if args.align is None:
        args.align = ["tied"] if args.mode == "tiles" else ["32", "64", "128"]
    if args.mode == "align" and any(a in ("prod", "tied") for a in args.align):
        p.error("--mode align needs integer --align values")
    sweep(args)


if __name__ == "__main__":
    main()
