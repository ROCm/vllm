#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune the launch knobs of the RDNA3.5 attention kernels, and apply them.

Decode, per (Hq, Hkv, D, window, M): every point of the knob grid is checked
against a float reference and timed over --contexts; the finalists are timed
against the current row in an interleaved A/B. A point may lose at most
loss_limit(S) at any context, and wins on the mean speedup weighted by
sqrt(S / 128) if above --min-gain. With --batch, the batched build, timed
over batch specs instead of contexts.

Prefill (--prefill), per (Hq, Hkv, D, window): one row per step of M, chosen
in an interleaved A/B against TRITON_ATTN.

Prefill batches (--prefill-batch): per cell, the build and head group.

`--apply` writes accepted decisions into csrc/rocm/rdna_*_configs.h,
refusing those measured on another kernel source or row.

    PYTHONPATH=$PWD python \\
        benchmarks/kernels/rocm_rdna_hip_attn/tools/tune.py \\
        --hq 32 --hkv 8 --head-dim 128 --m 1 \\
        --points points.jsonl --decisions decisions.jsonl
"""

import argparse
import functools
import gc
import itertools
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
import types
from pathlib import Path

import regex as re

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "attention_benchmarks"))
sys.path.insert(0, str(_HERE.parent))

import utils  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

CONTEXTS = [128, 512, 1024, 4096, 8192, 16384, 32768]
LONG_S, LONG_TARGET = 16384, 0.90
KNOBS = utils.KNOBS
# Relative error bound per dtype.
RTOL = {"fp16": 1e-3, "bf16": 8e-3}

# Search space; Hkv * row_groups * max_segments is capped at MAX_WG.
GRID = {
    "waves": (2, 4, 8),
    "max_segments": (1, 2, 3, 4, 5, 8),
    "min_segment_blocks": (1, 2, 4),
    "pipe": ("none", "prefetch", "v_in_lds"),
}
DOT_GRID = {
    "waves": (2, 4, 8),
    "max_segments": (1, 2, 3, 4, 5, 8),
}
MAX_WG = 64
# A point stuck for HANG_S is recorded as a hang and the process exits with
# HANG_RC; rerun with --reuse.
HANG_S = 240
HANG_RC = 75
# Collect garbage every GC_EVERY points.
GC_EVERY = 25


def dspl_rule(d):
    """The kernel's default HEAD_DIM_SPLIT."""
    return d // 128 if d >= 256 else 1


def grid(hq, hkv, d, m, batch=False):
    """Valid, distinct knob sets for one configuration; no dot product for a
    batch."""
    gqa = hq // hkv
    rule = dspl_rule(d)
    out = []
    # Row groups split the gqa * m rows evenly.
    rows = gqa * m
    for row_groups in (r for r in range(1, rows + 1) if rows % r == 0):
        for waves, max_segments, pipe, min_segment_blocks in itertools.product(
            GRID["waves"],
            GRID["max_segments"],
            GRID["pipe"],
            GRID["min_segment_blocks"],
        ):
            if hkv * row_groups * max_segments > MAX_WG:
                continue
            if max_segments == 1 and min_segment_blocks != 1:
                continue
            for head_dim_split in sorted({rule, 2 * rule}):
                lane_dims = (d // head_dim_split) // 16
                if lane_dims not in (2, 4, 6, 8) or waves % head_dim_split:
                    continue
                out.append(
                    {
                        "waves": waves,
                        "row_groups": row_groups,
                        "max_segments": max_segments,
                        "min_segment_blocks": min_segment_blocks,
                        "prefetch": int(pipe == "prefetch"),
                        "v_in_lds": int(pipe == "v_in_lds"),
                        # Write the kernel's default head_dim_split as 0.
                        "head_dim_split": 0
                        if head_dim_split == rule
                        else head_dim_split,
                    }
                )
    # The dot path needs whole dot2 pairs per lane (not D=96).
    if not batch and (8 if d == 128 else max(2, d // 32)) % 2 == 0:
        for waves, max_segments in itertools.product(*DOT_GRID.values()):
            out.append({"dot_product": 1, "waves": waves, "max_segments": max_segments})
    return out


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def rank(contexts, pct):
    # Cells on Triton are left out.
    served = [(s, c) for s, c in zip(contexts, pct) if c > 0]
    long_ = [c for s, c in served if s >= LONG_S]
    return (
        min(min(long_), LONG_TARGET) if long_ else LONG_TARGET,
        geomean([c for _, c in served]) if served else 0.0,
    )


def cell_ratio(ref_us, us):
    """Speedup over the reference at one cell; inf (Triton) is neutral for
    both, a loss for the point alone."""
    if math.isinf(ref_us):
        return 1.0
    return ref_us / us


def loss_limit(s):
    return min(0.05, max(0.01, 0.05 - 0.005 * (math.log2(s) - 7)))


def weighted(contexts, ratio):
    w = [math.sqrt(s / 128) for s in contexts]
    return sum(a * b for a, b in zip(w, ratio)) / sum(w)


def within_limits(contexts, ratio):
    return all(r >= 1 - loss_limit(s) for s, r in zip(contexts, ratio))


# Subspaces whose best point is also a finalist.
PINS = {
    "no_dot": lambda k, r: not k["dot_product"],
    "only_dot": lambda k, r: bool(k["dot_product"]),
    "no_prefetch": lambda k, r: not k["prefetch"],
    "no_v_in_lds": lambda k, r: not k["v_in_lds"],
    "no_head_dim_split": lambda k, r: k["head_dim_split"] in (0, r),
    "min_segment_blocks_1": lambda k, r: (
        k["min_segment_blocks"] == 1 or k["dot_product"]
    ),
    "max_segments_pow2": lambda k, r: k["max_segments"] & (k["max_segments"] - 1) == 0,
    "no_waves_2": lambda k, r: k["waves"] != 2,
    "simple": lambda k, r: (
        not (k["dot_product"] or k["prefetch"] or k["v_in_lds"])
        and k["head_dim_split"] in (0, r)
        and k["min_segment_blocks"] == 1
    ),
}


def canonical(k, d):
    """The knobs, normalized so equivalent builds compare equal."""
    if k["dot_product"]:
        return {n: k[n] for n in ("dot_product", "waves", "max_segments")}
    rule = dspl_rule(d)
    return {
        **k,
        "min_segment_blocks": 1 if k["max_segments"] == 1 else k["min_segment_blocks"],
        "head_dim_split": 0 if k["head_dim_split"] == rule else k["head_dim_split"],
    }


def knobs_of(v):
    return {k: getattr(v, k) for k in KNOBS}


def label(k):
    if k["dot_product"]:
        return f"dot waves={k['waves']} max_segments={k['max_segments']}"
    extra = [f"{n}={k[n]}" for n in ("head_dim_split", "prefetch", "v_in_lds") if k[n]]
    head = " ".join(
        f"{n}={k[n]}"
        for n in ("waves", "row_groups", "max_segments", "min_segment_blocks")
    )
    return " ".join([head, *extra])


# ----------------------------------------------------------------- reference --


def reference(q, kv, s, window=0):
    import torch

    m, hq, hkv, hd = q.shape[0], q.shape[1], kv.shape[1], q.shape[2]
    flat = kv.transpose(1, 2).reshape(-1, hkv, 2 * hd)[:s]
    k, v = flat[..., :hd], flat[..., hd:]
    g = hq // hkv
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2).repeat_interleave(g, 0)
    vf = v.float().permute(1, 0, 2).repeat_interleave(g, 0)
    sc = torch.bmm(qf, kf.transpose(1, 2)) * (hd**-0.5)
    pos = torch.arange(s, device=q.device).view(1, s)
    lim = (s - m + torch.arange(m, device=q.device)).view(m, 1)
    masked = pos > lim
    if window:
        masked |= pos < lim - (window - 1)
    sc = sc.masked_fill(masked.view(1, m, s), float("-inf"))
    return torch.bmm(torch.softmax(sc, -1), vf).permute(1, 0, 2)


def check_batch(module, v, repeat=2):
    """Worst relative error of a batched build over a long, a short and a
    padded sequence; the padded rows must stay unwritten."""
    import torch

    from vllm.v1.attention.ops.rocm_rdna_hip import make_scratch

    base = max(1040, v.window + 64, v.page_size + 1000)
    lens = [base, 48, 0]
    torch.manual_seed(0)
    width = -(-base // v.page_size)
    kv = (
        torch.randn(
            len(lens) * width,
            v.num_kv_heads,
            v.page_size,
            2 * v.head_size,
            device="cuda",
            dtype=v.dtype,
        )
        * 0.5
    )
    table = torch.arange(len(lens) * width, device="cuda", dtype=torch.int32)
    table = table.view(len(lens), width)
    q = (
        torch.randn(
            len(lens) * v.max_query_len,
            v.num_q_heads,
            v.head_size,
            device="cuda",
            dtype=v.dtype,
        )
        * 0.5
    )
    sl = torch.tensor(lens, device="cuda", dtype=torch.int32)
    scratch = make_scratch(v, torch.device("cuda"), max_seqs=len(lens))
    worst = 0.0
    for _ in range(repeat):
        out = torch.full_like(q, 7.0)
        module.decode_attn(q, kv, table, out, *scratch, sl, v.head_size**-0.5)
        torch.accelerator.synchronize()
        for i, s in enumerate(lens):
            rows = out[i * v.max_query_len : (i + 1) * v.max_query_len].float()
            if s == 0:
                if not (rows == 7.0).all():
                    return float("inf")
                continue
            if not torch.isfinite(rows).all():
                return float("inf")
            ref = reference(
                q[i * v.max_query_len : (i + 1) * v.max_query_len],
                kv[table[i].long()],
                s,
                v.window,
            )
            err = ((rows - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
            worst = max(worst, err)
    return worst


def check(module, v, repeat=2):
    """Worst relative error over several contexts, each on new inputs and a
    NaN-filled output, launched `repeat` times on one scratch.
    """
    import torch

    from vllm.v1.attention.ops.rocm_rdna_hip import make_scratch

    worst = 0.0
    scratch = make_scratch(v, torch.device("cuda"))
    base = max(1040, v.window + 64, v.page_size + 1000)
    for seed, s in enumerate((48, base, base + 1000, 300)):
        torch.manual_seed(seed)
        nb = -(-s // v.page_size)
        kv = (
            torch.randn(
                nb,
                v.num_kv_heads,
                v.page_size,
                2 * v.head_size,
                device="cuda",
                dtype=v.dtype,
            )
            * 0.5
        )
        q = (
            torch.randn(
                v.max_query_len,
                v.num_q_heads,
                v.head_size,
                device="cuda",
                dtype=v.dtype,
            )
            * 0.5
        )
        table = torch.arange(nb, device="cuda", dtype=torch.int32)
        ref = reference(q, kv, s, v.window)
        sl = torch.tensor([s], device="cuda", dtype=torch.int32)
        out = torch.empty_like(q)
        for _ in range(repeat):
            out.fill_(float("nan"))
            module.decode_attn(q, kv, table, out, *scratch, sl, v.head_size**-0.5)
            torch.accelerator.synchronize()
            if not torch.isfinite(out).all():
                return float("inf")
            err = ((out.float() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
            worst = max(worst, err)
    return worst


# -------------------------------------------------------------------- tuning --


def tune(a) -> None:
    import common
    import jit
    import torch
    from runner import run_attention_benchmark

    import vllm
    import vllm.v1.attention.backends.rocm_rdna_attn as backend
    from vllm.v1.attention.ops.rocm_rdna_hip import KernelVariant, variant_for

    jit.install()
    common.gc = types.SimpleNamespace(collect=lambda *x: 0)
    dtype = utils.torch_dtype(a.dtype)
    hq, hkv, d, m, win, bs = a.hq, a.hkv, a.head_dim, a.m, a.window, a.block_size
    batch = list(a.batch or [])
    # Contexts, or batch specs weighted as their longest sequence.
    cells = batch or list(a.contexts)
    if batch:
        from batch_spec import parse_batch_spec

        for spec in batch:
            if any(r.q_len != m for r in parse_batch_spec(spec)):
                raise SystemExit(f"{spec}: every sequence of a batch takes M={m}")
        ctxs = [max(r.kv_len for r in parse_batch_spec(spec)) for spec in batch]
    else:
        ctxs = cells
    tag = (
        f"{hq}/{hkv}/{d} M={m}"
        + (f" w{win}" if win else "")
        + (" batch" if batch else "")
        + f" {a.dtype}"
    )
    fixed = {
        "hq": hq,
        "hkv": hkv,
        "d": d,
        "m": m,
        "window": win,
        "dtype": a.dtype,
        "block_size": bs,
        "source": jit.source_digest(),
        "contexts": cells,
        "batch": bool(batch),
    }
    meta = {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "host": platform.node(),
    }
    if batch:
        from matrix import spec_roofline

        roof = [
            spec_roofline(spec, hq, hkv, d, block_size=bs, window=win)[0]
            for spec in batch
        ]
    else:
        roof = [
            utils.roofline_us(hq, hkv, d, m, s, block_size=bs, window=win) for s in ctxs
        ]

    def variant(knobs):
        return KernelVariant(
            d,
            hq,
            hkv,
            m,
            bs,
            1,
            **knobs,
            dtype=dtype,
            window=win,
            batched=int(bool(batch)),
        )

    def decode(v, cell):
        """(us, spread %) of m tokens at context `cell`, or of batch spec `cell`,
        on v; (inf, 0) if it runs on Triton."""
        from common import BenchmarkConfig

        cfg = BenchmarkConfig(
            backend="ROCM_RDNA_ATTN",
            batch_spec=cell if batch else f"q{m}s{cell}",
            num_layers=10,
            min_working_set_mb=96,
            head_dim=d,
            num_q_heads=hq,
            num_kv_heads=hkv,
            block_size=bs,
            device="cuda:0",
            dtype=dtype,
            sliding_window=win or None,
        )
        with jit.paths() as path, jit.override(knobs_of(v)):
            r = run_attention_benchmark(cfg)
        progress["t"] = time.time()
        if batch and path() == "triton":
            return math.inf, 0.0
        if path() != "kernel":
            raise RuntimeError(f"ran {path()} at {cell}")
        return r.median_time * 1e6, r.std_time / r.median_time * 100

    def pid(knobs):
        return json.dumps(sorted(knobs.items()))

    known: dict[str, dict] = {}
    if a.reuse and Path(a.points).exists():
        for line in Path(a.points).read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # truncated line
                continue
            if all(rec.get(k) == v for k, v in fixed.items()):
                known[pid(rec["knobs"])] = rec

    current = variant_for(hq, hkv, d, m, bs, 1, win, dtype, bool(batch))
    raw = (
        [json.loads(x) for x in Path(a.candidates).read_text().splitlines() if x]
        if a.candidates
        else grid(hq, hkv, d, m, bool(batch))
    )
    cands: dict = {} if current is None else {current: knobs_of(current)}
    space = set()
    for k in raw:
        try:
            v = variant(k)
        except (TypeError, ValueError):
            continue
        cands.setdefault(v, knobs_of(v))
        space.add(pid(canonical(knobs_of(v), d)))
    outside = current is not None and pid(canonical(knobs_of(current), d)) not in space
    items = list(cands.items())[: a.limit or None]
    todo = [v for v, k in items if pid(k) not in known]
    t0 = time.time()
    # Reused points too: any may be a finalist.
    jit.precompile([v for v, _ in items])
    jit.seal()
    print(
        f"# {tag}: {len(items)} points, {len(items) - len(todo)} reused, "
        f"built in {time.time() - t0:.0f} s",
        flush=True,
    )
    time.sleep(5)  # let clocks settle after the build

    # Watchdog for hung points.
    progress: dict = {"t": time.time(), "rec": None}

    def watchdog():
        while True:
            time.sleep(5)
            rec = progress["rec"]
            if rec is not None and time.time() - progress["t"] > HANG_S:
                rec = {
                    **rec,
                    "status": "hang",
                    "error": f"no progress for {HANG_S} s (GPU hang)",
                }
                with open(a.points, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                print(f"# HANG: {rec['knobs']}", flush=True)
                os._exit(HANG_RC)

    threading.Thread(target=watchdog, daemon=True).start()
    tol = RTOL[a.dtype]
    scored = []
    with open(a.points, "a") as fh:
        for i, (v, k) in enumerate(items):
            rec = known.get(pid(k))
            if rec is None:
                rec = {
                    **fixed,
                    "knobs": k,
                    "current": v == current,
                    **meta,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                progress["t"], progress["rec"] = time.time(), dict(rec)
                try:
                    mod = jit.load(v)
                    if mod is None:
                        raise RuntimeError(jit.failure(v))
                    rec["max_rel"] = err = (check_batch if batch else check)(mod, v)
                    if err > tol:
                        rec["status"] = "wrong"
                    else:
                        us, spread = zip(*(decode(v, c) for c in cells))
                        if all(math.isinf(u) for u in us):
                            raise RuntimeError("ran triton at every batch")
                        rec.update(
                            status="ok",
                            us=[round(u, 3) for u in us],
                            spread_pct=[round(x, 2) for x in spread],
                            pct_roof=[round(r / u, 4) for r, u in zip(roof, us)],
                        )
                except Exception as e:  # noqa: BLE001 - recorded, the search goes on
                    rec["status"] = "build" if "does not build" in str(e) else "error"
                    rec["error"] = f"{type(e).__name__}: {e}"[:400]
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                known[pid(k)] = rec
                progress["rec"] = None
                # Free the batched scratch the backend keeps.
                backend._SCRATCH.clear()
            if (i + 1) % GC_EVERY == 0 or rec.get("status") == "error":
                gc.collect()
            if rec.get("status") == "ok":
                scored.append((rank(ctxs, rec["pct_roof"]), v, k))
            if (i + 1) % 100 == 0:
                print(
                    f"#   {i + 1}/{len(items)} points, "
                    f"{(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )

    # Rank against the current row, or by %roof without one.
    cur = None if current is None else known.get(pid(knobs_of(current)))
    ref_us = cur["us"] if cur is not None and cur.get("status") == "ok" else None
    if ref_us is not None:
        scored = [
            ((within_limits(ctxs, r), weighted(ctxs, r)), v, k)
            for _, v, k in scored
            for r in [[cell_ratio(b, u) for b, u in zip(ref_us, known[pid(k)]["us"])]]
        ]
    scored.sort(key=lambda x: x[0], reverse=True)
    rule = dspl_rule(d)
    finalists: dict = {}
    for _, v, k in scored[: a.top]:
        finalists.setdefault(v, (k, ["top"]))
    for name, pred in PINS.items():
        best_in = next((x for x in scored if pred(x[2], rule)), None)
        if best_in is not None:
            finalists.setdefault(best_in[1], (best_in[2], []))[1].append(name)
    if ref_us is not None:
        # Also the best points that lose at no context.
        safe = [
            x
            for x in scored
            if x[1] != current
            and all(b >= u for b, u in zip(ref_us, known[pid(x[2])]["us"]))
        ]
        for _, v, k in safe[:3]:
            finalists.setdefault(v, (k, []))[1].append("safe")
    finalists.pop(current, None)

    # Interleaved A/B over all contexts.
    runs = [(v, k, tags) for v, (k, tags) in finalists.items()]
    if current is not None:
        runs.insert(0, (current, knobs_of(current), ["current"]))
    times = [{c: [] for c in cells} for _ in runs]
    for _ in range(a.confirm_rounds):
        for c in cells:
            for (v, _, _), t in zip(runs, times):
                try:
                    t[c].append(decode(v, c)[0])
                except Exception:  # noqa: BLE001
                    t[c].append(float("inf"))
    med = [[statistics.median(t[c]) for c in cells] for t in times]
    base = med[0] if current is not None else None
    results = []
    for (v, k, tags), us in zip(runs, med):
        if "current" in tags:
            continue
        # Against the row, or roof without one.
        ratio = (
            [cell_ratio(b, u) for b, u in zip(base, us)]
            if base
            else [r / u for r, u in zip(roof, us)]
        )
        score = weighted(ctxs, ratio)
        accepted = base is None or (score > a.min_gain and within_limits(ctxs, ratio))
        results.append(
            {
                "knobs": k,
                "for": tags,
                "us": us,
                "ratio": ratio,
                "score": score,
                "worst": min(ratio),
                "accepted": accepted,
            }
        )
    ok = [r for r in results if r["accepted"]]
    best = max(ok, key=lambda r: r["score"]) if ok else None
    replaced = best is None and outside and bool(results)
    if replaced:
        best = max(results, key=lambda r: (within_limits(ctxs, r["ratio"]), r["score"]))
    winner = None if current is None else knobs_of(current)
    if best:
        winner = best["knobs"]
    decision = {
        **fixed,
        "current": None if current is None else knobs_of(current),
        "current_us": base,
        "finalists": results,
        "accepted": best is not None,
        # The row is outside the search space and nothing beat it.
        "replaced": replaced,
        "winner": winner,
        "score": best["score"] if best else 1.0,
        "points": len(items),
        "ok_points": len(scored),
        **meta,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(a.decisions, "a") as out:
        out.write(json.dumps(decision) + "\n")
    if best:
        print(
            f"{tag}: {'REPLACE (row outside the grid)' if replaced else 'ACCEPT'} "
            f"{label(best['knobs'])}  score {best['score']:.3f}, "
            f"worst cell {best['worst']:.3f}",
            flush=True,
        )
    elif current is not None:
        print(f"{tag}: keep current {label(knobs_of(current))}", flush=True)
    else:
        print(f"{tag}: no point measured correct", flush=True)
    for r in results:
        print(
            f"   {','.join(r['for']):<24} score {r['score']:.3f} worst "
            f"{r['worst']:.3f}  {label(r['knobs'])}",
            flush=True,
        )


# ------------------------------------------------------------------- prefill --

# Steps of M; a row serves M from its step up to the next.
STEPS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
PREFIXES = (0, 1024, 16384)
# A step keeps the previous step's row when within this factor of the winner.
PREFILL_SLACK = 1.02
PREFILL_BASELINES = ("TRITON_ATTN",)


def prefill_grid(d):
    """Prefill knob sets for head size d."""
    split = max(1, d // 128)
    if d <= 128:
        axes = {
            "waves": (4, 8),
            "key_tile": (16, 32, 64),
            "v_t": (0, 1),
            "key_waves": (1, 2),
            "double_buffer": (0,),
        }
    elif d == 256:
        axes = {
            "waves": (4, 8, 16),
            "key_tile": (16, 32),
            "v_t": (1,),
            "key_waves": (1, 2),
            "double_buffer": (0, 1),
        }
    else:
        axes = {
            "waves": (8, 16),
            "key_tile": (16,),
            "v_t": (1,),
            "key_waves": (1, 2),
            "double_buffer": (0, 1),
        }
    axes |= {"max_segments": (1, 4, 16), "head_group": (1, 2, 4, 8)}
    out = []
    for values in itertools.product(*axes.values()):
        k = dict(zip(axes, values), head_dim_split=split)
        if (
            k["waves"] % (split * k["key_waves"])
            or k["key_tile"] // 16 % k["key_waves"]
        ):
            continue
        out.append(k)
    return out


def prefill_knobs_of(v):
    return {k: getattr(v, k) for k in utils.PREFILL_KNOBS}


def prefill_label(k):
    return " ".join(f"{n}={k[n]}" for n in utils.PREFILL_KNOBS)


def weighted_speedup(weights, ratio):
    """Weighted geometric mean of the speedups (other / ours)."""
    return math.exp(sum(w * math.log(r) for w, r in zip(weights, ratio)) / sum(weights))


def tune_prefill(a) -> None:
    import common
    import jit
    import torch
    from runner import run_attention_benchmark
    from triton.testing import do_bench

    import vllm
    import vllm.v1.attention.ops.rocm_rdna_hip as R
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    jit.install()
    common.gc = types.SimpleNamespace(collect=lambda *x: 0)
    dtype = utils.torch_dtype(a.dtype)
    hq, hkv, d, win, bs = a.hq, a.hkv, a.head_dim, a.window, a.block_size
    tag = f"{hq}/{hkv}/{d}" + (f" w{win}" if win else "") + f" prefill {a.dtype}"
    cells = [(m, p) for m in STEPS for p in a.prefixes]
    flops = [utils.attention_flops(hq, d, m, m + p, win) for m, p in cells]
    total = sum(flops)
    weight = {c: math.sqrt(f / total) for c, f in zip(cells, flops)}
    fixed = {
        "kind": "prefill",
        "hq": hq,
        "hkv": hkv,
        "d": d,
        "window": win,
        "dtype": a.dtype,
        "block_size": bs,
        "source": jit.source_digest(prefill=True),
        "cells": cells,
    }
    meta = {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "host": platform.node(),
    }

    def variant(knobs):
        return R.PrefillVariant(d, hq, hkv, win, bs, STEPS[0], **knobs, dtype=dtype)

    def pid(knobs):
        return json.dumps(sorted(knobs.items()))

    current = {
        m: R.prefill_variant_for(hq, hkv, d, bs, 1, win, dtype, m) for m in STEPS
    }
    raw = (
        [json.loads(x) for x in Path(a.candidates).read_text().splitlines() if x]
        if a.candidates
        else prefill_grid(d)
    )
    cands: dict[str, R.PrefillVariant] = {}
    for k in [*raw, *(prefill_knobs_of(v) for v in current.values() if v)]:
        try:
            cands.setdefault(pid(k), variant(k))
        except (TypeError, ValueError):
            continue
    items = list(cands.items())[: a.limit or None]
    known: dict[str, dict] = {}
    if a.reuse and Path(a.points).exists():
        for line in Path(a.points).read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(rec.get(k) == v for k, v in fixed.items()):
                known[pid(rec["knobs"])] = rec
    t0 = time.time()
    jit.precompile([v for p, v in items if p not in known])
    jit.seal()
    print(
        f"# {tag}: {len(items)} points, built in {time.time() - t0:.0f} s", flush=True
    )
    time.sleep(5)  # let clocks settle after the build

    dev = torch.device("cuda")

    def inputs(m, p, seed):
        torch.manual_seed(seed)
        s = m + p
        pages = -(-s // bs)
        kv = (torch.randn(pages, hkv, bs, 2 * d, device=dev) * 0.5).to(dtype)
        table = torch.randperm(pages, device=dev).to(torch.int32)
        q = (torch.randn(m, hq, d, device=dev) * 0.5).to(dtype)
        return q, kv, table, torch.tensor([s], device=dev, dtype=torch.int32)

    def triton(q, kv, table, sl):
        m, s = q.shape[0], int(sl.item())
        k, v = kv.transpose(1, 2).split(d, dim=-1)
        out = torch.empty_like(q)
        one = torch.ones(1, hkv, device=dev)
        starts = torch.tensor([0, m], device=dev, dtype=torch.int32)
        unified_attention(
            q,
            k,
            v,
            out,
            starts,
            m,
            sl,
            s,
            d**-0.5,
            True,
            (win - 1, 0) if win else (-1, -1),
            table.view(1, -1),
            0,
            None,
            one,
            one,
        )
        return out

    scratch = {}

    def op(module, v, q, kv, table, sl, out):
        key = v.scratch_rows(dev)
        if key not in scratch:
            scratch[key] = R.make_prefill_scratch([v], dev)
        s = int(sl.item())
        return lambda: module.prefill_attn(
            q, kv, table, out, *scratch[key], sl, s, d**-0.5
        )

    # 1. Check and time every point on the op.
    with open(a.points, "a") as fh:
        for i, (key, v) in enumerate(items):
            if key in known:
                continue
            rec = {
                **fixed,
                "knobs": prefill_knobs_of(v),
                **meta,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            module = jit.load(v)
            if module is None:
                rec.update(status="build", error=(jit.failure(v) or "")[:400])
            else:
                us = []
                for seed, (m, p) in enumerate(cells):
                    q, kv, table, sl = inputs(m, p, seed)
                    out = torch.full_like(q, float("nan"))
                    run = op(module, v, q, kv, table, sl, out)
                    run()
                    torch.accelerator.synchronize()
                    err = (out.float() - triton(q, kv, table, sl).float()).abs().max()
                    if not err.item() <= 5e-3:
                        rec.update(status="wrong", cell=[m, p], err=err.item())
                        break
                    us.append(
                        do_bench(run, warmup=5, rep=40, return_mode="median") * 1e3
                    )
                else:
                    rec.update(status="ok", us=[round(u, 2) for u in us])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            known[key] = rec
            if (i + 1) % 20 == 0:
                print(
                    f"#   {i + 1}/{len(items)} points, "
                    f"{(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    ok = {
        k: dict(zip(map(tuple, cells), r["us"]))
        for k, r in known.items()
        if r.get("status") == "ok"
    }
    if not ok:
        print(f"{tag}: no point measured correct", flush=True)
        return

    # 2. Time the finalists of each step through the backend.
    def step_cells(m):
        return [c for c in cells if c[0] == m]

    finalists: dict[int, set[str]] = {}
    for m in STEPS:
        sc = step_cells(m)
        fastest = {c: min(t[c] for t in ok.values()) for c in sc}
        ranked = sorted(
            ok,
            key=lambda k: (
                -weighted_speedup(
                    [weight[c] for c in sc], [fastest[c] / ok[k][c] for c in sc]
                )
            ),
        )
        finalists[m] = set(ranked[: a.top])
        if current[m] is not None and pid(prefill_knobs_of(current[m])) in ok:
            finalists[m].add(pid(prefill_knobs_of(current[m])))
    for m, prev in zip(STEPS[1:], STEPS):
        finalists[m] |= finalists[prev]

    def harness(backend, m, p, knobs=None):
        from common import BenchmarkConfig

        cfg = BenchmarkConfig(
            backend=backend,
            batch_spec=f"q{m}s{m + p}",
            num_layers=10,
            min_working_set_mb=96,
            head_dim=d,
            num_q_heads=hq,
            num_kv_heads=hkv,
            block_size=bs,
            device="cuda:0",
            dtype=dtype,
            sliding_window=win or None,
            use_cuda_graphs=False,
            max_model_len=max(STEPS) * 2 + max(a.prefixes),
        )
        if knobs is None:
            return run_attention_benchmark(cfg).median_time * 1e6
        with jit.paths() as path, jit.override(knobs):
            us = run_attention_benchmark(cfg).median_time * 1e6
        if path() != "prefill":
            raise RuntimeError(f"ran {path()} at q{m}s{m + p}")
        return us

    times: dict = {}
    for _ in range(a.confirm_rounds):
        for m in STEPS:
            for c in step_cells(m):
                for b in PREFILL_BASELINES:
                    try:
                        times.setdefault((b, c), []).append(harness(b, *c))
                    except Exception:  # noqa: BLE001 - the backend refuses the shape
                        times.setdefault((b, c), []).append(float("inf"))
                for k in sorted(finalists[m]):
                    times.setdefault((k, c), []).append(
                        harness("ROCM_RDNA_ATTN", *c, dict(json.loads(k)))
                    )
    med = {key: statistics.median(t) for key, t in times.items()}

    # 3. Per step, prefer points never slower than a baseline, then speedup.
    steps, prev = [], None
    for m in STEPS:
        sc = step_cells(m)
        base = {c: min(med[(b, c)] for b in PREFILL_BASELINES) for c in sc}
        w = [weight[c] for c in sc]
        scored = {}
        for k in finalists[m]:
            ratio = [base[c] / med[(k, c)] for c in sc]
            scored[k] = (min(ratio) >= 1.0, weighted_speedup(w, ratio), ratio)
        best = max(scored, key=lambda k: scored[k][:2])
        if (
            prev in scored
            and (scored[prev][0] or not scored[best][0])
            and scored[prev][1] * PREFILL_SLACK >= scored[best][1]
        ):
            best = prev
        cur = None if current[m] is None else prefill_knobs_of(current[m])
        steps.append(
            {
                "m": m,
                "current": cur,
                "winner": dict(json.loads(best)),
                "speedup": scored[best][1],
                "worst": min(scored[best][2]),
                "cells": {
                    f"{c[0]}:{c[1]}": {"ours": med[(best, c)], "base": base[c]}
                    for c in sc
                },
            }
        )
        prev = best
        print(
            f"{tag} M>={m}: {prefill_label(steps[-1]['winner'])}  "
            f"speedup {steps[-1]['speedup']:.2f}x, "
            f"worst cell {steps[-1]['worst']:.2f}x",
            flush=True,
        )
    decision = {
        **fixed,
        "current": {m: s["current"] for m, s in zip(STEPS, steps)},
        "steps": steps,
        "accepted": True,
        **meta,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(a.decisions, "a") as out:
        out.write(json.dumps(decision) + "\n")


def apply_prefill(dec: dict, block_sizes: list[int]) -> int:
    """Write one prefill decision into the prefill configs."""
    import jit
    import torch

    import vllm.v1.attention.ops.rocm_rdna_hip as R

    hq, hkv, d, win = dec["hq"], dec["hkv"], dec["d"], dec["window"]
    tag = f"{hq}/{hkv}/{d}" + (f" w{win}" if win else "") + " prefill"
    if dec["source"] != jit.source_digest(prefill=True):
        print(f"{tag}: refused, measured on another kernel source")
        return 0
    lines, rows = read_configs(PREFILL_CONFIGS_H)
    col = {f: i for i, f in enumerate(R.PREFILL_CONFIG_FIELDS)}
    dtype = utils.torch_dtype(dec["dtype"])
    for step in dec["steps"]:
        now = R.prefill_variant_for(
            hq, hkv, d, dec["block_size"], 1, win, dtype, step["m"]
        )
        if step["current"] != (None if now is None else prefill_knobs_of(now)):
            print(f"{tag}: refused, the rows changed since they were measured")
            return 0

    def mine(r):
        return (
            r[col["head_size"]],
            r[col["num_q_heads"]],
            r[col["num_kv_heads"]],
            r[col["window"]],
        ) == (d, hq, hkv, win)

    sizes = sorted({r[col["page_size"]] for r in rows if mine(r)}) or block_sizes
    rows = [r for r in rows if not mine(r)]
    merged = []
    for step in dec["steps"]:
        if not merged or merged[-1]["winner"] != step["winner"]:
            merged.append(step)
    written = 0
    for bf16, bs, step in itertools.product((0, 1), sizes, merged):
        v = R.PrefillVariant(
            d,
            hq,
            hkv,
            win,
            bs,
            step["m"],
            **step["winner"],
            dtype=torch.bfloat16 if bf16 else torch.float16,
        )
        rows.append(R.prefill_config_fields(v))
        written += 1
    write_configs(PREFILL_CONFIGS_H, lines, rows, R.PREFILL_CONFIG_FIELDS)
    for step in merged:
        print(f"{tag} M>={step['m']}: {prefill_label(step['winner'])}")
    return written


# ------------------------------------------------------------- prefill batch --

# A cell's representative mean M and keys, by its lower bounds.
BATCH_M = {1: 4, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096}
BATCH_KEYS = {0: 256, 512: 1024, 2048: 4096, 8192: 16384}
# Tokens of a cell's larger batch (the smaller has a quarter), within
# BATCH_KV_BYTES of KV.
BATCH_TOKENS = 4096
BATCH_KV_BYTES = 1 << 30
# Launches rotate over KV copies this large, so caches do not flatter small
# batches.
BATCH_WORKING_SET = 96 << 20


def prefill_batch_cells(window, kv_bytes_per_key):
    """((min mean M, min keys), representative batches) of every cell."""
    import numpy as np

    import vllm.v1.attention.ops.rocm_rdna_hip as R

    out = []
    for i, m_lo in enumerate(R.PREFILL_BATCH_MEAN_M):
        m = BATCH_M[m_lo]
        for j, k_lo in enumerate(R.PREFILL_BATCH_KEYS):
            k_hi = (R.PREFILL_BATCH_KEYS + (float("inf"),))[j + 1]
            context = max(0, BATCH_KEYS[k_lo] - (m + 1) // 2)
            n = min(64, max(2, BATCH_TOKENS // m))
            n = max(2, min(n, BATCH_KV_BYTES // (kv_bytes_per_key * (m + context))))
            specs = []
            for nseq in dict.fromkeys((n, max(2, n // 4))):
                starts = np.arange(nseq + 1) * m
                lens = np.full(nseq, m + context)
                mean_m, keys = R.prefill_batch_features(starts, lens, window)
                if m_lo <= mean_m and k_lo <= keys < k_hi:
                    specs.append((starts, lens, f"{nseq}q{m}s{m + context}"))
            if specs:
                out.append(((m_lo, k_lo), specs))
    return out


def tune_prefill_batch(a) -> None:
    """Pick, per cell of a shape, the fastest prefill build and head group."""
    import jit
    import torch

    import vllm.v1.attention.ops.rocm_rdna_hip as R

    jit.install()
    dtype = utils.torch_dtype(a.dtype)
    hq, hkv, d, win, bs = a.hq, a.hkv, a.head_dim, a.window, a.block_size
    tag = f"{hq}/{hkv}/{d}" + (f" w{win}" if win else "") + f" batch {a.dtype}"
    dev = torch.device("cuda")
    rows = R.prefill_variants(hq, hkv, d, bs, win, dtype)
    if not rows:
        raise SystemExit(f"{tag}: no prefill rows")
    # Rows differing in max_segments or head_group build the same kernel.
    builds: dict = {}
    for v in rows:
        k = prefill_knobs_of(v)
        k.pop("max_segments"), k.pop("head_group")
        builds.setdefault(json.dumps(sorted(k.items())), v)
    groups = [g for g in range(1, hkv + 1) if hkv % g == 0]
    jit.precompile(list(builds.values()))
    jit.seal()
    time.sleep(5)  # a parallel build moves the SoC clock
    scratch = R.make_prefill_scratch(rows, dev)
    target = R.prefill_target_workgroups(dev)

    def timed(launches, iters=25):
        """Median us of one launch, `launches` taken in turn."""
        for fn in launches[:3]:
            fn()
        torch.accelerator.synchronize()
        ts = []
        for i in range(iters):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            launches[i % len(launches)]()
            end.record()
            end.synchronize()
            ts.append(start.elapsed_time(end) * 1000)
        return statistics.median(ts)

    cells = []
    kv_bytes_per_key = hkv * 2 * d * torch.finfo(dtype).bits // 8
    for (m_lo, k_lo), specs in prefill_batch_cells(win, kv_bytes_per_key):
        times: dict = {}
        for starts, lens, spec in specs:
            torch.manual_seed(0)
            nseq, pages = len(lens), -(-int(lens[0]) // bs)
            kv = (torch.randn(nseq * pages, hkv, bs, 2 * d, device=dev) * 0.5).to(dtype)
            kvs = [kv] + [
                kv.clone()
                for _ in range(BATCH_WORKING_SET // (kv.numel() * kv.element_size()))
            ]
            table = torch.randperm(nseq * pages, device=dev).to(torch.int32)
            table = table.view(nseq, pages)
            q = (torch.randn(int(starts[-1]), hq, d, device=dev) * 0.5).to(dtype)
            out = torch.empty_like(q)
            sl = torch.tensor(lens, dtype=torch.int32, device=dev)
            qsl = torch.tensor(starts, dtype=torch.int32, device=dev)
            for v in builds.values():
                mod = jit.load(v)
                for g in groups:
                    plan, hg = R.plan_prefill(starts, lens, v, win, target, g)
                    plan = torch.from_numpy(plan).to(dev)
                    t = timed(
                        [
                            functools.partial(
                                mod.prefill_attn,
                                q,
                                cache,
                                table,
                                out,
                                *scratch,
                                sl,
                                int(lens.max()),
                                d**-0.5,
                                plan=plan,
                                query_start_loc=qsl,
                                head_group=hg,
                            )
                            for cache in kvs
                        ]
                    )
                    times.setdefault((v.min_query_len, g), {})[spec] = t
        best = {spec: min(t[spec] for t in times.values()) for _, _, spec in specs}
        score = {
            c: geomean([t[spec] / best[spec] for spec in t]) for c, t in times.items()
        }
        (row, group), s_ = min(score.items(), key=lambda x: x[1])
        cells.append(
            {
                "min_mean_m": m_lo,
                "min_keys": k_lo,
                "row": row,
                "head_group": group,
                "score": s_,
                "specs": [spec for _, _, spec in specs],
                "us": {spec: times[(row, group)][spec] for spec in best},
            }
        )
        print(
            f"{tag}: mean M >= {m_lo:4}, keys >= {k_lo:4}: row {row:5} head group "
            f"{group}, {s_:.3f} of the best per batch",
            flush=True,
        )
    with open(a.decisions, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "prefill_batch",
                    "hq": hq,
                    "hkv": hkv,
                    "d": d,
                    "window": win,
                    "dtype": a.dtype,
                    "block_size": bs,
                    "source": jit.source_digest(prefill=True),
                    "rows": [
                        prefill_knobs_of(v) | {"min_query_len": v.min_query_len}
                        for v in rows
                    ],
                    "cells": cells,
                    "host": platform.node(),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            + "\n"
        )


def apply_prefill_batch(dec: dict, block_sizes: list[int]) -> int:
    """Write a shape's cells into the prefill batch configs; refused if its
    prefill builds changed since."""
    import torch

    import vllm.v1.attention.ops.rocm_rdna_hip as R

    hq, hkv, d, win = dec["hq"], dec["hkv"], dec["d"], dec["window"]
    tag = f"{hq}/{hkv}/{d}" + (f" w{win}" if win else "") + " batch"
    rows = R.prefill_variants(hq, hkv, d, dec["block_size"], win, torch.float16)
    now = [prefill_knobs_of(v) | {"min_query_len": v.min_query_len} for v in rows]
    if now != dec["rows"]:
        print(f"{tag}: refused, the prefill builds changed since it was measured")
        return 0
    lines, table = read_configs(PREFILL_BATCH_CONFIGS_H)
    pages = (
        sorted(
            {k[3] for k in R._prefill_table() if k[:3] == (hq, hkv, d) and k[4] == win}
        )
        or block_sizes
    )
    table = [r for r in table if (r[0], r[1], r[2], r[3]) != (d, hq, hkv, win)]
    for page in pages:
        for c in dec["cells"]:
            table.append(
                [
                    d,
                    hq,
                    hkv,
                    win,
                    page,
                    c["min_mean_m"],
                    c["min_keys"],
                    c["row"],
                    c["head_group"],
                ]
            )
    write_configs(PREFILL_BATCH_CONFIGS_H, lines, table, R.PREFILL_BATCH_CONFIG_FIELDS)
    print(f"{tag}: {len(dec['cells'])} cells at pages {pages}")
    return len(dec["cells"]) * len(pages)


# ------------------------------------------------------------------- --apply --


_CSRC = _HERE.parents[4] / "csrc" / "rocm"
DECODE_CONFIGS_H = _CSRC / "rdna_decode_configs.h"
DECODE_BATCH_CONFIGS_H = _CSRC / "rdna_decode_batch_configs.h"
PREFILL_CONFIGS_H = _CSRC / "rdna_prefill_configs.h"
PREFILL_BATCH_CONFIGS_H = _CSRC / "rdna_prefill_batch_configs.h"
# One config entry per line.
_ENTRY = re.compile(r"^\s*\{([\w:, ]+)\},$")


def read_configs(path: Path) -> tuple[list[str], list[list[int]]]:
    """The lines of a configs header, and its entries."""
    import jit

    lines = path.read_text().splitlines()
    rows = [jit.parse_cpp_fields(m[1]) for m in map(_ENTRY.match, lines) if m]
    return lines, rows


def write_configs(
    path: Path, lines: list[str], rows: list[list[int]], names: tuple[str, ...]
) -> None:
    """Replace the entries of a configs header with `rows`, sorted; `names`
    are their fields."""
    import jit

    at = [i for i, line in enumerate(lines) if _ENTRY.match(line)]
    entries = [f"    {{{jit.cpp_fields(r, names)}}}," for r in sorted(rows)]
    text = lines[: at[0]] + entries + lines[at[-1] + 1 :]
    path.write_text("\n".join(text) + "\n")


def apply(path: str, block_sizes: list[int]) -> None:
    """Write the accepted decisions of `path` into the configs headers."""
    import jit
    import torch

    import vllm.v1.attention.ops.rocm_rdna_hip as R

    # The shape: the first seven fields.
    assert R.DECODE_CONFIG_FIELDS[:7] == (
        "head_size",
        "num_q_heads",
        "num_kv_heads",
        "window",
        "max_query_len",
        "dtype",
        "page_size",
    ), R.DECODE_CONFIG_FIELDS
    # Keyed by (batched, shape).
    headers = (DECODE_CONFIGS_H, DECODE_BATCH_CONFIGS_H)
    lines, rows = [], {}
    for batched, header in enumerate(headers):
        text, entries = read_configs(header)
        lines.append(text)
        rows |= {(batched, *r[:7]): r for r in entries}
    source = jit.source_digest()
    written = 0
    for line in Path(path).read_text().splitlines():
        dec = json.loads(line)
        if dec.get("kind") == "prefill":
            written += apply_prefill(dec, block_sizes)
            continue
        if dec.get("kind") == "prefill_batch":
            written += apply_prefill_batch(dec, block_sizes)
            continue
        if not dec["accepted"]:
            continue
        hq, hkv, d, m, win = dec["hq"], dec["hkv"], dec["d"], dec["m"], dec["window"]
        batched = int(dec.get("batch", False))
        tag = (
            f"{hq}/{hkv}/{d} M={m}"
            + (f" w{win}" if win else "")
            + (" batch" if batched else "")
        )
        if dec["source"] != source:
            print(f"{tag}: refused, measured on another kernel source")
            continue
        now = R.variant_for(
            hq,
            hkv,
            d,
            m,
            dec["block_size"],
            1,
            win,
            utils.torch_dtype(dec["dtype"]),
            bool(batched),
        )
        if dec["current"] != (None if now is None else knobs_of(now)):
            print(f"{tag}: refused, the row changed since it was measured")
            continue
        sizes = sorted({k[7] for k in rows if k[:6] == (batched, d, hq, hkv, win, m)})
        for bf16, bs in itertools.product((0, 1), sizes or block_sizes):
            v = R.KernelVariant(
                d,
                hq,
                hkv,
                m,
                bs,
                1,
                **dec["winner"],
                window=win,
                dtype=torch.bfloat16 if bf16 else torch.float16,
                batched=batched,
            )
            row = R.config_fields(v)
            rows[(batched, *row[:7])] = row
            written += 1
        print(f"{tag}: {label(dec['winner'])}")
    for batched, header in enumerate(headers):
        write_configs(
            header,
            lines[batched],
            [r for k, r in rows.items() if k[0] == batched],
            R.DECODE_CONFIG_FIELDS,
        )
    print(f"# {written} rows written")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--apply", metavar="DECISIONS", help="write decisions to the configs"
    )
    p.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[16],
        help="with --apply: page sizes of a configuration not built yet",
    )
    p.add_argument(
        "--prefill",
        action="store_true",
        help="tune the prefill kernel: every step of M of one configuration",
    )
    p.add_argument(
        "--prefill-batch",
        action="store_true",
        help="tune the prefill batch configs: the prefill build and "
        "head group of every cell of one configuration's batches",
    )
    p.add_argument(
        "--prefixes",
        type=int,
        nargs="+",
        default=list(PREFIXES),
        help="with --prefill: cached tokens before each step's M",
    )
    p.add_argument("--hq", type=int)
    p.add_argument("--hkv", type=int)
    p.add_argument("--head-dim", type=int)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--m", type=int)
    utils.add_dtype_argument(p)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    p.add_argument(
        "--batch",
        nargs="+",
        help="tune the batched row over these batch specs instead of --contexts",
    )
    p.add_argument("--points", help="append every point (JSONL)")
    p.add_argument("--decisions", help="append the decision (JSONL)")
    p.add_argument("--reuse", action="store_true", help="skip points in --points")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--confirm-rounds", type=int, default=3)
    p.add_argument(
        "--min-gain", type=float, default=1.01, help="weighted score to accept"
    )
    p.add_argument("--limit", type=int, default=0, help="first N points (smoke)")
    p.add_argument("--candidates", help="JSONL of knob sets instead of the grid")
    a = p.parse_args()
    if a.apply:
        apply(a.apply, a.block_sizes)
        return
    needed = ("hq", "hkv", "head_dim", "decisions")
    if not a.prefill_batch:
        needed += ("points",)
    batchless = a.prefill or a.prefill_batch
    missing = [
        n for n in (*needed, *(() if batchless else ("m",))) if getattr(a, n) is None
    ]
    if missing:
        p.error("tuning needs --" + ", --".join(n.replace("_", "-") for n in missing))
    if a.prefill_batch:
        tune_prefill_batch(a)
    elif a.prefill:
        tune_prefill(a)
    else:
        tune(a)


if __name__ == "__main__":
    main()
