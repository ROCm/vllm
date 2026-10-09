#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune one (Hq, Hkv, D, window, M) of the RDNA3.5 decode kernel, or with
--prefill one (Hq, Hkv, D, window) of the prefill kernel, and apply.

A configuration may not depend on S: the grid is fixed when the CUDA graph is
captured and S is a runtime argument, so one knob set has to serve 128 and
32768 alike.  Every candidate is scored over the whole context range.

1. The knob grid (`grid`, pruned of what the kernel refuses or what builds the
   same code), every point built up front in parallel (jit.py, ~1 s each).
2. Every point checked against a float reference -- a wrong kernel is fast
   because it skips work -- then timed through ROCM_ATTN at each of
   --contexts.  Points are appended to --points as they are measured, so a
   killed run loses nothing; --reuse skips the ones already there.
3. Finalists: the --top best; the best of every simplified subspace (PINS:
   no dot, no prefetch, max_segments a power of two...), which also says
   what each knob is worth; and the best few that lose at no context.  They
   are timed against the current row of rdna35_variants.csv in an
   interleaved A/B, --confirm-rounds rounds over --contexts, medians per
   cell.  One decision line goes to --decisions.

The choice, in the search and in the A/B, scores a point by its ratios to the
current row (row us / point us) at each context:

- it may lose at most loss_limit(S) at any context, a limit that shrinks as
  the context grows (5 % at 128 down to 1 % at 32k): losses, if any, fall on
  the short contexts;
- among those, the best mean of the ratios weighted by sqrt(S / 128) wins if
  it is above --min-gain.  Weighting by time alone lets 32k decide everything
  (a 30 % loss at 128 for 1 % at 32k); flat weights do the opposite.

A row whose knobs the grid (or --candidates) no longer has is still the
reference, but it cannot stay: if no finalist is accepted, the best one by the
same rule replaces it, so that every row ends up inside the search space and
knobs left out of it can leave the kernel.

A configuration without a row ranks by the worst long-context cell (S >=
16384) up to 90 % of roof first, then the geomean of %roof.

    cd <worktree> && PYTHONPATH=$PWD amd-gpu-lock \\
        <venv>/bin/python benchmarks/kernels/gfx1151_decode_attn/tools/tune.py \\
        --hq 32 --hkv 8 --head-dim 128 --m 1 \\
        --points points.jsonl --decisions decisions.jsonl

`--apply decisions.jsonl` writes accepted decisions into rdna35_variants.csv:
the configuration's rows in both dtypes, for one sequence and for a batch, at
every block size the configuration has (--block-sizes for a new one).  A batch
row whose winner is the dot decomposition takes the best WMMA point instead:
the dot path buys latency for one sequence and loses once a batch fills the
machine.  A decision measured against another kernel source or another row
is refused.

--prefill tunes the prefill kernel (rdna35_prefill_attn.cu), one row per step
of M (STEPS), all steps of a configuration in one run.  It runs outside CUDA
graphs and the host knows M, so a row only has to serve its step:

1. The prefill knob grid (`prefill_grid`), built up front; every point checked
   against Triton's unified attention, then timed calling the op directly
   (L2 flushed) at each cell: the start of every step after each of
   --prefixes cached tokens.
2. Per step, the --top best by the weighted mean of their speedups over the
   step's fastest point, the current row and the previous step's winner are
   timed through ROCM_ATTN in an interleaved A/B with TRITON_ATTN and
   ROCM_SEGMENTED_ATTN, each in the KV cache layout vLLM gives it.
3. Per step, a point that is slower than the best baseline at no cell wins
   over one that is, then the best mean of its speedups weighted by
   sqrt(FLOPs); the previous step's row stays if it is as safe and within
   PREFILL_SLACK, so that steps share rows.  The decision is one line with
   every step's winner.

`--apply` writes a prefill decision's rows into rdna35_prefill_variants.csv,
both dtypes, every page size the configuration has (--block-sizes for a new
one), contiguous steps with the same winner merged.
"""

import argparse
import csv
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

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "attention_benchmarks"))
sys.path.insert(0, str(_HERE.parent))

import utils  # noqa: E402

# The harness's stand-in model config is cached; resolving its revision online
# costs ~0.1 s per cell.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

CONTEXTS = [128, 512, 1024, 4096, 8192, 16384, 32768]
LONG_S, LONG_TARGET = 16384, 0.90
KNOBS = utils.KNOBS
# Relative error bound check() applies per dtype; bf16 output rounding alone
# is up to 2^-8 relative.
RTOL = {"fp16": 1e-3, "bf16": 8e-3}

# The search space.  max_segments: segments per (kv head, row group);
# workgroups at long context = Hkv * row_groups * max_segments, capped at
# MAX_WG.  max_segments 16 and 32 left the grid: across a full fp16 tuning they
# moved no row beyond noise.
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
# Some knob sets deadlock the GPU (a split-KV merge waiting for workgroups
# that never run).  A point with no progress for this long is recorded as
# "hang" and the process exits with HANG_RC; rerun with --reuse to go on.
HANG_S = 240
HANG_RC = 75
# The harness collects garbage after every timing, 0.38 of a cell's ~0.6 s;
# the loop collects every GC_EVERY points instead.
GC_EVERY = 25


def dspl_rule(d):
    """The kernel's HEAD_DIM_SPLIT when the row says 0."""
    return d // 128 if d >= 256 else 1


def grid(hq, hkv, d, m):
    """Knob sets for one configuration, without the ones the kernel refuses
    (static_asserts) or that build the same code as another
    (min_segment_blocks at max_segments 1)."""
    gqa = hq // hkv
    rule = dspl_rule(d)
    out = []
    # Row groups split the gqa * m (head, token) rows evenly; they need not
    # split whole q heads.
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
                if (d // head_dim_split) // 16 not in (4, 8) or waves % head_dim_split:
                    continue
                out.append(
                    {
                        "waves": waves,
                        "row_groups": row_groups,
                        "max_segments": max_segments,
                        "min_segment_blocks": min_segment_blocks,
                        "prefetch": int(pipe == "prefetch"),
                        "v_in_lds": int(pipe == "v_in_lds"),
                        # head_dim_split 0 is the kernel's rule; spell the rule as 0.
                        "head_dim_split": 0
                        if head_dim_split == rule
                        else head_dim_split,
                    }
                )
    for waves, max_segments in itertools.product(*DOT_GRID.values()):
        out.append({"dot_product": 1, "waves": waves, "max_segments": max_segments})
    return out


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def rank(contexts, pct):
    long_ = [c for s, c in zip(contexts, pct) if s >= LONG_S]
    return (min(min(long_), LONG_TARGET) if long_ else LONG_TARGET, geomean(pct))


def loss_limit(s):
    return min(0.05, max(0.01, 0.05 - 0.005 * (math.log2(s) - 7)))


def weighted(contexts, ratio):
    w = [math.sqrt(s / 128) for s in contexts]
    return sum(a * b for a, b in zip(w, ratio)) / sum(w)


def within_limits(contexts, ratio):
    return all(r >= 1 - loss_limit(s) for s, r in zip(contexts, ratio))


# Simplified subspaces whose best point is a finalist too.  A predicate over
# the knobs and the kernel's head_dim_split rule.
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
    """The knobs that change a build: the dot path reads only waves and
    max_segments, min_segment_blocks is moot at max_segments 1, and
    head_dim_split 0 is the kernel's rule."""
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


def check(module, v, repeat=2):
    """Worst relative error over four contexts -- a short one, one past a page,
    a partial tile and the window, a longer one, a short one again -- each on
    new inputs and launched `repeat` times on one scratch (the split-KV
    counters must come back clean), the output NaN-filled before every launch.
    New inputs catch a merge that leaves part of the output to stale data (the
    same inputs again would reproduce the previous, correct numbers); NaN
    catches what is not written."""
    import torch

    from vllm.v1.attention.ops.rdna35_hip_decode import make_scratch

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
    from vllm.v1.attention.ops.rdna35_hip_decode import KernelVariant, variant_for

    jit.install()
    common.gc = types.SimpleNamespace(collect=lambda *x: 0)
    dtype = utils.torch_dtype(a.dtype)
    hq, hkv, d, m, win, bs = a.hq, a.hkv, a.head_dim, a.m, a.window, a.block_size
    ctxs = list(a.contexts)
    tag = f"{hq}/{hkv}/{d} M={m}" + (f" w{win}" if win else "") + f" {a.dtype}"
    fixed = {
        "hq": hq,
        "hkv": hkv,
        "d": d,
        "m": m,
        "window": win,
        "dtype": a.dtype,
        "block_size": bs,
        "source": jit.source_digest(),
        "contexts": ctxs,
    }
    meta = {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "host": platform.node(),
    }
    roof = [
        utils.roofline_us(hq, hkv, d, m, s, block_size=bs, window=win) for s in ctxs
    ]

    def variant(knobs):
        return KernelVariant(d, hq, hkv, m, bs, 1, **knobs, dtype=dtype, window=win)

    def decode(v, s):
        """(us, spread %) of one sequence of m tokens at context s, on v."""
        from common import BenchmarkConfig

        cfg = BenchmarkConfig(
            backend="ROCM_ATTN",
            batch_spec=f"q{m}s{s}",
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
        if path() != "kernel":
            raise RuntimeError(f"ran {path()} at S={s}")
        return r.median_time * 1e6, r.std_time / r.median_time * 100

    def pid(knobs):
        return json.dumps(sorted(knobs.items()))

    known: dict[str, dict] = {}
    if a.reuse and Path(a.points).exists():
        for line in Path(a.points).read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a line cut by a kill
                continue
            if all(rec.get(k) == v for k, v in fixed.items()):
                known[pid(rec["knobs"])] = rec

    current = variant_for(hq, hkv, d, m, bs, 1, win, dtype, False)
    raw = (
        [json.loads(x) for x in Path(a.candidates).read_text().splitlines() if x]
        if a.candidates
        else grid(hq, hkv, d, m)
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
    jit.precompile(todo)
    jit.seal()
    print(
        f"# {tag}: {len(items)} points, {len(items) - len(todo)} reused, "
        f"built in {time.time() - t0:.0f} s",
        flush=True,
    )
    time.sleep(5)  # a parallel build moves the SoC clock

    # Watchdog: a hung point is recorded and the process exits.
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
                    rec["max_rel"] = err = check(mod, v)
                    if err > tol:
                        rec["status"] = "wrong"
                    else:
                        us, spread = zip(*(decode(v, s) for s in ctxs))
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
            if (i + 1) % GC_EVERY == 0:
                gc.collect()
            if rec.get("status") == "ok":
                scored.append((rank(ctxs, rec["pct_roof"]), v, k))
            if (i + 1) % 100 == 0:
                print(
                    f"#   {i + 1}/{len(items)} points, "
                    f"{(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )

    # The current row's search times, the reference of the ranking; without
    # one (no row, or it failed) the roof ranking stands.
    cur = None if current is None else known.get(pid(knobs_of(current)))
    ref_us = cur["us"] if cur is not None and cur.get("status") == "ok" else None
    if ref_us is not None:
        scored = [
            ((within_limits(ctxs, r), weighted(ctxs, r)), v, k)
            for _, v, k in scored
            for r in [[b / u for b, u in zip(ref_us, known[pid(k)]["us"])]]
        ]
    scored.sort(key=lambda x: x[0], reverse=True)
    wmma = next(
        (
            x
            for x in scored
            if not x[2]["dot_product"] and not (outside and x[1] == current)
        ),
        None,
    )
    rule = dspl_rule(d)
    finalists: dict = {}
    for _, v, k in scored[: a.top]:
        finalists.setdefault(v, (k, ["top"]))
    for name, pred in PINS.items():
        best_in = next((x for x in scored if pred(x[2], rule)), None)
        if best_in is not None:
            finalists.setdefault(best_in[1], (best_in[2], []))[1].append(name)
    if ref_us is not None:
        # The best that lose nowhere in the search: the ranking alone can
        # fill the A/B with points that all lose a little at one context.
        safe = [
            x
            for x in scored
            if x[1] != current
            and all(b >= u for b, u in zip(ref_us, known[pid(x[2])]["us"]))
        ]
        for _, v, k in safe[:3]:
            finalists.setdefault(v, (k, []))[1].append("safe")
    finalists.pop(current, None)

    # Interleaved A/B: each round times every context on all of them in turn.
    runs = [(v, k, tags) for v, (k, tags) in finalists.items()]
    if current is not None:
        runs.insert(0, (current, knobs_of(current), ["current"]))
    times = [{s: [] for s in ctxs} for _ in runs]
    for _ in range(a.confirm_rounds):
        for s in ctxs:
            for (v, _, _), t in zip(runs, times):
                try:
                    t[s].append(decode(v, s)[0])
                except Exception:  # noqa: BLE001
                    t[s].append(float("inf"))
    med = [[statistics.median(t[s]) for s in ctxs] for t in times]
    base = med[0] if current is not None else None
    results = []
    for (v, k, tags), us in zip(runs, med):
        if "current" in tags:
            continue
        # Against the row, or against roof for a configuration without one.
        ratio = [b / u for b, u in zip(base or roof, us)]
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
    # Every finalist competes, whatever made it one.
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
        # The row was outside the search space and nothing beat it.
        "replaced": replaced,
        "winner": winner,
        "wmma": wmma[2] if wmma else None,
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

# Steps of M: a row serves its step, from the first value up to the next one.
STEPS = (128, 256, 512, 1024, 2048, 4096)
PREFIXES = (0, 1024, 16384)
# A step keeps the previous step's row within this factor of its winner.
PREFILL_SLACK = 1.02
PREFILL_BASELINES = ("TRITON_ATTN", "ROCM_SEGMENTED_ATTN")


def prefill_grid(d):
    """Prefill knob sets for head size d: the head dim splits that keep a
    lane's V slice to 4..16 elements, 1 for D <= 128."""
    split = max(1, d // 128)
    if d <= 128:
        axes = {
            "waves": (2, 4, 8),
            "key_tile": (16, 32, 64),
            "v_t": (0, 1),
            "key_waves": (1, 2),
            "double_buffer": (0,),
        }
    elif d == 256:
        axes = {
            "waves": (4, 8, 16),
            "key_tile": (16, 32),
            "v_t": (0, 1),
            "key_waves": (1,),
            "double_buffer": (0, 1),
        }
    else:
        axes = {
            "waves": (8, 16),
            "key_tile": (16,),
            "v_t": (0,),
            "key_waves": (1,),
            "double_buffer": (0, 1),
        }
    axes |= {"max_segments": (8, 16), "head_group": (4, 8)}
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
    """Mean of the speedups (other / ours), weighted, in log space."""
    return math.exp(sum(w * math.log(r) for w, r in zip(weights, ratio)) / sum(weights))


def tune_prefill(a) -> None:
    import common
    import jit
    import torch
    from runner import run_attention_benchmark
    from triton.testing import do_bench

    import vllm
    import vllm.v1.attention.ops.rdna35_hip_decode as R
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
    time.sleep(5)  # a parallel build moves the SoC clock

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
        key = v.scratch_rows()
        if key not in scratch:
            scratch[key] = R.make_prefill_scratch([v], dev)
        s = int(sl.item())
        return lambda: module.prefill_attn(
            q, kv, table, out, *scratch[key], sl, s, d**-0.5
        )

    # 1. Search: correctness, then the op directly at every cell.
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

    # 2. Finalists per step, timed through ROCM_ATTN against the baselines.
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
                        harness("ROCM_ATTN", *c, dict(json.loads(k)))
                    )
    med = {key: statistics.median(t) for key, t in times.items()}

    # 3. Per step: safe first, then the weighted speedup over the best baseline.
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
    """Write one prefill decision's rows into rdna35_prefill_variants.csv."""
    import jit
    import torch

    import vllm.v1.attention.ops.rdna35_hip_decode as R

    hq, hkv, d, win = dec["hq"], dec["hkv"], dec["d"], dec["window"]
    tag = f"{hq}/{hkv}/{d}" + (f" w{win}" if win else "") + " prefill"
    if dec["source"] != jit.source_digest(prefill=True):
        print(f"{tag}: refused, measured on another kernel source")
        return 0
    path = R.PREFILL_VARIANTS_CSV
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [list(map(int, r)) for r in reader]
    col = {c: i for i, c in enumerate(header)}
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
            r[col["HEAD_DIM"]],
            r[col["NUM_Q_HEADS"]],
            r[col["NUM_KV_HEADS"]],
            r[col["WINDOW"]],
        ) == (d, hq, hkv, win)

    sizes = sorted({r[col["PAGE_SIZE"]] for r in rows if mine(r)}) or block_sizes
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
        rows.append(list(R.prefill_variant_defines(v).values()))
        written += 1
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(sorted(rows))
    for step in merged:
        print(f"{tag} M>={step['m']}: {prefill_label(step['winner'])}")
    return written


# ------------------------------------------------------------------- --apply --


def apply(path: str, block_sizes: list[int]) -> None:
    """Write the accepted decisions of `path` into rdna35_variants.csv, and
    prefill decisions into rdna35_prefill_variants.csv."""
    import jit
    import torch

    import vllm.v1.attention.ops.rdna35_hip_decode as R

    csv_path = R.VARIANTS_CSV
    with csv_path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = {tuple(map(int, r[:8])): list(map(int, r)) for r in reader}
    assert header[:8] == [
        "HEAD_DIM",
        "NUM_Q_HEADS",
        "NUM_KV_HEADS",
        "WINDOW",
        "MAX_QUERY_LEN",
        "BF16",
        "BATCHED",
        "PAGE_SIZE",
    ], header
    source = jit.source_digest()
    written = 0
    for line in Path(path).read_text().splitlines():
        dec = json.loads(line)
        if dec.get("kind") == "prefill":
            written += apply_prefill(dec, block_sizes)
            continue
        if not dec["accepted"]:
            continue
        hq, hkv, d, m, win = dec["hq"], dec["hkv"], dec["d"], dec["m"], dec["window"]
        tag = f"{hq}/{hkv}/{d} M={m}" + (f" w{win}" if win else "")
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
            False,
        )
        if dec["current"] != (None if now is None else knobs_of(now)):
            print(f"{tag}: refused, the row changed since it was measured")
            continue
        sizes = sorted({k[7] for k in rows if k[:5] == (d, hq, hkv, win, m)})
        for bf16, batch, bs in itertools.product((0, 1), (0, 1), sizes or block_sizes):
            knobs = dec["winner"]
            if batch and knobs["dot_product"]:
                knobs = dec["wmma"]
            v = R.KernelVariant(
                d,
                hq,
                hkv,
                m,
                bs,
                1,
                **knobs,
                batched=batch,
                window=win,
                dtype=torch.bfloat16 if bf16 else torch.float16,
            )
            row = list(R.variant_defines(v).values())
            rows[tuple(row[:8])] = row
            written += 1
        print(f"{tag}: {label(dec['winner'])}")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows[k] for k in sorted(rows))
    print(f"# {written} rows written")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", metavar="DECISIONS", help="write decisions to the CSV")
    p.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[16],
        help="with --apply: page sizes of a configuration the CSV does not have",
    )
    p.add_argument(
        "--prefill",
        action="store_true",
        help="tune the prefill kernel: every step of M of one configuration",
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
    needed = ("hq", "hkv", "head_dim", "points", "decisions")
    missing = [
        n for n in (*needed, *(() if a.prefill else ("m",))) if getattr(a, n) is None
    ]
    if missing:
        p.error("tuning needs --" + ", --".join(n.replace("_", "-") for n in missing))
    if a.prefill:
        tune_prefill(a)
    else:
        tune(a)


if __name__ == "__main__":
    main()
