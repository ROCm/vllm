# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep tuning configs for the ViT TRITON_ATTN kernel on gfx1151.

Default shape is Gemma3-4B SigLIP (B=1, S=4096, H=16, D=72, bf16, non-causal);
pass --seq / --heads / --head-dim / --dtype to target a different ViT (e.g.
Qwen3-Omni's S=3200, fp16). Shape args are forwarded to bench.py and stamped
into the output directory so sweeps for different shapes don't collide.

Calls bench.py for each config. The bench uses triton.testing.do_bench,
which clears the L2 cache before every measurement iteration -- so results
approximate the kernel's behavior inside a real transformer block (where
surrounding qkv-proj / MLP / norm work evicts q/k/v between layer calls).

Phases:
  axis    - vary one parameter at a time around baseline
  refine  - cross-product around top-K axis-best (small)
  custom  - read configs from JSON file
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).parent
BENCH = REPO / "bench.py"

# Baseline from triton_prefill_attention.py:get_block_size/get_num_warps on RDNA bf16:
# BLOCK_M=BLOCK_N=32, num_warps=8, num_stages=1, no waves_per_eu override.
BASELINE = dict(bm=32, bn=32, nw=8, ns=1, we=None)


def _shape_tag(shape: dict) -> str:
    return (
        f"b{shape['batch']}_s{shape['seq']}_h{shape['heads']}"
        f"_d{shape['head_dim']}_{shape['dtype']}"
    )


def _out_dir(shape: dict) -> Path:
    base = os.environ.get(
        "VLLM_VIT_SWEEP_OUT",
        str(Path(tempfile.gettempdir()) / "vit_attn_sweep"),
    )
    p = Path(base) / _shape_tag(shape)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _gpu_lock_cmd(cmd: list[str]) -> list[str]:
    gpu_lock = shutil.which("gpu-lock")
    if gpu_lock:
        return [gpu_lock] + cmd
    return cmd


def _bench_cmd(cfg: dict, shape: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(BENCH),
        "--batch",
        str(shape["batch"]),
        "--seq",
        str(shape["seq"]),
        "--heads",
        str(shape["heads"]),
        "--head-dim",
        str(shape["head_dim"]),
        "--dtype",
        shape["dtype"],
        "--num-layers",
        str(shape["num_layers"]),
        "--bm",
        str(cfg["bm"]),
        "--bn",
        str(cfg["bn"]),
        "--nw",
        str(cfg["nw"]),
        "--ns",
        str(cfg["ns"]),
    ]
    if cfg.get("we") is not None:
        cmd += ["--we", str(cfg["we"])]
    return cmd


def run_one(cfg: dict, shape: dict, out_dir: Path) -> dict:
    tag = f"bm{cfg['bm']}_bn{cfg['bn']}_nw{cfg['nw']}_ns{cfg['ns']}_we{cfg.get('we')}"
    log_out = out_dir / f"{tag}.log"
    json_out = out_dir / f"{tag}.json"
    cmd = _gpu_lock_cmd(_bench_cmd(cfg, shape))
    t0 = time.time()
    with open(log_out, "w") as f:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=f, timeout=600)
    elapsed = time.time() - t0
    if p.returncode != 0:
        return {
            "tag": tag,
            "error": f"rc={p.returncode}, log={log_out}",
            "elapsed": elapsed,
        }
    text = p.stdout.decode().strip().splitlines()
    if not text:
        return {"tag": tag, "error": f"no stdout, log={log_out}", "elapsed": elapsed}
    try:
        data = json.loads(text[-1])
    except json.JSONDecodeError as e:
        return {
            "tag": tag,
            "error": f"bad json: {e}, log={log_out}",
            "elapsed": elapsed,
        }
    json_out.write_text(json.dumps(data, indent=2))
    return {"tag": tag, **data, "elapsed": elapsed}


def axis_sweep() -> list[dict]:
    base = BASELINE
    grid: list[dict] = []
    for bm in (16, 32, 64, 128):
        c = dict(base)
        c["bm"] = bm
        c["bn"] = bm
        grid.append(c)
    for nw in (2, 4, 8, 16):
        c = dict(base)
        c["nw"] = nw
        grid.append(c)
    for ns in (1, 2, 3):
        c = dict(base)
        c["ns"] = ns
        grid.append(c)
    for we in (1, 2, 4, 6, 8):
        c = dict(base)
        c["we"] = we
        grid.append(c)
    seen = set()
    uniq = []
    for c in grid:
        key = tuple(sorted(c.items(), key=lambda kv: kv[0]))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def refine_grid(seed_configs: list[dict]) -> list[dict]:
    bms = sorted({c["bm"] for c in seed_configs} | {16, 32, 64})
    nws = sorted({c["nw"] for c in seed_configs} | {4, 8})
    nss = sorted({c["ns"] for c in seed_configs} | {1})
    wes = sorted(
        {c.get("we") for c in seed_configs if c.get("we") is not None} | {2, 4}
    )
    return [
        dict(bm=bm, bn=bm, nw=nw, ns=ns, we=we)
        for bm, nw, ns, we in itertools.product(bms, nws, nss, wes)
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("axis", "refine", "custom"), default="axis")
    p.add_argument("--configs", default=None)
    p.add_argument(
        "--seed-configs",
        default=None,
        help="for refine: JSON list of axis-best configs",
    )
    p.add_argument(
        "--batch", type=int, default=1, help="ViT batch (forwarded to bench)"
    )
    p.add_argument(
        "--seq", type=int, default=4096, help="ViT seq length (forwarded to bench)"
    )
    p.add_argument(
        "--heads", type=int, default=16, help="ViT num heads (forwarded to bench)"
    )
    p.add_argument(
        "--head-dim", type=int, default=72, help="ViT head dim (forwarded to bench)"
    )
    p.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp16", "fp32"),
        help="ViT dtype (forwarded to bench)",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=27,
        help="ViT depth, used only to scale per-image totals in the report",
    )
    p.add_argument(
        "--out",
        default=None,
        help="results JSON path (default: <out_dir>/sweep_results.json)",
    )
    args = p.parse_args()

    shape = dict(
        batch=args.batch,
        seq=args.seq,
        heads=args.heads,
        head_dim=args.head_dim,
        dtype=args.dtype,
        num_layers=args.num_layers,
    )
    out_dir = _out_dir(shape)
    out_path = Path(args.out) if args.out else out_dir / "sweep_results.json"

    if args.phase == "axis":
        grid = axis_sweep()
    elif args.phase == "refine":
        with open(args.seed_configs) as f:
            seeds = json.load(f)
        grid = refine_grid(seeds)
    else:
        with open(args.configs) as f:
            grid = json.load(f)

    print(f"Shape: {_shape_tag(shape)}  ({shape})", file=sys.stderr)
    print(f"Out dir: {out_dir}", file=sys.stderr)
    print(f"Configs to sweep: {len(grid)}", file=sys.stderr)

    results: list[dict] = []
    for i, c in enumerate(grid):
        print(
            f"[{i + 1}/{len(grid)}] bm={c['bm']} bn={c['bn']} nw={c['nw']} "
            f"ns={c['ns']} we={c.get('we')}",
            file=sys.stderr,
            flush=True,
        )
        r = run_one(c, shape, out_dir)
        results.append({"input_cfg": c, **r})
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        if "error" in r:
            print(f"  ERROR: {r['error']}", file=sys.stderr)
        else:
            print(
                f"  per_call median = {r['per_call_ms_median']:.3f} ms  "
                f"min = {r['per_call_ms_min']:.3f} ms  "
                f"total/image = {r['total_per_image_ms_median']:.1f} ms",
                file=sys.stderr,
            )

    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r["per_call_ms_median"])
    print("\nTop 10:", file=sys.stderr)
    for r in ok[:10]:
        c = r["input_cfg"]
        print(
            f"  bm={c['bm']:>3} bn={c['bn']:>3} nw={c['nw']:>2} ns={c['ns']} "
            f"we={c.get('we')}  "
            f"median={r['per_call_ms_median']:.3f} ms  "
            f"min={r['per_call_ms_min']:.3f} ms",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
