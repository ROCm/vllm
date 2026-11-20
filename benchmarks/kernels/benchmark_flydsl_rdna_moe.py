# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark of the FlyDSL W4A16 MoE kernels (gfx115x) at model shapes.

Runs ``flydsl_rdna_moe``, the function the ``rdna_flydsl`` MoE backend calls,
the way a model runs it: one graph replays a stack of layers, every layer has
its own expert weights (so they are read cold from DRAM, as in a model with
tens of GiB of weights), and what shares the GPU with the MoE in a model is
part of the graph:

- other weights streamed between two MoE layers (``context_mb``),
- cached traffic that displaces the kernels' code and small buffers from the
  caches (``evict_mb``): without it the one-token kernels are up to 12%
  faster than in a model,
- the router, and
- the shared expert, which runs on a second stream next to GEMM1.

Replays are waited for one by one, like decode steps. Prefill times depend
on how tokens spread over experts; ``--routing`` replays captured ids.

Two timings are reported per (shape, number of tokens):

- ``layer us``: graph replay time per layer minus the same graph without
  the MoE, and
- with ``--kernel-trace``: the mean duration of every MoE kernel dispatch
  from ``rocprofv3 --kernel-trace``, the numbers a kernel trace of a model
  run shows for the same kernel names.

Examples:
    python benchmarks/kernels/benchmark_flydsl_rdna_moe.py
    python benchmarks/kernels/benchmark_flydsl_rdna_moe.py \
        --shapes qwen3.6-35b-a3b-g128 --tokens 1 2 4 --kernel-trace

"""

import argparse
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class MoEShape:
    hidden: int
    inter: int
    experts: int
    top_k: int
    group: int
    layers: int
    # Tokens per MoE call: decode steps and prefills of the regression rows.
    tokens: tuple[int, ...]
    models: str
    zero_points: bool = False
    activation: str = "silu"
    # Width of the shared expert that overlaps GEMM1 (0: none).
    shared_inter: int = 0
    # Other weights read between two MoE layers, MiB.
    context_mb: float = 16.0
    # Cached traffic between two MoE layers, MiB; matched to kernel traces of
    # the models (16 for Qwen3.6-35B-A3B, 8 for Qwen3-30B-A3B).
    evict_mb: float = 8.0


# Routed-expert shapes of the quantized MoE rows of the nightly regression.
SHAPES = {
    "qwen3.6-35b-a3b-g128": MoEShape(
        2048,
        512,
        256,
        8,
        128,
        40,
        (1, 2, 4, 512, 4352),
        "Qwen3.5/3.6-35B-A3B W4A16 (llm-compressor)",
        shared_inter=512,
        evict_mb=16.0,
    ),
    "qwen3.6-35b-a3b-g32-zp": MoEShape(
        2048,
        512,
        256,
        8,
        32,
        40,
        (1, 512),
        "Qwen3.6-35B-A3B AWQ",
        zero_points=True,
        shared_inter=512,
        evict_mb=16.0,
    ),
    "qwen3.5-35b-a3b-g32": MoEShape(
        2048,
        512,
        256,
        8,
        32,
        40,
        (1, 512),
        "Qwen3.5-35B-A3B AWQ",
        shared_inter=512,
        evict_mb=16.0,
    ),
    "qwen3-30b-a3b-g32": MoEShape(
        2048,
        768,
        128,
        8,
        32,
        48,
        (1, 2, 128, 4096),
        "Qwen3-30B-A3B AWQ, Qwen3-Omni-30B-A3B AWQ",
    ),
    "qwen3-30b-a3b-g128": MoEShape(
        2048, 768, 128, 8, 128, 48, (1, 128), "Qwen3-30B-A3B .w4a16"
    ),
    "gemma-4-26b-a4b-g32": MoEShape(
        2816,
        704,
        128,
        8,
        32,
        30,
        (1, 800),
        "Gemma-4-26B-A4B AWQ",
        activation="gelu_tanh",
    ),
}


# Small kernels of different code, as the layers of a model run between MoEs.
_FILLERS = [
    torch.sigmoid,
    torch.tanh,
    torch.exp,
    torch.abs,
    torch.neg,
    torch.sqrt,
    torch.cos,
    torch.sin,
    torch.erf,
    torch.floor,
    lambda x: x + x,
    lambda x: x * x,
    lambda x: torch.cumsum(x, dim=-1),
    lambda x: torch.sum(x, dim=-1),
    lambda x: torch.amax(x, dim=-1),
    lambda x: torch.nn.functional.softmax(x, dim=-1),
    lambda x: torch.sort(x, dim=-1),
    lambda x: torch.nn.functional.silu(x),
    lambda x: torch.nn.functional.gelu(x),
    lambda x: x.to(torch.bfloat16),
    lambda x: torch.where(x > 0, x, -x),
    lambda x: torch.roll(x, 1, dims=-1),
    lambda x: torch.flip(x, dims=(-1,)),
    lambda x: torch.clamp(x, -1, 1),
]


def _packed_layer(shape: MoEShape, device, check: bool):
    """One layer's packed weights and, with ``check``, its dequantized
    reference weights for the first experts."""
    from vllm.model_executor.layers.fused_moe.fused_flydsl_rdna_moe import (
        gate_up_tile_order,
        pack_w4a16_weights,
    )

    g = torch.Generator(device=device).manual_seed(0)
    ref = {}

    def pack(name, n_out, n_in, order):
        ws, cs = [], []
        for i in range(0, shape.experts, 8):
            q = torch.randint(0, 16, (8, n_out, n_in), device=device, generator=g)
            groups = n_in // shape.group
            scale = torch.rand((8, n_out, groups), device=device, generator=g)
            scale = (scale * 0.02 + 0.001).to(torch.bfloat16)
            zp = None
            if shape.zero_points:
                zp = torch.randint(0, 16, scale.shape, device=device, generator=g)
            if check:
                z = 8 if zp is None else zp.repeat_interleave(shape.group, dim=2)
                deq = (q - z).float() * scale.float().repeat_interleave(
                    shape.group, dim=2
                )
                ref.setdefault(name, []).append(deq.to(torch.bfloat16))
            if order is not None:
                q, scale = q[:, order], scale[:, order]
                zp = None if zp is None else zp[:, order]
            w, c = pack_w4a16_weights(q, scale, zp)
            ws.append(w)
            cs.append(c)
        return torch.cat(ws), torch.cat(cs)

    gate_up = 2 * shape.inter
    order = None
    if shape.activation == "silu":
        order = gate_up_tile_order(gate_up, device)
    packed = pack("w1", gate_up, shape.hidden, order) + pack(
        "w2", shape.hidden, shape.inter, None
    )
    return packed, {k: torch.cat(v) for k, v in ref.items()}


class Stack:
    """A stack of MoE layers with distinct weights and its graphs."""

    def __init__(
        self,
        shape: MoEShape,
        layers: int,
        context: bool,
        check: bool,
        hip: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from vllm.model_executor.layers.fused_moe.fused_flydsl_rdna_moe import (
            pad_expert_stride,
        )

        self.shape = shape
        self.layers = layers
        self.dev = dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(1)
        self.hip = hip
        self.dtype = dtype
        if hip:
            self.w = [self._hip_layer(g) for _ in range(layers)]
        else:
            packed, self.ref = _packed_layer(shape, dev, check)
            # Layer 0 is the packed layer; the others hold its experts permuted.
            self.perm = [torch.arange(shape.experts, device=dev)]
            self.perm += [
                torch.randperm(shape.experts, device=dev, generator=g)
                for _ in range(layers - 1)
            ]
            self.w = [[pad_expert_stride(t[p]) for t in packed] for p in self.perm]
            del packed
        hidden = shape.hidden
        n_ctx = round(shape.context_mb * 2**20 / (hidden * hidden * 2))
        n_ctx = n_ctx if context else 0
        self.ctx_w = [
            [
                torch.randn((hidden, hidden), device=dev, generator=g).to(
                    torch.bfloat16
                )
                for _ in range(n_ctx)
            ]
            for _ in range(layers)
        ]

        def bf16(n_out, n_in):
            w = torch.randn((n_out, n_in), device=dev, generator=g)
            return w.to(torch.bfloat16)

        self.router_w = []
        if context:
            self.router_w = [bf16(shape.experts, hidden) for _ in range(layers)]
        self.shared_w = []
        if shape.shared_inter and context:
            s = shape.shared_inter
            self.shared_w = [
                (bf16(2 * s, hidden), bf16(hidden, s), bf16(1, hidden).view(-1))
                for _ in range(layers)
            ]
        self.cu = torch.cuda.get_device_properties(0).multi_processor_count
        self.evict = None
        if context and shape.evict_mb:
            n = int(shape.evict_mb * 2**20 / 4)
            self.evict = (torch.randn(n, device=dev), torch.empty(n, device=dev))
        self.events = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(layers)]
        self.gen = g

    def _hip_layer(self, g: torch.Generator) -> list[torch.Tensor]:
        """Random weights in the layout of the HIP int4 MoE decode kernel:
        little-endian nibbles ``[E, N, K / 8]`` int32, scales ``[E, N, K / group]``."""
        s = self.shape
        out = []
        for n_out, n_in in ((2 * s.inter, s.hidden), (s.hidden, s.inter)):
            k8 = n_in // 8
            # The loader pads rows whose stride is a multiple of 2048 B.
            pad = 32 if (k8 * 4) % 2048 == 0 else 0
            w = torch.randint(
                -(2**31),
                2**31 - 1,
                (s.experts, n_out, k8 + pad),
                device=self.dev,
                dtype=torch.int32,
                generator=g,
            )
            scale = torch.rand(
                (s.experts, n_out, n_in // s.group), device=self.dev, generator=g
            )
            out += [w[:, :, :k8], (scale * 0.02 + 0.001).to(torch.bfloat16)]
        return out

    def moe_hip(self, layer: int) -> None:
        """A few tokens on the HIP int4 MoE kernels the hybrid backend decodes
        with: GEMM1, SiLU-and-multiply, GEMM2, top-k sum."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
            moe_unpermute,
        )

        s = self.shape
        w1, s1, w2, s2 = self.w[layer]
        ids = self.ids[layer].view(-1)
        rows = self.rows
        n = rows.numel()
        o1, act, o2 = self.ws.split([n * 2 * s.inter, n * s.inter, n * s.hidden])
        o1, act, o2 = o1.view(n, -1), act.view(n, -1), o2.view(n, -1)
        ops.fused_moe_wvSplitK_int4_gemm(
            self.x, w1, s1, o1, ids, 1, self.cu, s.group, None, rows, s.top_k
        )
        torch.ops._C.silu_and_mul(act, o1)
        ops.fused_moe_wvSplitK_int4_gemm(
            act, w2, s2, o2, ids, 1, self.cu, s.group, None, rows, 1
        )
        moe_unpermute(
            out=self.out,
            permuted_hidden_states=o2,
            topk_weights=self.tw[layer],
            inv_permuted_idx=rows,
        )

    def inputs(self, tokens: int, routing: list[torch.Tensor] | None):
        s, dev = self.shape, self.dev
        self.x = torch.randn((tokens, s.hidden), device=dev, generator=self.gen).to(
            self.dtype
        )
        # The context kernels are bf16 whatever the MoE runs in.
        self.x1 = self.x[:1].to(torch.bfloat16)
        self.xf = self.x1.float()
        self.ids, self.tw = [], []
        for layer in range(self.layers):
            if routing is not None:
                ids = routing[layer % len(routing)][:tokens].to(dev)
            else:
                ids = (
                    torch.rand((tokens, s.experts), device=dev, generator=self.gen)
                    .topk(s.top_k)
                    .indices
                )
            self.ids.append(ids.to(torch.int32).contiguous())
            tw = torch.rand((tokens, s.top_k), device=dev, generator=self.gen)
            self.tw.append(tw / tw.sum(dim=1, keepdim=True))
        rows = tokens * s.top_k
        self.ws = torch.empty(
            rows * (2 * s.inter + s.inter + s.hidden), device=dev, dtype=self.dtype
        )
        self.out = torch.empty((tokens, s.hidden), device=dev, dtype=self.dtype)
        self.rows = torch.arange(rows, device=dev, dtype=torch.int32)

    def _activation(self):
        if self.shape.activation == "silu":
            return None
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
            apply_moe_activation,
        )

        act = MoEActivation(self.shape.activation)
        return lambda out, inp: apply_moe_activation(act, out, inp)

    def moe(self, layer: int) -> None:
        from vllm.model_executor.layers.fused_moe.fused_flydsl_rdna_moe import (
            flydsl_rdna_moe,
        )

        if self.hip:
            return self.moe_hip(layer)
        s = self.shape
        w1, c1, w2, c2 = self.w[layer]
        flydsl_rdna_moe(
            self.out,
            self.x,
            w1,
            c1,
            w2,
            c2,
            self.tw[layer],
            self.ids[layer],
            self.ws,
            s.group,
            s.experts,
            s.inter,
            self._activation(),
            s.zero_points,
            s.zero_points,
        )

    def layer(self, layer: int, moe: bool) -> None:
        """A layer as the GPU sees it in a model: other weights, then the
        shared expert on the second stream next to the router and the MoE."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
            fused_topk,
        )
        from vllm.model_executor.layers.utils import _tiny_dot_triton
        from vllm.utils.torch_utils import aux_stream

        for w in self.ctx_w[layer]:
            ops.wvSplitK(w, self.x1, self.cu)
        if self.evict is not None:
            # A copy, not a read: only with written lines in the caches do
            # the kernel times match a model.
            self.evict[1].copy_(self.evict[0])
        for f in _FILLERS[: int(os.environ.get("BENCH_FILLERS", "0"))]:
            f(self.xf)
        done = None
        if self.shared_w:
            main, aux = torch.cuda.current_stream(), aux_stream()
            ready, done = self.events[layer]
            ready.record(main)
            with torch.cuda.stream(aux):
                ready.wait(aux)
                gate_up, down, gate = self.shared_w[layer]
                h = ops.wvSplitK(gate_up, self.x1, self.cu)
                scale = torch.sigmoid(_tiny_dot_triton(self.x1.view(-1), gate).view(1))
                ops.wvSplitK_fused_silu_gate_mul(down, h, scale, self.cu)
                done.record(aux)
        if self.router_w:
            logits = ops.wvSplitK(self.router_w[layer], self.x1, self.cu)
            fused_topk(self.x1, logits, self.shape.top_k, renormalize=True)
        if moe:
            self.moe(layer)
        if done is not None:
            done.wait(torch.cuda.current_stream())

    def graph(self, moe: bool) -> torch.cuda.CUDAGraph:
        for layer in range(min(2, self.layers)):
            self.layer(layer, moe)
        torch.accelerator.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for layer in range(self.layers):
                self.layer(layer, moe)
        torch.accelerator.synchronize()
        return g

    def check(self, tokens: int) -> float:
        """Max relative error of layer 0 against dequantize-then-matmul for
        the first tokens."""
        s = self.shape
        self.moe(0)
        n = min(tokens, 4)
        ids = self.ids[0][:n].long()
        x = self.x[:n].float()
        h = torch.einsum("tenk,tk->ten", self.ref["w1"][ids].float(), x)
        gate, up = h.to(self.dtype).float().chunk(2, dim=-1)
        if s.activation == "silu":
            act = torch.nn.functional.silu(gate) * up
        else:
            act = torch.nn.functional.gelu(gate, approximate="tanh") * up
        act = act.to(self.dtype).float()
        down = torch.einsum("tenk,tek->ten", self.ref["w2"][ids].float(), act)
        ref = (down * self.tw[0][:n].unsqueeze(-1)).sum(dim=1)
        return ((self.out[:n].float() - ref).abs().max() / ref.std()).item()


_ROUNDS = 5
_TRACED = "FLYDSL_RDNA_MOE_BENCH_TRACED"
_WARMUP = 3


def _time_graph(g: torch.cuda.CUDAGraph, replays: int) -> float:
    """Median seconds per replay. Replays do not overlap: like the decode
    steps of a model, each one is waited for before the next starts."""
    n = _ROUNDS * replays
    # Timing events and a second stream stall rocprofv3; the trace has the
    # kernel times then, so the host clock is enough.
    host_clock = bool(os.environ.get(_TRACED))
    times = []
    start, end = torch.cuda.Event(not host_clock), torch.cuda.Event(not host_clock)
    for i in range(_WARMUP + n):
        t0 = time.perf_counter()
        if not host_clock:
            start.record()
        g.replay()
        if not host_clock:
            end.record()
        torch.accelerator.synchronize()
        if i >= _WARMUP:
            dt = time.perf_counter() - t0
            times.append(dt if host_clock else start.elapsed_time(end) * 1e-3)
    return statistics.median(times)


def _mark(name: str, **kw) -> None:
    torch.accelerator.synchronize()
    clocks = {
        "mono": time.clock_gettime_ns(time.CLOCK_MONOTONIC),
        "real": time.clock_gettime_ns(time.CLOCK_REALTIME),
        "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME),
    }
    print("MARK " + json.dumps({"name": name, **clocks, **kw}), flush=True)


def run(args) -> None:
    torch.set_grad_enabled(False)
    if args.row_path_max_tokens is not None:
        from vllm.model_executor.layers.fused_moe import fused_flydsl_rdna_moe

        fused_flydsl_rdna_moe.ROW_PATH_MAX_TOKENS = args.row_path_max_tokens
    for name in args.shapes:
        shape = SHAPES[name]
        layers = args.layers or shape.layers
        tokens_list = args.tokens or shape.tokens
        check = not args.no_check
        if args.hip:
            if shape.activation != "silu" or shape.zero_points:
                print(f"{name}: no HIP reference for this checkpoint format")
                continue
            check = False
            tokens_list = [t for t in tokens_list if t <= 5]
        dtype = getattr(torch, args.dtype)
        stack = Stack(shape, layers, not args.no_context, check, args.hip, dtype)
        routing = torch.load(args.routing) if args.routing else None
        for tokens in tokens_list:
            stack.inputs(tokens, routing)
            err = stack.check(tokens) if check else float("nan")
            assert not err > 5e-2, f"{name} M={tokens}: relative error {err:.3g}"
            replays = max(2, min(20, 4096 // tokens))
            g_ctx = stack.graph(moe=False)
            g_moe = stack.graph(moe=True)
            t_ctx = _time_graph(g_ctx, replays)
            _mark("begin", shape=name, tokens=tokens)
            t_moe = _time_graph(g_moe, replays)
            n = _WARMUP + _ROUNDS * replays
            _mark("end", shape=name, tokens=tokens, calls=layers * n)
            # The replayed stack has to leave the last layer's result.
            replayed = stack.out.clone()
            stack.moe(layers - 1)
            torch.accelerator.synchronize()
            same = torch.equal(replayed.view(torch.int16), stack.out.view(torch.int16))
            assert same or not check, f"{name} M={tokens}: replay differs"
            us = (t_moe - t_ctx) / layers * 1e6
            rows = tokens * shape.top_k
            gate_up = 2 * shape.inter
            flops = 2 * rows * shape.hidden * (gate_up + shape.inter)
            # Weights and coefficients of the routed experts, each read once.
            used = sum(len(torch.unique(i)) for i in stack.ids) / layers
            w_bytes = used * sum(t[0].numel() * t.element_size() for t in stack.w[0])
            label = name + ("/hip" if args.hip else "")
            print(
                "RESULT "
                + json.dumps(
                    {
                        "shape": label,
                        "tokens": tokens,
                        "layer_us": round(us, 2),
                        "context_us": round(t_ctx / layers * 1e6, 2),
                        "tflops": round(flops / us * 1e-6, 2),
                        "weight_gbps": round(w_bytes / us * 1e-3, 1),
                        "experts_used": round(used, 1),
                        "max_rel_err": round(err, 4),
                    }
                ),
                flush=True,
            )
            del g_ctx, g_moe
        del stack
        torch.cuda.empty_cache()


def _kernel_trace(argv: list[str], keep: str | None) -> None:
    """Re-run under ``rocprofv3 --kernel-trace`` and print the mean dispatch
    duration of every kernel inside each case's timed window."""
    bindir = Path(sys.executable).parent
    out = Path(keep or tempfile.mkdtemp(prefix="flydsl_rdna_moe_"))
    cmd = [str(bindir / "rocprofv3"), "--kernel-trace", "-d", str(out), "-o", "trace"]
    cmd += ["--", sys.executable, __file__, *argv]
    # rocprofv3 can abort while it finalizes; the trace is complete by then.
    env = {**os.environ, _TRACED: "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)
    marks = []
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("MARK "):
            marks.append(json.loads(line[5:]))
        elif line.startswith("RESULT "):
            print(line, end="", flush=True)
    proc.wait()
    if not marks:
        sys.exit("no case finished under rocprofv3")
    db = next(out.rglob("trace_results.db"))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute('select "start", "end", name from kernels').fetchall()
    t_min, t_max = min(r[0] for r in rows), max(r[1] for r in rows)
    clock = next(
        c
        for c in ("mono", "real", "boot")
        if t_min <= marks[0][c] <= t_max and t_min <= marks[-1][c] <= t_max
    )
    for begin, end in zip(marks[::2], marks[1::2]):
        durs: dict[str, list[int]] = {}
        for s, e, n in rows:
            if begin[clock] <= s and e <= end[clock]:
                durs.setdefault(n.split("(")[0], []).append(e - s)
        calls = end["calls"]
        print(f"== {begin['shape']} M={begin['tokens']} ({calls} layer runs)")
        for n, d in sorted(durs.items(), key=lambda kv: -sum(kv[1])):
            print(
                f"   {len(d) / calls:5.2f}/layer "
                f"{statistics.mean(d) / 1e3:9.2f} us mean "
                f"{statistics.median(d) / 1e3:9.2f} us median  {n[:90]}"
            )
    print(f"trace: {db}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    p.add_argument("--tokens", nargs="+", type=int, help="tokens per MoE call")
    p.add_argument("--layers", type=int, help="layers in the stack")
    p.add_argument(
        "--routing", help="torch file with a list of per-layer [tokens, top_k] ids"
    )
    p.add_argument("--no-context", action="store_true", help="MoE kernels only")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="activation dtype of the MoE",
    )
    p.add_argument(
        "--row-path-max-tokens",
        type=int,
        help="override the token count up to which rows are not grouped by expert",
    )
    p.add_argument("--no-check", action="store_true", help="skip the result check")
    p.add_argument(
        "--hip",
        action="store_true",
        help="time the HIP int4 MoE decode kernels instead (up to 5 tokens, symmetric)",
    )
    p.add_argument("--kernel-trace", action="store_true")
    p.add_argument("--trace-dir", help="keep the rocprofv3 output here")
    args = p.parse_args()
    if args.kernel_trace:
        argv = [a for a in sys.argv[1:] if a != "--kernel-trace"]
        os.environ.setdefault("ROCPROFILER_SET_ROCPROFILER_REGISTER_LIBRARY", "0")
        _kernel_trace(argv, args.trace_dir)
    else:
        run(args)


if __name__ == "__main__":
    main()
