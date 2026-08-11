# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness and timing harness for the gated delta net prefill op.

Exercises ``chunk_gated_delta_rule`` at delta-net prefill shapes and reports
the whole-op time alongside a per-kernel breakdown of the Triton path.

Examples::

    # baseline breakdown at the production prefill shape
    amd-gpu-lock python benchmarks/kernels/benchmark_gdn_chunk.py

    # sweep sequence lengths
    amd-gpu-lock python benchmarks/kernels/benchmark_gdn_chunk.py \
        --seqlens 941 --seqlens 2048 --seqlens 4096

    # numerics against an fp64 reference
    amd-gpu-lock python benchmarks/kernels/benchmark_gdn_chunk.py --check --exact
"""

from __future__ import annotations

import argparse
import itertools
import time

import torch

from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.fla.ops.wy_fast_doubly_fused import (
    fused_kkt_solve_tril_recompute_w_u_fwd,
)

# Qwen3.5/3.6 35B-A3B delta-net: 32 value heads, 16 key heads, head dim 128.
DEFAULT_HV = 32
DEFAULT_HG = 16
DEFAULT_K = 128
DEFAULT_V = 128


class Inputs:
    """One prefill batch, laid out exactly as the model hands it to the op."""

    def __init__(
        self,
        seqlens: list[int],
        hv: int,
        hg: int,
        k_dim: int,
        v_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int = 0,
        g_scale: float = 0.05,
    ):
        gen = torch.Generator(device=device).manual_seed(seed)
        total = sum(seqlens)
        self.seqlens = seqlens
        self.n_seqs = len(seqlens)

        def randn(*shape, dt=dtype):
            return torch.randn(*shape, generator=gen, device=device, dtype=dt)

        # q and k reach the op l2-normalised, and the conditioning of (I + A)
        # depends on it.
        self.q = torch.nn.functional.normalize(randn(1, total, hg, k_dim), dim=-1)
        self.k = torch.nn.functional.normalize(randn(1, total, hg, k_dim), dim=-1)
        self.v = randn(1, total, hv, v_dim)
        # g is log-space decay: g = -exp(A_log) * softplus(a + dt_bias) <= 0.
        # g_scale sets how fast the state forgets.  Large values crush the
        # within-chunk decay exp(g_i - g_j) to zero, degenerating Tinv to the
        # identity; real gates sit near zero, which is what exercises the
        # triangular inverse and the chunk-to-chunk carry.
        self.g = -g_scale * torch.nn.functional.softplus(
            randn(1, total, hv, dt=torch.float32)
        )
        self.beta = torch.sigmoid(randn(1, total, hv, dt=torch.float32))
        self.h0 = randn(self.n_seqs, hv, v_dim, k_dim, dt=torch.float32)

        self.cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device=device
        )
        self.chunk_indices = prepare_chunk_indices(self.cu_seqlens, FLA_CHUNK_SIZE)
        self.chunk_offsets = prepare_chunk_offsets(self.cu_seqlens, FLA_CHUNK_SIZE)
        self.scale = k_dim**-0.5

    def via_op(self, hip_enabled: bool):
        """Call the public op with the HIP kernel forced on or off."""
        import vllm.envs as envs
        import vllm.model_executor.layers.fla.ops.chunk_rocm as cr

        saved = envs.VLLM_GDN_HIP
        envs.VLLM_GDN_HIP = hip_enabled
        cr._available.cache_clear()
        try:
            return self.baseline()
        finally:
            envs.VLLM_GDN_HIP = saved
            cr._available.cache_clear()

    def baseline(self):
        # Cloned so a kernel that wrongly writes through it cannot poison a
        # later comparison.
        return chunk_gated_delta_rule(
            q=self.q,
            k=self.k,
            v=self.v,
            g=self.g,
            beta=self.beta,
            scale=self.scale,
            initial_state=self.h0.clone(),
            output_final_state=True,
            cu_seqlens=self.cu_seqlens,
            chunk_indices=self.chunk_indices,
            chunk_offsets=self.chunk_offsets,
            use_qk_l2norm_in_kernel=False,
        )

    def reference(self):
        """Token-by-token gated delta rule in fp64.

        The chunked formulation collapses to this at chunk size 1::

            S <- S * exp(g_t)
            u <- beta_t * (v_t - S k_t)
            S <- S + outer(u, k_t)
            o <- scale * S q_t

        Slow, but it is the only thing here that is not itself bf16, so it is
        what decides whether a kernel is better or worse rather than merely
        different.
        """
        q = self.q[0].double()
        k = self.k[0].double()
        v = self.v[0].double()
        g = self.g[0].double()
        beta = self.beta[0].double()
        h = self.h0.double().clone()  # [N, H, V, K]
        hv = v.shape[1]
        rep = hv // k.shape[1]
        o = torch.empty_like(v)
        for i_n in range(self.n_seqs):
            bos = int(self.cu_seqlens[i_n])
            eos = int(self.cu_seqlens[i_n + 1])
            s = h[i_n]  # [H, V, K]
            for t in range(bos, eos):
                kt = k[t].repeat_interleave(rep, dim=0)  # [H, K]
                qt = q[t].repeat_interleave(rep, dim=0)  # [H, K]
                s = s * torch.exp(g[t])[:, None, None]
                u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, kt))
                s = s + u[:, :, None] * kt[:, None, :]
                o[t] = self.scale * torch.einsum("hvk,hk->hv", s, qt)
            h[i_n] = s
        return o.unsqueeze(0), h


_FLUSH: torch.Tensor | None = None
_FLUSH_MIB = 512
_DEFAULT_FLUSH = True


def _flush_buffer() -> torch.Tensor:
    """Scratch big enough to evict the last-level cache between iterations."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(
            _FLUSH_MIB * 1024 * 1024 // 4, dtype=torch.float32, device="cuda"
        )
    return _FLUSH


def board_warmup(fn, seconds: float) -> None:
    """Drive the board to its sustained clock state before measuring.

    This APU gets slower under sustained load, not faster, so measuring straight
    after process start reports boost clocks no real prefill would see and hands
    whichever implementation runs first an advantage.
    """
    if seconds <= 0:
        return
    t0 = time.time()
    while time.time() - t0 < seconds:
        fn()
    torch.accelerator.synchronize()


def _time(fn, warmup: int = 5, iters: int = 20, flush: bool | None = None) -> float:
    """Median GPU time of ``fn`` in milliseconds.

    With ``flush``, the last-level cache is evicted before each iteration.  It
    matters asymmetrically: the Triton path rereads its state from memory and a
    warm cache flatters it, while the single kernel rereads nothing.  In the
    model these caches are contended, so the flushed number is representative.
    """
    if flush is None:
        flush = _DEFAULT_FLUSH
    buf = _flush_buffer() if flush else None
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    times = []
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    for _ in range(iters):
        if buf is not None:
            buf.zero_()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _cpu_time(fn, iters: int = 20) -> float:
    """Mean CPU time per call in ms, with no sync inside the loop.

    Separates host dispatch from device work.  Below roughly 512 tokens the
    Triton path's dispatch cost is comparable to its device time, so timings
    there say more about dispatch than about the kernels.
    """
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - t0) * 1000 / iters
    torch.accelerator.synchronize()
    return elapsed


def breakdown(inp: Inputs) -> None:
    """Time the Triton path as a whole and one kernel at a time."""
    g_cs = chunk_local_cumsum(
        inp.g,
        chunk_size=FLA_CHUNK_SIZE,
        cu_seqlens=inp.cu_seqlens,
        chunk_indices=inp.chunk_indices,
    )
    w, u = fused_kkt_solve_tril_recompute_w_u_fwd(
        k=inp.k,
        v=inp.v,
        beta=inp.beta,
        g_cumsum=g_cs,
        cu_seqlens=inp.cu_seqlens,
        chunk_indices=inp.chunk_indices,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=inp.k,
        w=w,
        u=u,
        g=g_cs,
        initial_state=inp.h0,
        output_final_state=True,
        cu_seqlens=inp.cu_seqlens,
        chunk_indices=inp.chunk_indices,
        chunk_offsets=inp.chunk_offsets,
    )

    stages = {
        "cumsum": lambda: chunk_local_cumsum(
            inp.g,
            chunk_size=FLA_CHUNK_SIZE,
            cu_seqlens=inp.cu_seqlens,
            chunk_indices=inp.chunk_indices,
        ),
        "kkt+solve+wu": lambda: fused_kkt_solve_tril_recompute_w_u_fwd(
            k=inp.k,
            v=inp.v,
            beta=inp.beta,
            g_cumsum=g_cs,
            cu_seqlens=inp.cu_seqlens,
            chunk_indices=inp.chunk_indices,
        ),
        "chunk_delta_h": lambda: chunk_gated_delta_rule_fwd_h(
            k=inp.k,
            w=w,
            u=u,
            g=g_cs,
            initial_state=inp.h0,
            output_final_state=True,
            cu_seqlens=inp.cu_seqlens,
            chunk_indices=inp.chunk_indices,
            chunk_offsets=inp.chunk_offsets,
        ),
        "chunk_o": lambda: chunk_fwd_o(
            q=inp.q,
            k=inp.k,
            v=v_new,
            h=h,
            g=g_cs,
            scale=inp.scale,
            cu_seqlens=inp.cu_seqlens,
            chunk_indices=inp.chunk_indices,
        ),
    }

    # The op dispatches to the HIP kernel when it can, so it has to be switched
    # off to time the Triton path.
    total = _time(lambda: inp.via_op(False))
    print(
        f"  {'triton':<16} {total:8.3f} ms   "
        f"cpu={_cpu_time(lambda: inp.via_op(False)):6.3f} ms"
    )
    summed = 0.0
    for name, fn in stages.items():
        ms = _time(fn)
        summed += ms
        print(f"    {name:<14} {ms:8.3f} ms  ({100 * ms / total:5.1f}%)")
    print(f"    {'(sum)':<14} {summed:8.3f} ms")

    h_bytes = h.numel() * h.element_size()
    print(
        f"  h tensor {tuple(h.shape)} {h.dtype}: "
        f"{h_bytes / 2**20:.1f} MiB written + read per layer"
    )


def hip(inp: Inputs, **kw):
    """The whole op in one HIP kernel."""
    del kw
    from vllm.model_executor.layers.fla.ops.chunk_rocm import chunk_gdn_hip_fwd

    return chunk_gdn_hip_fwd(
        q=inp.q,
        k=inp.k,
        v=inp.v,
        g=inp.g,
        beta=inp.beta,
        scale=inp.scale,
        initial_state=inp.h0,
        cu_seqlens=inp.cu_seqlens,
    )


def _report(name: str, got: torch.Tensor, ref: torch.Tensor) -> float:
    """Print the error of ``got`` against ``ref`` and return its relative RMS."""
    got32, ref32 = got.float(), ref.float()
    adiff = (got32 - ref32).abs()
    peak = ref32.abs().max().item()
    # Scale by the tensor's peak, not per-element: near-zero elements carry no
    # meaningful relative error in bf16 and would otherwise dominate.
    rel_peak = adiff.max().item() / max(peak, 1e-9)
    rms = (
        adiff.pow(2).mean().sqrt() / max(ref32.pow(2).mean().sqrt().item(), 1e-9)
    ).item()
    print(
        f"    {name:<12} max_abs={adiff.max().item():.3e} "
        f"max_abs/peak={rel_peak:.3e} rel_rms={rms:.3e} "
        f"|ref|_peak={peak:.3e}"
    )
    return rms


def check_wired(inp: Inputs) -> None:
    """Compare the public op with the HIP kernel off vs on."""
    o_off, s_off = inp.via_op(False)
    o_on, s_on = inp.via_op(True)
    print("  chunk_gated_delta_rule: FLA_FUSED_CHUNK=0 vs 1")
    _report("out", o_on, o_off)
    _report("state", s_on, s_off)
    took_fused = not torch.equal(o_on, o_off) or not torch.equal(s_on, s_off)
    print(f"    fused path taken: {took_fused}")
    ms_off = _time(lambda: inp.via_op(False))
    ms_on = _time(lambda: inp.via_op(True))
    print(f"    op time  off={ms_off:.3f} ms  on={ms_on:.3f} ms  {ms_off / ms_on:.2f}x")


def check(inp: Inputs, exact: bool = False, **kw) -> None:
    o_base, s_base = inp.via_op(False)
    o_fused, s_fused = hip(inp, **kw)

    if not exact:
        print("  hip vs existing chain (both bf16, so a difference is not an error)")
        _report("out", o_fused, o_base)
        _report("state", s_fused, s_base)
        return

    print("  vs fp64 token-by-token reference")
    o_ref, s_ref = inp.reference()
    results = {
        "chain": (o_base, s_base),
        "hip": (o_fused, s_fused),
    }
    scores = {}
    for name, (o_got, s_got) in results.items():
        scores[name] = max(
            _report(f"{name} out", o_got, o_ref),
            _report(f"{name} state", s_got, s_ref),
        )
    # Both paths are bf16, so the bar is the Triton path's own error, not zero.
    base = scores["chain"]
    bad = []
    for name, score in scores.items():
        if name == "chain":
            continue
        ratio = score / base
        print(
            f"    -> {name}/chain rel_rms = {ratio:.3f}  "
            f"{'ok' if ratio <= 1.25 else 'WORSE THAN CHAIN'}"
        )
        if ratio > 1.25:
            bad.append(name)
    if bad:
        raise SystemExit(f"materially worse than the existing chain: {bad}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seqlens",
        action="append",
        type=str,
        default=None,
        help="comma-separated lengths of one batch; repeat for several batches",
    )
    p.add_argument("--hv", type=int, default=DEFAULT_HV)
    p.add_argument("--hg", type=int, default=DEFAULT_HG)
    p.add_argument("--head-k", type=int, default=DEFAULT_K)
    p.add_argument("--head-v", type=int, default=DEFAULT_V)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--g-scale",
        type=float,
        default=0.05,
        help="log-decay magnitude per token; small means slow forgetting",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--warmup-seconds",
        type=float,
        default=20.0,
        help="drive the board to its sustained clock state before measuring",
    )
    p.add_argument("--no-flush", action="store_true", help="leave caches warm")
    p.add_argument("--check", action="store_true", help="compare numerics only")
    p.add_argument(
        "--exact",
        action="store_true",
        help="score both implementations against an fp64 reference (slow)",
    )
    p.add_argument("--bv", type=int, default=None)
    p.add_argument(
        "--wired",
        action="store_true",
        help="check the chunk.py dispatch, not just the kernel",
    )
    p.add_argument("--num-warps", type=int, default=None)
    p.add_argument("--num-stages", type=int, default=None)
    p.add_argument("--waves-per-eu", type=int, default=None)
    args = p.parse_args()
    overrides = {
        "BV": args.bv,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "waves_per_eu": args.waves_per_eu,
    }
    kw = {k: v for k, v in overrides.items() if v is not None}
    if args.no_flush:
        global _DEFAULT_FLUSH
        _DEFAULT_FLUSH = False

    batches = [[int(x) for x in s.split(",")] for s in (args.seqlens or ["941"])]
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")

    warmed = False
    for seqlens in batches:
        inp = Inputs(
            seqlens,
            args.hv,
            args.hg,
            args.head_k,
            args.head_v,
            dtype,
            device,
            seed=args.seed,
            g_scale=args.g_scale,
        )
        label = "+".join(str(s) for s in seqlens)
        print(
            f"seqlens=[{label}] HV={args.hv} Hg={args.hg} "
            f"K={args.head_k} V={args.head_v} BT={FLA_CHUNK_SIZE} {dtype} "
            f"g_scale={args.g_scale}"
        )
        if not warmed:
            board_warmup(lambda i=inp: i.via_op(False), args.warmup_seconds)
            warmed = True
        if args.wired:
            check_wired(inp)
        elif args.check:
            check(inp, exact=args.exact, **kw)
        else:
            breakdown(inp)
            call = lambda i=inp: hip(i, **kw)
            print(
                f"  {'hip':<16} {_time(call):8.3f} ms   cpu={_cpu_time(call):6.3f} ms"
            )
        print()


if __name__ == "__main__":
    main()
