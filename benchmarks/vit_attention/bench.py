#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for ViT attention backends at a parametrizable shape,
default = Gemma3-4B SigLIP ViT.

Calls `triton.testing.do_bench`, which clears the L2 cache before every
measurement iteration to approximate the kernel's cold-cache behavior inside
a real transformer block (where surrounding qkv-proj / MLP / norm work
evicts q/k/v/output between layer calls).

Per-call shape (one SigLIP layer of Gemma3): B=1, S=4096, num_q_heads=
num_kv_heads=16, head_dim=72, dtype=bf16, is_causal=False. 27 layers / image.

--backend selects the attention implementation, matching the values accepted
by --mm-encoder-attn-backend in vllm serve (TRITON_ATTN, TORCH_SDPA,
FLASH_ATTN, ROCM_AITER_FA).  The default "all" runs every backend in sequence
and emits one JSON line per backend.

Triton-specific tuning knobs (--bm/--bn/--nw/--ns/--we) are passed directly
to the kernel launch and are ignored for other backends.

Operand layout: a real ViT fuses the QKV projection, so V is a *non-contiguous*
view into a packed tensor (token stride = mult*H*D, mult=3 for a standard fused
QKV) while Q/K are made contiguous by the serving wrappers. --v-stride-mult
controls this (default 3 = model-matching fused QKV; 1 = fully contiguous).

--no-flush measures with a warm cache (no L2 clear between iterations) instead
of do_bench's default cold-cache behavior.

Output: one JSON line per backend to stdout.
"""

import argparse
import json
import os
import site
import sys

# Without amd_smi importable, vllm.platforms falls back to UnspecifiedPlatform
# and get_block_size returns the wrong default for ROCm (BLOCK_M=64 instead
# of the cuda_alike+capability(80) branch's 128). The TheRock ROCm SDK ships
# the amd_smi Python package under a non-standard share/ path; add it to
# sys.path if present, BEFORE importing anything from vllm.
for _site in site.getsitepackages():
    _amdsmi = os.path.join(_site, "_rocm_sdk_core", "share", "amd_smi")
    if os.path.isdir(_amdsmi) and _amdsmi not in sys.path:
        sys.path.insert(0, _amdsmi)
        break

import torch  # noqa: E402

from vllm.triton_utils import triton  # noqa: E402
from vllm.utils.math_utils import RCP_LN2  # noqa: E402
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402
from vllm.v1.attention.ops.triton_prefill_attention import (  # noqa: E402
    _fwd_kernel,
    _split_head_dim,
    get_block_n,
    get_block_size,
    get_num_warps,
)

_SUPPORTED_BACKENDS = [
    "TRITON_ATTN",
    "TORCH_SDPA",
    "FLASH_ATTN",
    "ROCM_AITER_FA",
]


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=72)

    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument(
        "--backend",
        default="all",
        choices=_SUPPORTED_BACKENDS + ["all"],
        help=(
            "Attention backend (same values as --mm-encoder-attn-backend), "
            "or 'all' to run every backend in sequence (default)"
        ),
    )
    # TRITON_ATTN-only tuning knobs
    p.add_argument("--bm", type=int, default=None, help="BLOCK_M (TRITON_ATTN only)")
    p.add_argument("--bn", type=int, default=None, help="BLOCK_N (TRITON_ATTN only)")
    p.add_argument("--nw", type=int, default=None, help="num_warps (TRITON_ATTN only)")
    p.add_argument("--ns", type=int, default=1, help="num_stages (TRITON_ATTN only)")
    p.add_argument(
        "--we", type=int, default=None, help="waves_per_eu (TRITON_ATTN only)"
    )
    p.add_argument(
        "--v-stride-mult",
        type=int,
        default=3,
        help=(
            "V token-stride multiplier: V is a non-contiguous slice of a packed "
            "(B,S,mult,H,D) tensor, so its token stride is mult*H*D. Default 3 "
            "matches a fused QKV projection (the real model layout); 1 = the "
            "old fully-contiguous layout."
        ),
    )
    p.add_argument(
        "--no-flush",
        action="store_true",
        help=(
            "Measure with a warm cache (no L2 flush between iterations) instead "
            "of do_bench's default cold-cache flush."
        ),
    )
    p.add_argument("--warmup-ms", type=int, default=200)
    p.add_argument("--rep-ms", type=int, default=600)
    return p.parse_args()


def _do_bench_no_flush(fn, warmup_ms: int, rep_ms: int) -> list[float]:
    """do_bench variant that does NOT clear the cache between iterations.

    Mirrors triton.testing.do_bench's event-timed loop (5-call cost estimate ->
    warmup -> timed reps) but omits the L2 flush, so q/k/v stay resident (warm).
    Returns per-call times in milliseconds.
    """
    di = triton.runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    start = di.Event(enable_timing=True)
    end = di.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    di.synchronize()
    estimate_ms = start.elapsed_time(end) / 5
    n_warmup = max(1, int(warmup_ms / estimate_ms))
    n_repeat = max(1, int(rep_ms / estimate_ms))

    for _ in range(n_warmup):
        fn()
    starts = [di.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [di.Event(enable_timing=True) for _ in range(n_repeat)]
    di.synchronize()
    for i in range(n_repeat):
        starts[i].record()
        fn()
        ends[i].record()
    di.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _bench_one(
    backend_name: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    q4: torch.Tensor,
    k4: torch.Tensor,
    v4: torch.Tensor,
    cu: torch.Tensor,
    max_seqlen: torch.Tensor,
) -> None:
    """Benchmark one backend and write a JSON result line to stdout."""
    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim
    backend = AttentionBackendEnum[backend_name]
    scale = 1.0 / (D**0.5)

    if backend == AttentionBackendEnum.TRITON_ATTN:
        # Keep direct kernel launch so tuning knobs are honoured.
        BLOCK_M = args.bm if args.bm is not None else get_block_size(dtype, head_dim=D)
        BLOCK_N = args.bn if args.bn is not None else get_block_n(dtype, head_dim=D)
        num_warps = args.nw if args.nw is not None else get_num_warps(D)
        BLOCK_DMODEL, BLOCK_DMODEL_TAIL = _split_head_dim(D)
        sm_scale = scale * RCP_LN2
        grid = (B, H, triton.cdiv(S, BLOCK_M))
        kv_group_num = H // H

        # Flat (B*S, H, D) layout expected by _fwd_kernel
        q = q4.view(B * S, H, D)
        k = k4.view(B * S, H, D)
        v = v4.view(B * S, H, D)
        o = torch.empty_like(q)
        seqlen = cu[1:] - cu[:-1]
        head_stride_aligned_8 = (
            q.stride(1) % 8 == 0
            and k.stride(1) % 8 == 0
            and v.stride(1) % 8 == 0
            and o.stride(1) % 8 == 0
        )

        extra_kwargs: dict = {}
        if args.we is not None:
            extra_kwargs["waves_per_eu"] = args.we

        def _fn():
            _fwd_kernel[grid](
                q,
                k,
                v,
                sm_scale,
                cu[:-1],
                seqlen,
                o,
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                o.stride(0),
                o.stride(1),
                kv_group_num=kv_group_num,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DMODEL_TAIL=BLOCK_DMODEL_TAIL,
                BLOCK_N=BLOCK_N,
                IS_CAUSAL=False,
                SLIDING_WINDOW_Q=0,
                SLIDING_WINDOW_K=0,
                num_warps=num_warps,
                num_stages=args.ns,
                Lk=D,
                HEAD_STRIDE_ALIGNED_8=head_stride_aligned_8,
                **extra_kwargs,
            )

    elif backend in (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
    ):
        from vllm.v1.attention.backends.fa_utils import (
            get_flash_attn_version,  # noqa: E402
        )
        from vllm.v1.attention.ops.vit_attn_wrappers import (
            vit_flash_attn_wrapper,  # noqa: E402
        )

        fa_version = get_flash_attn_version(head_size=D)
        is_rocm_aiter = backend == AttentionBackendEnum.ROCM_AITER_FA

        def _fn():
            vit_flash_attn_wrapper(
                q=q4,
                k=k4,
                v=v4,
                batch_size=B,
                is_rocm_aiter=is_rocm_aiter,
                fa_version=fa_version,
                scale=scale,
                cu_seqlens=cu,
                max_seqlen=max_seqlen,
            )

    elif backend == AttentionBackendEnum.TORCH_SDPA:
        from vllm.v1.attention.ops.vit_attn_wrappers import (
            vit_torch_sdpa_wrapper,  # noqa: E402
        )

        def _fn():
            vit_torch_sdpa_wrapper(
                q=q4,
                k=k4,
                v=v4,
                scale=scale,
                cu_seqlens=cu,
            )

    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

    if args.no_flush:
        # Warm cache: no L2 flush between iterations.
        all_times = _do_bench_no_flush(_fn, args.warmup_ms, args.rep_ms)
    else:
        # do_bench clears the L2 cache before each measurement iteration.
        all_times = triton.testing.do_bench(
            _fn,
            warmup=args.warmup_ms,
            rep=args.rep_ms,
            return_mode="all",
        )
    all_times = sorted(all_times)
    n = len(all_times)
    mean = sum(all_times) / n
    median = all_times[n // 2]
    p10 = all_times[max(0, int(n * 0.10))]
    p90 = all_times[min(n - 1, int(n * 0.90))]
    fastest = all_times[0]

    config: dict = {
        "B": B,
        "S": S,
        "H": H,
        "D": D,
        "dtype": args.dtype,
        "backend": backend_name,
        "v_stride_mult": args.v_stride_mult,
        "v_token_stride": int(v4.stride(-3)),
        "flush": not args.no_flush,
    }
    if backend == AttentionBackendEnum.TRITON_ATTN:
        config.update(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_DMODEL": BLOCK_DMODEL,
                "BLOCK_DMODEL_TAIL": BLOCK_DMODEL_TAIL,
                "BLOCK_N": BLOCK_N,
                "num_warps": num_warps,
                "num_stages": args.ns,
                "waves_per_eu": args.we,
            }
        )

    result = {
        "per_call_ms_mean": mean,
        "per_call_ms_median": median,
        "per_call_ms_min": fastest,
        "per_call_ms_p10": p10,
        "per_call_ms_p90": p90,
        "samples": n,
        "config": config,
    }
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse()
    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    torch.manual_seed(0)

    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim

    # Wrappers (FLASH_ATTN, ROCM_AITER_FA, TRITON_ATTN) expect (B, S, H, D).
    # TORCH_SDPA expects the same via vit_torch_sdpa_wrapper.
    #
    # Q/K are contiguous (the serving wrappers make them so). V models the fused
    # QKV projection: it is a non-contiguous slice of a packed (B,S,mult,H,D)
    # tensor, giving a token stride of mult*H*D (mult=3 => the real fused-QKV
    # layout). --v-stride-mult 1 restores the old fully-contiguous V.
    q4 = torch.randn(B, S, H, D, dtype=dtype, device=device)
    k4 = torch.randn(B, S, H, D, dtype=dtype, device=device)
    mult = args.v_stride_mult
    if mult <= 1:
        v4 = torch.randn(B, S, H, D, dtype=dtype, device=device)
    else:
        # Packed buffer; take index 0 along the fused dim -> V keeps a token
        # stride of mult*H*D. The (B,S) dims stay mergeable, so the TRITON path's
        # .view(B*S, H, D) still works while preserving the non-contiguous stride.
        v_packed = torch.randn(B, S, mult, H, D, dtype=dtype, device=device)
        v4 = v_packed[:, :, 0]

    # cu_seqlens / max_seqlen used by FA and Triton backends
    cu = torch.tensor([i * S for i in range(B + 1)], dtype=torch.int32, device=device)
    max_seqlen = torch.tensor(S, dtype=torch.int32)

    backends = _SUPPORTED_BACKENDS if args.backend == "all" else [args.backend]
    for backend_name in backends:
        try:
            _bench_one(backend_name, args, dtype, q4, k4, v4, cu, max_seqlen)
        except Exception as exc:
            if args.backend == "all":
                sys.stderr.write(f"[skip] {backend_name}: {exc}\n")
            else:
                raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
