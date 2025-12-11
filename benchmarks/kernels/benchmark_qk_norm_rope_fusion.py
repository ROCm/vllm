# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def apply_qk_norm_rope_unfused(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: RotaryEmbedding,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = q_norm.forward_native(q_by_head)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = k_norm.forward_native(k_by_head)
    k = k_by_head.view(k.shape)

    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)


def apply_qk_norm_rope_vllm_cuda(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    eps: float,
    is_neox: bool,
) -> torch.Tensor:
    torch.ops._C.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_kv,
        num_heads_kv,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        positions.view(-1),
    )
    return qkv


def apply_qk_norm_rope_aiter(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    eps: float,
    is_neox: bool,
) -> torch.Tensor:
    rocm_aiter_ops.fused_qk_norm_rope(
        qkv=qkv,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_kv,
        num_heads_v=num_heads_kv,
        head_dim=head_dim,
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        position_ids=positions.view(-1),
    )
    return qkv


def calculate_diff(num_tokens, num_heads, num_kv_heads, head_dim, dtype, is_neox, eps):
    device = "cuda"
    total_dim = (num_heads + 2 * num_kv_heads) * head_dim

    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)
    q_weight = q_norm.weight.data
    k_weight = k_norm.weight.data

    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)

    # Unfused reference
    output_unfused = apply_qk_norm_rope_unfused(
        qkv_base.clone(),
        positions,
        q_norm,
        k_norm,
        rope,
        num_heads,
        num_kv_heads,
        head_dim,
    )

    # vLLM CUDA kernel
    if hasattr(torch.ops._C, "fused_qk_norm_rope"):
        qkv_vllm = qkv_base.clone()
        output_vllm = apply_qk_norm_rope_vllm_cuda(
            qkv_vllm,
            positions,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            is_neox,
        )
        vllm_matches = torch.allclose(output_unfused, output_vllm, atol=5e-2, rtol=1e-2)
        print(f"vLLM CUDA kernel: {'Matches' if vllm_matches else 'Differs'}")
    else:
        print("vLLM CUDA kernel: Not available")

    # AITER kernel
    if (
        current_platform.is_rocm()
        and rocm_aiter_ops.is_enabled()
        and rocm_aiter_ops.is_fused_qk_norm_rope_enabled()
    ):
        qkv_aiter = qkv_base.clone()
        output_aiter = apply_qk_norm_rope_aiter(
            qkv_aiter,
            positions,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            is_neox,
        )
        aiter_matches = torch.allclose(
            output_unfused, output_aiter, atol=5e-2, rtol=1e-2
        )
        print(f"AITER kernel: {'Matches' if aiter_matches else 'Differs'}")
    else:
        print(
            "AITER kernel: Not available "
            "(requires ROCm and VLLM_ROCM_USE_AITER_FUSED_QK_NORM_ROPE=1)"
        )


num_tokens_range = [64, 256, 1024, 4096]
num_heads_range = [32, 64]
num_kv_heads_range = [8, 16]
head_dim_range = [64, 128]
configs = list(
    itertools.product(
        num_tokens_range, num_heads_range, num_kv_heads_range, head_dim_range
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_heads", "num_kv_heads", "head_dim"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["unfused", "vllm_cuda", "aiter"],
        line_names=["Unfused", "vLLM CUDA", "AITER"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="us",
        plot_name="qk-norm-rope-fusion-perf",
        args={},
    )
)
def benchmark(num_tokens, num_heads, num_kv_heads, head_dim, provider):
    dtype = torch.bfloat16
    device = "cuda"
    eps = 1e-6
    is_neox = True

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_weight = q_norm.weight.data
    k_weight = k_norm.weight.data

    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "unfused":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: apply_qk_norm_rope_unfused(
                qkv.clone(),
                positions,
                q_norm,
                k_norm,
                rope,
                num_heads,
                num_kv_heads,
                head_dim,
            ),
            quantiles=quantiles,
        )
    elif provider == "vllm_cuda":
        if not hasattr(torch.ops._C, "fused_qk_norm_rope"):
            return float("nan"), float("nan"), float("nan")
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: apply_qk_norm_rope_vllm_cuda(
                qkv.clone(),
                positions,
                q_weight,
                k_weight,
                rope.cos_sin_cache,
                num_heads,
                num_kv_heads,
                head_dim,
                eps,
                is_neox,
            ),
            quantiles=quantiles,
        )
    elif provider == "aiter":
        if not (
            current_platform.is_rocm()
            and rocm_aiter_ops.is_enabled()
            and rocm_aiter_ops.is_fused_qk_norm_rope_enabled()
        ):
            return float("nan"), float("nan"), float("nan")
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: apply_qk_norm_rope_aiter(
                qkv.clone(),
                positions,
                q_weight,
                k_weight,
                rope.cos_sin_cache,
                num_heads,
                num_kv_heads,
                head_dim,
                eps,
                is_neox,
            ),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(
        description="Benchmark QK norm + RoPE fusion kernels"
    )
    parser.add_argument("--num-tokens", type=int, default=256, help="Number of tokens")
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of query heads"
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=8, help="Number of key/value heads"
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type",
    )
    parser.add_argument("--is-neox", action="store_true", help="Use Neox-style RoPE")
    parser.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/qk_norm_rope/",
        help="Path to save benchmark results",
    )

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print("=" * 80)
    print("Correctness Test")
    print("=" * 80)
    calculate_diff(
        args.num_tokens,
        args.num_heads,
        args.num_kv_heads,
        args.head_dim,
        dtype,
        args.is_neox,
        args.eps,
    )

    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)
    benchmark.run(print_data=True, save_path=args.save_path)
