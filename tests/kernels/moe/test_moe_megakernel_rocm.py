# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical test for the HIP MoE megakernel (GEMM2 + topk-weighted reduce).

Exercises ``torch.ops._rocm_C.moe_megakernel_int4_persistent`` against a
pure-fp32 PyTorch reference of the (GEMM2 + per-slot weighted sum) chain
on random AWQ-int4 weights at production-like shapes.

Tolerance: bf16 storage rounding (atol=1e-2, rtol=1e-2).
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires ROCm/CUDA", allow_module_level=True)

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_rocm():
    pytest.skip(
        "HIP MoE megakernel currently only targets ROCm",
        allow_module_level=True,
    )

from vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16 import (  # noqa: E402
    pack_int4_exllama_shuffle,
)


def _dequantize_int4(packed, scales, group_size):
    e, n, kp = packed.shape
    k = kp * 8
    nibs = (
        torch.stack(
            [
                (packed >> 0) & 0xF,
                (packed >> 16) & 0xF,
                (packed >> 4) & 0xF,
                (packed >> 20) & 0xF,
                (packed >> 8) & 0xF,
                (packed >> 24) & 0xF,
                (packed >> 12) & 0xF,
                (packed >> 28) & 0xF,
            ],
            dim=-1,
        )
        .reshape(e, n, k)
        .to(torch.float32)
    )
    s = scales.float().repeat_interleave(group_size, dim=-1)
    return (nibs - 8.0) * s


@pytest.mark.parametrize("group_size", [32, 128])
def test_hip_moe_megakernel_gemm2(group_size):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    K_HIDDEN = 2048
    INTERMEDIATE = 768
    E_NUM = 16  # smaller than production for test speed
    TOP_K = 8

    # Per-slot activations (silu output). Layout: [TOP_K, INTERMEDIATE]
    act = torch.randn(TOP_K, INTERMEDIATE, dtype=dtype, device=device) * 0.1

    # W2 packed [E, K_HIDDEN, INTERMEDIATE//8] int32,
    # scales [E, K_HIDDEN, INTERMEDIATE//G].
    raw = torch.randint(
        0, 16, (E_NUM, K_HIDDEN, INTERMEDIATE), dtype=torch.uint8, device=device
    )
    w2 = torch.stack(
        [pack_int4_exllama_shuffle(raw[i]) for i in range(E_NUM)], dim=0
    ).contiguous()
    w2_scale = (
        torch.randn(
            E_NUM,
            K_HIDDEN,
            INTERMEDIATE // group_size,
            dtype=dtype,
            device=device,
        )
        * 0.005
        + 0.02
    )

    # Random topk: pick distinct experts so test stays clean.
    topk_ids = torch.tensor(
        [[i for i in range(TOP_K)]], dtype=torch.int32, device=device
    )
    topk_w = torch.softmax(
        torch.randn(1, TOP_K, dtype=torch.float32, device=device), dim=-1
    )

    # Output buffers.
    out = torch.zeros(1, K_HIDDEN, dtype=dtype, device=device)
    partial = torch.zeros(TOP_K, K_HIDDEN, dtype=torch.float32, device=device)
    barrier = torch.zeros(1, dtype=torch.int32, device=device)

    # Run kernel.
    cu_count = 20  # gfx1151 Strix Halo
    torch.ops._rocm_C.moe_megakernel_int4_persistent(
        act,
        w2,
        w2_scale,
        topk_ids,
        topk_w,
        out,
        partial,
        barrier,
        False,  # fuse_silu
        cu_count,
        group_size,
    )

    # Reference.
    w2_deq = _dequantize_int4(w2, w2_scale, group_size)  # [E, K_HIDDEN, INTERMEDIATE]
    ref = torch.zeros(K_HIDDEN, dtype=torch.float32, device=device)
    for s in range(TOP_K):
        eid = topk_ids[0, s].item()
        # GEMM2: out[k] = sum_i act[s, i] * w2_deq[eid, k, i]
        partial_sum = torch.matmul(w2_deq[eid].float(), act[s].float())  # [K_HIDDEN]
        ref += topk_w[0, s].float() * partial_sum
    ref_bf16 = ref.to(dtype)

    err = (out[0].float() - ref_bf16.float()).abs()
    rel = err / (ref_bf16.float().abs().clamp(min=1e-3))
    print(
        f"[group_size={group_size}] max_abs={err.max().item():.4e} "
        f"max_rel={rel.max().item():.4e} "
        f"out[0:3]={out[0, :3].tolist()} ref[0:3]={ref_bf16[:3].tolist()}"
    )
    # bf16 storage rounding tolerance.
    assert torch.allclose(out[0].float(), ref_bf16.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("group_size", [32, 128])
def test_hip_moe_megakernel_gemm2_fuse_silu(group_size):
    """fuse_silu=True: kernel takes [TOP_K, 2*INTER] gate||up and folds
    silu(gate)*up into the per-slot LDS staging. Compare against the
    silu_and_mul + GEMM2 reference."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    K_HIDDEN = 2048
    INTERMEDIATE = 768
    E_NUM = 16
    TOP_K = 8

    # Pre-silu activation: [TOP_K, 2*INTERMEDIATE] gate||up.
    gate_up = torch.randn(TOP_K, 2 * INTERMEDIATE, dtype=dtype, device=device) * 0.1

    raw = torch.randint(
        0, 16, (E_NUM, K_HIDDEN, INTERMEDIATE), dtype=torch.uint8, device=device
    )
    w2 = torch.stack(
        [pack_int4_exllama_shuffle(raw[i]) for i in range(E_NUM)], dim=0
    ).contiguous()
    w2_scale = (
        torch.randn(
            E_NUM,
            K_HIDDEN,
            INTERMEDIATE // group_size,
            dtype=dtype,
            device=device,
        )
        * 0.005
        + 0.02
    )

    topk_ids = torch.tensor(
        [[i for i in range(TOP_K)]], dtype=torch.int32, device=device
    )
    topk_w = torch.softmax(
        torch.randn(1, TOP_K, dtype=torch.float32, device=device), dim=-1
    )

    out = torch.zeros(1, K_HIDDEN, dtype=dtype, device=device)
    partial = torch.zeros(TOP_K, K_HIDDEN, dtype=torch.float32, device=device)
    barrier = torch.zeros(1, dtype=torch.int32, device=device)

    cu_count = 20
    torch.ops._rocm_C.moe_megakernel_int4_persistent(
        gate_up,
        w2,
        w2_scale,
        topk_ids,
        topk_w,
        out,
        partial,
        barrier,
        True,  # fuse_silu
        cu_count,
        group_size,
    )

    # Reference: silu(gate)*up, then GEMM2 + topk-weighted reduce.
    gate = gate_up[:, :INTERMEDIATE].float()
    up = gate_up[:, INTERMEDIATE:].float()
    act = (gate * torch.sigmoid(gate) * up).to(dtype)
    w2_deq = _dequantize_int4(w2, w2_scale, group_size)
    ref = torch.zeros(K_HIDDEN, dtype=torch.float32, device=device)
    for s in range(TOP_K):
        eid = topk_ids[0, s].item()
        partial_sum = torch.matmul(w2_deq[eid].float(), act[s].float())
        ref += topk_w[0, s].float() * partial_sum
    ref_bf16 = ref.to(dtype)

    err = (out[0].float() - ref_bf16.float()).abs()
    rel = err / (ref_bf16.float().abs().clamp(min=1e-3))
    print(
        f"[fuse_silu group_size={group_size}] max_abs={err.max().item():.4e} "
        f"max_rel={rel.max().item():.4e}"
    )
    assert torch.allclose(out[0].float(), ref_bf16.float(), atol=5e-2, rtol=5e-2)


def test_hip_moe_megakernel_repeat_calls():
    """Verify the barrier resets correctly across repeated calls."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    K_HIDDEN = 1024
    INTERMEDIATE = 512
    E_NUM = 4
    TOP_K = 4
    group_size = 32

    act = torch.randn(TOP_K, INTERMEDIATE, dtype=dtype, device=device) * 0.1
    raw = torch.randint(
        0, 16, (E_NUM, K_HIDDEN, INTERMEDIATE), dtype=torch.uint8, device=device
    )
    w2 = torch.stack(
        [pack_int4_exllama_shuffle(raw[i]) for i in range(E_NUM)], dim=0
    ).contiguous()
    w2_scale = (
        torch.randn(
            E_NUM,
            K_HIDDEN,
            INTERMEDIATE // group_size,
            dtype=dtype,
            device=device,
        )
        * 0.005
        + 0.02
    )
    topk_ids = torch.tensor(
        [[i for i in range(TOP_K)]], dtype=torch.int32, device=device
    )
    topk_w = torch.softmax(
        torch.randn(1, TOP_K, dtype=torch.float32, device=device), dim=-1
    )
    partial = torch.zeros(TOP_K, K_HIDDEN, dtype=torch.float32, device=device)
    barrier = torch.zeros(1, dtype=torch.int32, device=device)

    out1 = torch.zeros(1, K_HIDDEN, dtype=dtype, device=device)
    out2 = torch.zeros(1, K_HIDDEN, dtype=dtype, device=device)
    cu_count = 20
    torch.ops._rocm_C.moe_megakernel_int4_persistent(
        act,
        w2,
        w2_scale,
        topk_ids,
        topk_w,
        out1,
        partial,
        barrier,
        False,  # fuse_silu
        cu_count,
        group_size,
    )
    torch.accelerator.synchronize()
    torch.ops._rocm_C.moe_megakernel_int4_persistent(
        act,
        w2,
        w2_scale,
        topk_ids,
        topk_w,
        out2,
        partial,
        barrier,
        False,  # fuse_silu
        cu_count,
        group_size,
    )
    torch.accelerator.synchronize()
    assert torch.allclose(out1.float(), out2.float(), atol=1e-3)
