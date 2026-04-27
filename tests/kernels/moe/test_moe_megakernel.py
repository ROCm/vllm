# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical tests for the single-token MoE megakernel.

Compares the megakernel against a pure-fp32 PyTorch reference of the
chain (router GEMV -> topk softmax -> per-expert dequant W1 / silu_mul
/ dequant W2 -> weighted sum).  The megakernel and reference both
accumulate in fp32, so they agree to bf16 storage rounding.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires ROCm/CUDA", allow_module_level=True)

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_rocm():
    pytest.skip("MoE megakernel currently only targets ROCm", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16 import (  # noqa: E402
    pack_int4_exllama_shuffle,
)
from vllm.model_executor.layers.fused_moe.moe_megakernel import (  # noqa: E402
    moe_megakernel,
    moe_megakernel_post_router,
)

HIDDEN = 2048
INTERMEDIATE = 768
NUM_EXPERTS = 128
TOP_K = 8
GROUP_SIZE = 32
DTYPE = torch.bfloat16
ZP_BIAS = 8


def _make_int4_weights(e, n, k, group_size, device, scale_mag=0.005):
    raw = torch.randint(0, 16, (e, n, k), dtype=torch.uint8, device=device)
    packed = torch.stack(
        [pack_int4_exllama_shuffle(raw[i]) for i in range(e)], dim=0
    ).contiguous()
    scales = (
        torch.randn(e, n, k // group_size, dtype=DTYPE, device=device) * scale_mag
        + 0.02
    )
    return packed, scales


def _dequantize_int4(packed, scales, group_size):
    """[E, N, K//8] int32 + [E, N, K//G] -> [E, N, K] fp32."""
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


def _reference_pure_fp32(hidden, gate_w, w1, w1_scale, w2, w2_scale, top_k, group_size):
    """Pure fp32 reference: dequant -> matmul -> silu_mul -> matmul -> reduce."""
    deq1 = _dequantize_int4(w1, w1_scale, group_size)  # [E, N1, K_hidden]
    deq2 = _dequantize_int4(w2, w2_scale, group_size)  # [E, K_hidden, N2]
    h = hidden[0].float()
    K_h = h.shape[0]
    N1 = deq1.shape[1]
    N2 = N1 // 2

    # Router
    router = (h @ gate_w.float().t()).unsqueeze(0)  # [1, E]
    topk_w = torch.empty(1, top_k, dtype=torch.float32, device=hidden.device)
    topk_ids = torch.empty(1, top_k, dtype=torch.int32, device=hidden.device)
    tei = torch.empty(1, top_k, dtype=torch.int32, device=hidden.device)
    ops.topk_softmax(topk_w, topk_ids, tei, router, renormalize=True)

    out = torch.zeros(1, K_h, dtype=torch.float32, device=hidden.device)
    for i in range(top_k):
        e = int(topk_ids[0, i].item())
        wi = topk_w[0, i].item()
        gate = h @ deq1[e, :N2].t()  # [N2]
        up = h @ deq1[e, N2:].t()  # [N2]
        silu = gate / (1.0 + torch.exp(-gate))  # silu
        interim = silu * up  # [N2]
        expert_out = interim @ deq2[e].t()  # [K_h]
        out[0] += wi * expert_out
    return out


def test_megakernel_matches_pure_fp32_reference():
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    hidden = torch.randn(1, HIDDEN, dtype=DTYPE, device=device) * 0.5
    gate_w = torch.randn(NUM_EXPERTS, HIDDEN, dtype=DTYPE, device=device) * 0.02

    w1, w1_scale = _make_int4_weights(
        NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, GROUP_SIZE, device
    )
    w2, w2_scale = _make_int4_weights(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, GROUP_SIZE, device
    )

    out = torch.empty(1, HIDDEN, dtype=DTYPE, device=device)
    moe_megakernel(
        hidden,
        gate_w,
        w1,
        w1_scale,
        w2,
        w2_scale,
        out,
        top_k=TOP_K,
        group_size=GROUP_SIZE,
        zp_bias=ZP_BIAS,
    )

    ref = _reference_pure_fp32(
        hidden, gate_w, w1, w1_scale, w2, w2_scale, TOP_K, GROUP_SIZE
    )

    diff = (out.float() - ref).abs()
    rel = diff / (ref.abs().max() + 1e-9)
    print(f"\nMax abs diff: {diff.max().item():.5f}")
    print(f"Mean abs diff: {diff.mean().item():.5f}")
    print(f"Ref max abs: {ref.abs().max().item():.5f}")
    print(f"Ref RMS: {ref.pow(2).mean().sqrt().item():.5f}")
    print(f"Max rel diff (norm by max): {rel.max().item():.5f}")

    # bf16 store at end -> ~1/256 rel precision per output element.
    # Fused kernel matches pure-fp32 reference to bf16 storage rounding.
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)


def test_megakernel_post_router_matches_megakernel():
    """post_router variant should produce identical output when fed the same
    topk routing as the full megakernel computes internally."""
    torch.manual_seed(123)
    device = torch.device("cuda:0")

    hidden = torch.randn(1, HIDDEN, dtype=DTYPE, device=device) * 0.5
    gate_w = torch.randn(NUM_EXPERTS, HIDDEN, dtype=DTYPE, device=device) * 0.02
    w1, w1_scale = _make_int4_weights(
        NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, GROUP_SIZE, device
    )
    w2, w2_scale = _make_int4_weights(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, GROUP_SIZE, device
    )

    # Pre-route with the standard topk_softmax op
    router = hidden.float() @ gate_w.float().t()
    topk_w = torch.empty(1, TOP_K, dtype=torch.float32, device=device)
    topk_ids = torch.empty(1, TOP_K, dtype=torch.int32, device=device)
    tei = torch.empty(1, TOP_K, dtype=torch.int32, device=device)
    ops.topk_softmax(topk_w, topk_ids, tei, router, renormalize=True)

    out_post = torch.empty(1, HIDDEN, dtype=DTYPE, device=device)
    moe_megakernel_post_router(
        hidden=hidden,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        topk_ids=topk_ids,
        topk_weights=topk_w,
        output=out_post,
        top_k=TOP_K,
        group_size=GROUP_SIZE,
        zp_bias=ZP_BIAS,
    )

    out_full = torch.empty(1, HIDDEN, dtype=DTYPE, device=device)
    moe_megakernel(
        hidden,
        gate_w,
        w1,
        w1_scale,
        w2,
        w2_scale,
        out_full,
        top_k=TOP_K,
        group_size=GROUP_SIZE,
        zp_bias=ZP_BIAS,
    )

    diff = (out_post.float() - out_full.float()).abs()
    print(f"\npost-router vs full-megakernel max abs diff: {diff.max().item():.5f}")
    # Should be bit-identical: same kernels, same topk inputs.
    torch.testing.assert_close(out_post, out_full)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
