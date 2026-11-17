# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

from vllm.platforms.rocm import on_gfx11  # noqa: E402

if not (on_gfx11() and importlib.util.find_spec("flydsl") is not None):
    pytest.skip("These tests require gfx11 and FlyDSL.", allow_module_level=True)

from vllm.model_executor.layers.fused_moe.fused_flydsl_rdna_moe import (  # noqa: E402
    BLOCK_M,
    flydsl_rdna_moe_gemm,
    flydsl_rdna_moe_gemm_reduce,
    gate_up_tile_order,
    pack_w4a16_weights,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.utils.torch_utils import set_random_seed  # noqa: E402


@pytest.mark.parametrize("num_tokens", [1, 7, 96])
@pytest.mark.parametrize(
    "n_out,n_in,x_row_div",
    [(256, 512, 4), (512, 128, 1)],
    ids=["gate_up", "down"],
)
@pytest.mark.parametrize("group_size", [32, 128])
def test_flydsl_rdna_moe_gemm(
    num_tokens: int, n_out: int, n_in: int, x_row_div: int, group_size: int
):
    """The grouped GEMM matches dequantize-then-matmul for asymmetric int4,
    including blocks padded with rows that belong to no token.
    """
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k = 8, 4
    groups = n_in // group_size
    q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
    scale = torch.rand((num_experts, n_out, groups), device=device) * 0.02 + 0.001
    scale = scale.to(torch.bfloat16)
    scale[0, 0, 0] = 0
    zp = torch.randint(0, 16, (num_experts, n_out, groups), device=device)

    rows = num_tokens * top_k
    x = torch.randn((rows // x_row_div, n_in), device=device, dtype=torch.bfloat16)
    topk_ids = torch.rand((num_tokens, num_experts), device=device).topk(top_k).indices
    sorted_ids, expert_ids, num_valid = moe_align_block_size(
        topk_ids.to(torch.int32), BLOCK_M, num_experts
    )

    out = torch.empty((rows, n_out), device=device, dtype=torch.bfloat16)
    w, coef = pack_w4a16_weights(q, scale, zp)
    flydsl_rdna_moe_gemm(
        out, x, w, coef, sorted_ids, expert_ids, num_valid, x_row_div, group_size
    )

    dequant = (q - zp.repeat_interleave(group_size, dim=2)).float()
    dequant *= scale.float().repeat_interleave(group_size, dim=2)
    x_rows = x.float()[torch.arange(rows, device=device) // x_row_div]
    ref = torch.einsum("rnk,rk->rn", dequant[topk_ids.reshape(-1)], x_rows)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())


def test_flydsl_rdna_moe_gemm_single_token():
    """Without routing buffers, row ``r`` runs on ``expert_ids[r]`` and rows
    whose expert id is -1 are left untouched.
    """
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k, n_out, n_in, group_size = 8, 4, 256, 512, 128
    groups = n_in // group_size
    q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
    scale = torch.rand((num_experts, n_out, groups), device=device) * 0.02 + 0.001
    zp = torch.randint(0, 16, (num_experts, n_out, groups), device=device)
    w, coef = pack_w4a16_weights(q, scale.to(torch.bfloat16), zp)
    x = torch.randn((1, n_in), device=device, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[5, -1, 0, 3]], dtype=torch.int32, device=device)

    routed = topk_ids.clamp(min=0)
    ref = torch.full((top_k, n_out), 7.0, device=device, dtype=torch.bfloat16)
    flydsl_rdna_moe_gemm(
        ref,
        x,
        w,
        coef,
        *moe_align_block_size(routed, BLOCK_M, num_experts),
        top_k,
        group_size,
    )
    ref[1] = 7.0

    out = torch.full_like(ref, 7.0)
    flydsl_rdna_moe_gemm(
        out, x, w, coef, None, topk_ids.view(-1), None, top_k, group_size
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 20])
def test_flydsl_rdna_moe_gemm_silu_mul(num_tokens: int):
    """With the gate / up columns packed per tile, the GEMM returns
    ``silu(gate) * up`` of the plain GEMM's output.
    """
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k, n_out, n_in, group_size = 8, 4, 256, 512, 128
    groups = n_in // group_size
    q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
    scale = torch.rand((num_experts, n_out, groups), device=device) * 0.02 + 0.001
    scale = scale.to(torch.bfloat16)
    zp = torch.randint(0, 16, (num_experts, n_out, groups), device=device)
    x = torch.randn((num_tokens, n_in), device=device, dtype=torch.bfloat16)
    topk_ids = torch.rand((num_tokens, num_experts), device=device).topk(top_k).indices
    topk_ids = topk_ids.to(torch.int32)
    if num_tokens == 1:
        route = (None, topk_ids.view(-1), None)
    else:
        route = moe_align_block_size(topk_ids, BLOCK_M, num_experts)

    rows = num_tokens * top_k
    gate_up = torch.empty((rows, n_out), device=device, dtype=torch.bfloat16)
    flydsl_rdna_moe_gemm(
        gate_up, x, *pack_w4a16_weights(q, scale, zp), *route, top_k, group_size
    )
    gate, up = gate_up.float().chunk(2, dim=1)
    ref = torch.nn.functional.silu(gate) * up

    order = gate_up_tile_order(n_out, device)
    packed = pack_w4a16_weights(q[:, order], scale[:, order], zp[:, order])
    out = torch.empty((rows, n_out // 2), device=device, dtype=torch.bfloat16)
    flydsl_rdna_moe_gemm(out, x, *packed, *route, top_k, group_size, silu_mul=True)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())


def test_flydsl_rdna_moe_gemm_reduce():
    """The one-token GEMM with the fused top-k sum matches the plain GEMM
    followed by the weighted sum, and skips experts with id -1.
    """
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k, n_out, n_in, group_size = 16, 8, 256, 512, 128
    groups = n_in // group_size
    q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
    scale = torch.rand((num_experts, n_out, groups), device=device) * 0.02 + 0.001
    zp = torch.randint(0, 16, (num_experts, n_out, groups), device=device)
    w, coef = pack_w4a16_weights(q, scale.to(torch.bfloat16), zp)
    x = torch.randn((top_k, n_in), device=device, dtype=torch.bfloat16)
    expert_ids = torch.tensor(
        [5, -1, 0, 3, 15, 9, -1, 11], dtype=torch.int32, device=device
    )
    weights = torch.rand((1, top_k), device=device)

    rows = torch.zeros((top_k, n_out), device=device, dtype=torch.bfloat16)
    flydsl_rdna_moe_gemm(rows, x, w, coef, None, expert_ids, None, 1, group_size)
    ref = (rows.float() * weights.view(-1, 1)).sum(dim=0, keepdim=True)

    out = torch.empty((1, n_out), device=device, dtype=torch.bfloat16)
    flydsl_rdna_moe_gemm_reduce(out, x, w, coef, expert_ids, weights, group_size)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())
