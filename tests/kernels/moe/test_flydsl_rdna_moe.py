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
    GROUP_SIZE,
    flydsl_rdna_moe_gemm,
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
def test_flydsl_rdna_moe_gemm(num_tokens: int, n_out: int, n_in: int, x_row_div: int):
    """The grouped GEMM matches dequantize-then-matmul for asymmetric int4,
    including blocks padded with rows that belong to no token.
    """
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k = 8, 4
    groups = n_in // GROUP_SIZE
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
    flydsl_rdna_moe_gemm(out, x, w, coef, sorted_ids, expert_ids, num_valid, x_row_div)

    dequant = (q - zp.repeat_interleave(GROUP_SIZE, dim=2)).float()
    dequant *= scale.float().repeat_interleave(GROUP_SIZE, dim=2)
    x_rows = x.float()[torch.arange(rows, device=device) // x_row_div]
    ref = torch.einsum("rnk,rk->rn", dequant[topk_ids.reshape(-1)], x_rows)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())
