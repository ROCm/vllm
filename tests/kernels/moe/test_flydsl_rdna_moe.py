# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

from vllm.platforms.rocm import on_gfx115x  # noqa: E402

if not (on_gfx115x() and importlib.util.find_spec("flydsl") is not None):
    pytest.skip("These tests require gfx115x and FlyDSL.", allow_module_level=True)

from vllm.model_executor.layers.fused_moe.fused_flydsl_rdna_moe import (  # noqa: E402
    BLOCK_M,
    ROW_PATH_MAX_TOKENS,
    flydsl_rdna_moe,
    flydsl_rdna_moe_gemm,
    flydsl_rdna_moe_gemm_reduce,
    gate_up_tile_order,
    pack_w4a16_weights,
    pad_expert_stride,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (  # noqa: E402
    WNA16MoEBackend,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kInt4Static,
    kInt4Static32,
)
from vllm.utils.torch_utils import set_random_seed  # noqa: E402


@pytest.mark.parametrize("num_tokens", [1, 7, 96])
@pytest.mark.parametrize(
    "n_out,n_in,x_row_div",
    [(256, 512, 4), (512, 128, 1)],
    ids=["gate_up", "down"],
)
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("symmetric", [False, True])
def test_flydsl_rdna_moe_gemm(
    num_tokens: int,
    n_out: int,
    n_in: int,
    x_row_div: int,
    group_size: int,
    symmetric: bool,
):
    """The grouped GEMM matches dequantize-then-matmul for int4 with zero
    points and for symmetric int4 packed without them, including blocks
    padded with rows that belong to no token.
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
    if symmetric:
        zp.fill_(8)

    rows = num_tokens * top_k
    x = torch.randn((rows // x_row_div, n_in), device=device, dtype=torch.bfloat16)
    topk_ids = torch.rand((num_tokens, num_experts), device=device).topk(top_k).indices
    sorted_ids, expert_ids, num_valid = moe_align_block_size(
        topk_ids.to(torch.int32), BLOCK_M, num_experts
    )

    out = torch.empty((rows, n_out), device=device, dtype=torch.bfloat16)
    w, coef = pack_w4a16_weights(q, scale, None if symmetric else zp)
    flydsl_rdna_moe_gemm(
        out,
        x,
        w,
        coef,
        sorted_ids,
        expert_ids,
        num_valid,
        x_row_div,
        group_size,
        coef_pairs=not symmetric,
    )

    dequant = (q - zp.repeat_interleave(group_size, dim=2)).float()
    dequant *= scale.float().repeat_interleave(group_size, dim=2)
    x_rows = x.float()[torch.arange(rows, device=device) // x_row_div]
    ref = torch.einsum("rnk,rk->rn", dequant[topk_ids.reshape(-1)], x_rows)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())


@pytest.mark.parametrize("symmetric", [False, True])
def test_flydsl_rdna_moe_gemm_single_token(symmetric: bool):
    """Without routing buffers, row ``r`` runs on ``expert_ids[r]`` and rows
    whose expert id is -1 are left untouched.
    """
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k, n_out, n_in, group_size = 8, 4, 256, 512, 128
    groups = n_in // group_size
    q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
    scale = torch.rand((num_experts, n_out, groups), device=device) * 0.02 + 0.001
    zp = None
    if not symmetric:
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
        coef_pairs=not symmetric,
    )
    ref[1] = 7.0

    out = torch.full_like(ref, 7.0)
    flydsl_rdna_moe_gemm(
        out,
        x,
        w,
        coef,
        None,
        topk_ids.view(-1),
        None,
        top_k,
        group_size,
        coef_pairs=not symmetric,
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


@pytest.mark.parametrize("num_tokens", [1, 3])
@pytest.mark.parametrize("symmetric", [False, True])
def test_flydsl_rdna_moe_gemm_reduce(symmetric: bool, num_tokens: int):
    """The GEMM with the fused top-k sum matches the plain GEMM followed by
    the weighted sum per token, with a padded expert stride, and skips experts
    with id -1.
    """
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k, n_out, n_in, group_size = 16, 8, 256, 512, 128
    groups = n_in // group_size
    q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
    scale = torch.rand((num_experts, n_out, groups), device=device) * 0.02 + 0.001
    zp = None
    if not symmetric:
        zp = torch.randint(0, 16, (num_experts, n_out, groups), device=device)
    w, coef = map(
        pad_expert_stride, pack_w4a16_weights(q, scale.to(torch.bfloat16), zp)
    )
    pairs = not symmetric
    n_rows = num_tokens * top_k
    x = torch.randn((n_rows, n_in), device=device, dtype=torch.bfloat16)
    expert_ids = torch.tensor(
        [5, -1, 0, 3, 15, 9, -1, 11], dtype=torch.int32, device=device
    ).repeat(num_tokens)
    weights = torch.rand((num_tokens, top_k), device=device)

    rows = torch.zeros((n_rows, n_out), device=device, dtype=torch.bfloat16)
    flydsl_rdna_moe_gemm(
        rows, x, w, coef, None, expert_ids, None, 1, group_size, coef_pairs=pairs
    )
    ref = (rows.float().view(num_tokens, top_k, n_out) * weights.unsqueeze(-1)).sum(1)

    out = torch.empty((num_tokens, n_out), device=device, dtype=torch.bfloat16)
    flydsl_rdna_moe_gemm_reduce(
        out, x, w, coef, expert_ids, weights, group_size, coef_pairs=pairs
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())


@pytest.mark.parametrize(
    "num_tokens", [1, ROW_PATH_MAX_TOKENS, ROW_PATH_MAX_TOKENS + 1, 40]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fused_silu", [False, True])
def test_flydsl_rdna_moe(num_tokens: int, dtype: torch.dtype, fused_silu: bool):
    """Both GEMMs, the activation and the top-k sum match the dequantized
    reference for every routing path and for both activation dtypes.
    """
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k, hidden, inter, group_size = 16, 8, 256, 128, 32

    def weights(n_out, n_in, order=None):
        q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
        scale = torch.rand((num_experts, n_out, n_in // group_size), device=device)
        scale = (scale * 0.02 + 0.001).to(torch.bfloat16)
        dequant = (q - 8).float() * scale.float().repeat_interleave(group_size, dim=2)
        if order is not None:
            q, scale = q[:, order], scale[:, order]
        return map(pad_expert_stride, pack_w4a16_weights(q, scale, None)), dequant

    order = gate_up_tile_order(2 * inter, device) if fused_silu else None
    (w1, c1), d1 = weights(2 * inter, hidden, order)
    (w2, c2), d2 = weights(hidden, inter)
    x = torch.randn((num_tokens, hidden), device=device, dtype=dtype)
    topk_ids = torch.rand((num_tokens, num_experts), device=device).topk(top_k).indices
    topk_weights = torch.rand((num_tokens, top_k), device=device)

    def silu_and_mul(out, gate_up):
        gate, up = gate_up.float().chunk(2, dim=-1)
        out.copy_(torch.nn.functional.silu(gate) * up)

    out = torch.empty((num_tokens, hidden), device=device, dtype=dtype)
    workspace = torch.empty(
        num_tokens * top_k * (3 * inter + hidden), device=device, dtype=dtype
    )
    flydsl_rdna_moe(
        out,
        x,
        w1,
        c1,
        w2,
        c2,
        topk_weights,
        topk_ids.to(torch.int32),
        workspace,
        group_size,
        num_experts,
        inter,
        None if fused_silu else silu_and_mul,
        False,
        False,
    )

    gate, up = torch.einsum("tenk,tk->ten", d1[topk_ids], x.float()).chunk(2, dim=-1)
    act = (torch.nn.functional.silu(gate) * up).to(dtype).float()
    down = torch.einsum("tenk,tek->ten", d2[topk_ids], act)
    ref = (down * topk_weights.unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())


@pytest.mark.parametrize("group_size", [32, 128])
def test_rdna_flydsl_is_the_default(monkeypatch: pytest.MonkeyPatch, group_size: int):
    """Without an explicit --moe-backend, symmetric 4-bit layers select the
    FlyDSL kernels; layer widths they do not support and
    VLLM_MOE_RDNA_FLYDSL=0 fall back to another backend.
    """
    from compressed_tensors.quantization import QuantizationArgs

    from tests.kernels.moe.utils import make_dummy_moe_config

    args = QuantizationArgs(
        num_bits=4, type="int", symmetric=True, strategy="group", group_size=group_size
    )
    key = kInt4Static32 if group_size == 32 else kInt4Static

    def select(hidden: int) -> WNA16MoEBackend:
        moe = make_dummy_moe_config(
            num_experts=8, experts_per_token=2, hidden_dim=hidden, intermediate_size=256
        )
        return select_wna16_moe_backend(moe, key, args, False, False)[0]

    rdna_flydsl = WNA16MoEBackend.RDNA_FLYDSL
    assert select(512) == rdna_flydsl
    if group_size == 32:
        # Not a multiple of the 64-column tile.
        assert select(544) != rdna_flydsl
    monkeypatch.setenv("VLLM_MOE_RDNA_FLYDSL", "0")
    assert select(512) != rdna_flydsl
