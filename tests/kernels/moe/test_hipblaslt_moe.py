# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""hipBLASLt W4A16 MoE expert GEMM against dequantize-then-matmul.

Covers whichever path hipBLASLt takes: one grouped GEMM, or one dense GEMM per
expert where it has no grouped solution.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.kernels.linear.mixed_precision.hipblaslt_w4a16 import (  # noqa: E402
    is_built,
)

if not is_built():
    pytest.skip(
        "_rocm_C was built without the hipBLASLt w4a16 API", allow_module_level=True
    )

from vllm.model_executor.layers.fused_moe.experts.hipblaslt_moe import (  # noqa: E402
    _Groups,
    pack_moe_scale_buffers,
    pad_moe_weight_rows,
)
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

GROUP_SIZE = 32


def _pack_nibbles(q: torch.Tensor, dim: int) -> torch.Tensor:
    """Two nibbles per byte along `dim`, even index in the low nibble."""
    lo, hi = q.unflatten(dim, (-1, 2)).unbind(dim + 1)
    return (lo | (hi << 4)).to(torch.uint8)


@pytest.mark.parametrize("grouped", [False, True], ids=["host", "device"])
@pytest.mark.parametrize("has_zp", [True, False], ids=["asym", "sym"])
@pytest.mark.parametrize("num_tokens", [1, 7, 96])
@pytest.mark.parametrize(
    "n_out,n_in", [(1024, 2048), (2048, 512)], ids=["gate_up", "down"]
)
def test_hipblaslt_w4a16_moe_gemm(
    grouped: bool,
    has_zp: bool,
    num_tokens: int,
    n_out: int,
    n_in: int,
):
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k = 16, 4
    groups = n_in // GROUP_SIZE
    q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
    scale = torch.rand((num_experts, n_out, groups), device=device) * 0.02 + 0.001
    scale = scale.to(torch.bfloat16)
    if has_zp:
        zp = torch.randint(0, 16, (num_experts, n_out, groups), device=device)
        buf = pack_moe_scale_buffers(scale, _pack_nibbles(zp, 1))
    else:
        zp = torch.full_like(q[..., :groups], 8)  # implicit zero-point of 8
        buf = pack_moe_scale_buffers(scale, None)

    topk_ids = torch.rand((num_tokens, num_experts), device=device).topk(top_k).indices
    flat = topk_ids.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts).cpu()
    x = torch.randn((num_tokens, n_in), device=device, dtype=torch.bfloat16)
    x_sorted = x[order // top_k]
    out = torch.empty((x_sorted.size(0), n_out), device=device, dtype=torch.bfloat16)
    w = pad_moe_weight_rows(_pack_nibbles(q, 2))
    assert (w.stride(1) * w.element_size()) % 1024 != 0
    w = w.view(torch.int8)
    if grouped:
        rows = flat.numel()
        g = _Groups.make(flat[order], flat, num_experts, num_tokens, rows)
        if not torch.ops._rocm_C.hipblaslt_w4a16_moe_grouped_gemm(
            out,
            x_sorted,
            w,
            buf,
            g.expert,
            g.count,
            g.offset,
            g.max_n,
            GROUP_SIZE,
            has_zp,
        ):
            pytest.skip("hipBLASLt has no grouped W4A16 solution")
    else:
        torch.ops._rocm_C.hipblaslt_w4a16_moe_gemm(
            out, x_sorted, w, buf, counts, GROUP_SIZE, has_zp
        )

    dequant = (q - zp.repeat_interleave(GROUP_SIZE, dim=2)).float()
    dequant *= scale.float().repeat_interleave(GROUP_SIZE, dim=2)
    ref = torch.einsum("rnk,rk->rn", dequant[flat[order]], x_sorted.float())
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())


@pytest.mark.parametrize("num_tokens", [1, 2, 8])
@pytest.mark.parametrize(
    "n_out,n_in", [(1024, 2048), (2048, 512)], ids=["gate_up", "down"]
)
def test_hipblaslt_w4a16_moe_slot_order(num_tokens: int, n_out: int, n_in: int):
    """Decode path: one group per routed slot in slot order (no sort), with
    duplicate experts across tokens and one empty slot (id -1). gate_up-style
    calls read token row slot // top_k (x_row_div); empty slots give zeros."""
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k = 16, 4
    groups = n_in // GROUP_SIZE
    q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
    scale = torch.rand((num_experts, n_out, groups), device=device) * 0.02 + 0.001
    scale = scale.to(torch.bfloat16)
    zp = torch.randint(0, 16, (num_experts, n_out, groups), device=device)
    w = pad_moe_weight_rows(_pack_nibbles(q, 2)).view(torch.int8)
    buf = pack_moe_scale_buffers(scale, _pack_nibbles(zp, 1))

    ids = torch.rand((num_tokens, num_experts), device=device).topk(top_k).indices
    flat = ids.flatten().to(torch.int32)
    flat[-1] = -1
    rows = flat.numel()
    x = torch.randn((num_tokens, n_in), device=device, dtype=torch.bfloat16)
    x_rows = x.repeat_interleave(top_k, dim=0)
    out = torch.full((rows, n_out), float("nan"), device=device, dtype=torch.bfloat16)
    ones = torch.ones(rows, dtype=torch.int32, device=device)
    offsets = torch.arange(rows, dtype=torch.int32, device=device)
    if not torch.ops._rocm_C.hipblaslt_w4a16_moe_grouped_gemm(
        out, x, w, buf, flat, ones, offsets, 1, GROUP_SIZE, True, False, top_k
    ):
        pytest.skip("hipBLASLt has no grouped W4A16 solution")

    dequant = (q - zp.repeat_interleave(GROUP_SIZE, dim=2)).float()
    dequant *= scale.float().repeat_interleave(GROUP_SIZE, dim=2)
    valid = flat >= 0
    ref = torch.einsum("rnk,rk->rn", dequant[flat[valid].long()], x_rows[valid].float())
    torch.testing.assert_close(
        out[valid].float(), ref, rtol=2e-2, atol=2e-2 * ref.std()
    )
    assert torch.all(out[~valid] == 0)


@pytest.mark.parametrize("num_tokens", [9, 96])
@pytest.mark.parametrize(
    "n_out,n_in", [(1024, 2048), (2048, 512)], ids=["gate_up", "down"]
)
def test_hipblaslt_w4a16_moe_s_bound(num_tokens: int, n_out: int, n_in: int):
    """Prefill with host bounds {S, 0, ..., 0}, S = sum of per-expert rows
    rounded up to 32, and the real counts written on the device."""
    set_random_seed(0)
    device = "cuda"
    num_experts, top_k = 16, 4
    groups = n_in // GROUP_SIZE
    q = torch.randint(0, 16, (num_experts, n_out, n_in), device=device)
    scale = torch.rand((num_experts, n_out, groups), device=device) * 0.02 + 0.001
    scale = scale.to(torch.bfloat16)
    zp = torch.randint(0, 16, (num_experts, n_out, groups), device=device)
    w = pad_moe_weight_rows(_pack_nibbles(q, 2)).view(torch.int8)
    buf = pack_moe_scale_buffers(scale, _pack_nibbles(zp, 1))

    ids = torch.rand((num_tokens, num_experts), device=device).topk(top_k).indices
    flat = ids.flatten()
    order = torch.argsort(flat, stable=True)
    g = _Groups.make(flat[order], flat, num_experts, num_tokens, flat.numel())
    x = torch.randn((num_tokens, n_in), device=device, dtype=torch.bfloat16)
    x_sorted = x[order // top_k]
    s_bound = int(((g.count.cpu() + 31) // 32 * 32).sum())
    out = torch.empty((x_sorted.size(0), n_out), device=device, dtype=torch.bfloat16)
    if not torch.ops._rocm_C.hipblaslt_w4a16_moe_grouped_gemm(
        out,
        x_sorted,
        w,
        buf,
        g.expert,
        g.count,
        g.offset,
        s_bound,
        GROUP_SIZE,
        True,
        True,
    ):
        pytest.skip("hipBLASLt has no grouped W4A16 solution")

    dequant = (q - zp.repeat_interleave(GROUP_SIZE, dim=2)).float()
    dequant *= scale.float().repeat_interleave(GROUP_SIZE, dim=2)
    ref = torch.einsum("rnk,rk->rn", dequant[flat[order]], x_sorted.float())
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())


@pytest.mark.parametrize("m", [1, 2, 9, 96])
def test_hipblaslt_wna16_experts_vs_torch(m: int):
    """HipblasltWNA16Experts end to end (routing, both GEMMs, activation and
    the weighted reduction) against the dequantized torch reference; m <= 8
    takes the slot-order decode path, larger m the expert-sorted path."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from tests.kernels.utils import torch_experts
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        int4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.hipblaslt_moe import (
        HipblasltWNA16Experts,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    set_random_seed(0)
    device = torch.device("cuda")
    e, topk, k, n = 32, 8, 512, 256  # hidden k, intermediate n

    def make(n_out, n_in):
        q = torch.randint(0, 16, (e, n_out, n_in), device=device)
        s = torch.rand((e, n_out, n_in // GROUP_SIZE), device=device) * 0.02 + 0.001
        s = s.to(torch.bfloat16)
        z = torch.randint(0, 16, (e, n_out, n_in // GROUP_SIZE), device=device)
        w = pad_moe_weight_rows(_pack_nibbles(q, 2))
        buf = pack_moe_scale_buffers(s, _pack_nibbles(z, 1))
        ref = (q - z.repeat_interleave(GROUP_SIZE, dim=2)).to(torch.bfloat16)
        ref = ref * s.repeat_interleave(GROUP_SIZE, dim=2)
        return w, buf, ref

    w1, s1, w1_ref = make(2 * n, k)
    w2, s2, w2_ref = make(k, n)
    hidden = torch.randn(m, k, device=device, dtype=torch.bfloat16) / 4
    scores = torch.randn(m, e, device=device, dtype=torch.bfloat16)
    topk_weights, topk_ids, _ = fused_topk(hidden, scores, topk, False)

    zp_marker = torch.empty(0, dtype=torch.uint8, device=device)
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=s1,
        w2_scale=s2,
        w1_zp=zp_marker,
        w2_zp=zp_marker,
        block_shape=[0, GROUP_SIZE],
    )
    moe_config = make_dummy_moe_config(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size=n,
        in_dtype=torch.bfloat16,
    )
    kernel = FusedMoEKernelModularImpl(
        fused_experts=HipblasltWNA16Experts(moe_config, quant_config),
        prepare_finalize=MoEPrepareAndFinalizeNoDPEPModular(),
    )
    init_workspace_manager(device)
    with set_current_vllm_config(VllmConfig()):
        ref = torch_experts(
            hidden,
            w1_ref,
            w2_ref,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
        )
        out = kernel.apply(
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            expert_map=None,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
        )
    torch.testing.assert_close(
        out.float(), ref.float(), rtol=5e-2, atol=5e-2 * ref.float().std()
    )


def test_hipblaslt_w4a16_moe_rejects_unpadded_rows():
    """Unpadded expert rows (stride a multiple of 1024 B) are a hard error."""
    device = "cuda"
    w = torch.zeros(4, 1024, 1024, dtype=torch.int8, device=device)  # K = 2048
    buf = pack_moe_scale_buffers(
        torch.ones(4, 1024, 64, dtype=torch.bfloat16, device=device),
        torch.zeros(4, 512, 64, dtype=torch.uint8, device=device),
    )
    x = torch.zeros(4, 2048, dtype=torch.bfloat16, device=device)
    out = torch.empty(4, 1024, dtype=torch.bfloat16, device=device)
    ids = torch.arange(4, dtype=torch.int32, device=device)
    ones = torch.ones(4, dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match="must be padded"):
        torch.ops._rocm_C.hipblaslt_w4a16_moe_grouped_gemm(
            out, x, w, buf, ids, ones, ids, 1, GROUP_SIZE, True
        )
    with pytest.raises(RuntimeError, match="must be padded"):
        torch.ops._rocm_C.hipblaslt_w4a16_moe_gemm(
            out, x, w, buf, ones.long().cpu(), GROUP_SIZE, True
        )
