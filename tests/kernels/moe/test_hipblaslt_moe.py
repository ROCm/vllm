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
    pack_moe_scale_buffers,
)
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

GROUP_SIZE = 32


def _pack_nibbles(q: torch.Tensor, dim: int) -> torch.Tensor:
    """Two nibbles per byte along `dim`, even index in the low nibble."""
    lo, hi = q.unflatten(dim, (-1, 2)).unbind(dim + 1)
    return (lo | (hi << 4)).to(torch.uint8)


@pytest.mark.parametrize("has_zp", [True, False], ids=["asym", "sym"])
@pytest.mark.parametrize("num_tokens", [1, 7, 96])
@pytest.mark.parametrize(
    "n_out,n_in", [(1024, 2048), (2048, 512)], ids=["gate_up", "down"]
)
def test_hipblaslt_w4a16_moe_gemm(has_zp: bool, num_tokens: int, n_out: int, n_in: int):
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
    torch.ops._rocm_C.hipblaslt_w4a16_moe_gemm(
        out,
        x_sorted,
        _pack_nibbles(q, 2).view(torch.int8),
        buf,
        counts,
        GROUP_SIZE,
        has_zp,
    )

    dequant = (q - zp.repeat_interleave(GROUP_SIZE, dim=2)).float()
    dequant *= scale.float().repeat_interleave(GROUP_SIZE, dim=2)
    ref = torch.einsum("rnk,rk->rn", dequant[flat[order]], x_sorted.float())
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.std())
