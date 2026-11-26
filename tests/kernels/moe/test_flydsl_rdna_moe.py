# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The aiter_rdna MoE backend (AITER's FlyDSL W4A16 kernels): selection, and
checkpoint weights through the repacking and the experts. The kernels
themselves are tested in AITER."""

import types

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts import flydsl_rdna_moe
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    _process_weights_aiter_rdna,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.moe_wna16 import convert_awq_tensor
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    awq_pack,
    kInt4Static,
    kInt4Static32,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

AITER_RDNA = WNA16MoEBackend.AITER_RDNA


def _select(hidden: int, group_size: int) -> WNA16MoEBackend:
    args = QuantizationArgs(
        num_bits=4, type="int", symmetric=True, strategy="group", group_size=group_size
    )
    key = kInt4Static32 if group_size == 32 else kInt4Static
    moe = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=hidden, intermediate_size=256
    )
    return select_wna16_moe_backend(moe, key, args, False, False)[0]


def _on_gfx115x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx115x

    return on_gfx115x()


needs_aiter_rdna = pytest.mark.skipif(
    not _on_gfx115x() or flydsl_rdna_moe.aiter_rdna_moe() is None,
    reason="needs gfx115x and aiter.ops.flydsl.moe_w4a16_rdna",
)


@needs_aiter_rdna
@pytest.mark.parametrize("group_size", [32, 128])
def test_aiter_rdna_is_the_default(group_size: int):
    """Without an explicit --moe-backend, symmetric 4-bit layers select the
    AITER kernels; layer widths they do not support fall back to another
    backend.
    """
    assert _select(512, group_size) == AITER_RDNA
    if group_size == 32:
        # Not a multiple of the 64-column tile.
        assert _select(544, group_size) != AITER_RDNA


@pytest.mark.skipif(not _on_gfx115x(), reason="needs gfx115x")
def test_aiter_rdna_needs_aiter(monkeypatch: pytest.MonkeyPatch):
    """Without an importable aiter.ops.flydsl.moe_w4a16_rdna the backend is
    not selected."""
    monkeypatch.setattr(flydsl_rdna_moe, "aiter_rdna_moe", lambda: None)
    assert _select(512, 128) != AITER_RDNA


@pytest.mark.skipif(not current_platform.is_rocm(), reason="needs ROCm")
def test_aiter_rdna_checks_the_gpu_before_importing_aiter(
    monkeypatch: pytest.MonkeyPatch,
):
    """On other GPUs the backend is skipped without importing AITER."""
    import vllm.platforms.rocm as rocm

    def no_import():
        raise AssertionError("aiter imported on a GPU other than gfx115x")

    monkeypatch.setattr(rocm, "on_gfx115x", lambda: False)
    monkeypatch.setattr(flydsl_rdna_moe, "aiter_rdna_moe", no_import)
    assert _select(512, 128) != AITER_RDNA


def _awq_weights(E: int, N: int, K: int, group_size: int):
    """AutoAWQ tensors after the MoeWNA16 loader: uint8 weights
    [E, N, K // 2], uint8 zero points [E, N // 2, G], scales [E, N, G], and
    the dequantized [E, N, K] reference."""
    G = K // group_size
    q = torch.randint(0, 16, (E, K, N), dtype=torch.int32)
    zp = torch.randint(0, 16, (E, G, N), dtype=torch.int32)
    scales = torch.rand(E, G, N) / 50 + 1e-3
    ref = (q - zp.repeat_interleave(group_size, 1)) * scales.repeat_interleave(
        group_size, 1
    )
    qweight = torch.stack(
        [convert_awq_tensor(awq_pack(q[e], 4, K, N), "qweight") for e in range(E)]
    )
    qzeros = torch.stack(
        [convert_awq_tensor(awq_pack(zp[e], 4, G, N), "qzeros") for e in range(E)]
    )
    return qweight, scales.transpose(1, 2), qzeros, ref.transpose(1, 2)


def _symmetric_int32_weights(E: int, N: int, K: int, group_size: int):
    """Symmetric compressed-tensors tensors: int32 weights [E, N, K // 8] with
    the nibbles of a row in little-endian order (value + 8), no zero points."""
    G = K // group_size
    q = torch.randint(0, 16, (E, N, K), dtype=torch.int32)
    scales = torch.rand(E, N, G) / 50 + 1e-3
    ref = (q - 8) * scales.repeat_interleave(group_size, 2)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    packed = (q.view(E, N, K // 8, 8) << shifts).sum(-1, dtype=torch.int64)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    return packed, scales, None, ref


@needs_aiter_rdna
@pytest.mark.parametrize("m", [1, 5, 6, 64])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize(
    "make_weights", [_awq_weights, _symmetric_int32_weights], ids=["awq", "sym"]
)
def test_aiter_rdna_experts(m: int, group_size: int, make_weights):
    """Checkpoint weights repacked by _process_weights_aiter_rdna and run by
    AiterRdnaWNA16Experts match dequantize-then-matmul, on the path for up to
    5 tokens and on the grouped path above."""
    set_random_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    E, topk, K, N = 8, 2, 256, 128

    w13, s13, z13, w13_ref = make_weights(E, 2 * N, K, group_size)
    w2, s2, z2, w2_ref = make_weights(E, K, N, group_size)
    moe_config = make_dummy_moe_config(
        num_experts=E,
        experts_per_token=topk,
        hidden_dim=K,
        intermediate_size=N,
        in_dtype=dtype,
    )
    layer = types.SimpleNamespace(moe_config=moe_config)
    w13_p, w2_p, c13, c2, zp13, zp2, *_ = _process_weights_aiter_rdna(
        layer,
        *(None if t is None else t.to(device) for t in (w13, w2, s13, s2, z13, z2)),
    )
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=c13,
        w2_scale=c2,
        w1_zp=zp13,
        w2_zp=zp2,
        block_shape=[0, group_size],
    )
    kernel = FusedMoEKernelModularImpl(
        fused_experts=flydsl_rdna_moe.AiterRdnaWNA16Experts(moe_config, quant_config),
        prepare_finalize=MoEPrepareAndFinalizeNoDPEPModular(),
    )

    x = torch.randn(m, K, device=device, dtype=dtype) / 10
    scores = torch.randn(m, E, device=device, dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(x, scores, topk, False)
    init_workspace_manager(device)
    with set_current_vllm_config(VllmConfig()):
        ref = torch_experts(
            x,
            w13_ref.to(device, dtype),
            w2_ref.to(device, dtype),
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=E,
        )
        out = kernel.apply(
            hidden_states=x,
            w1=w13_p,
            w2=w2_p,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=E,
            expert_map=None,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
        )
    out, ref = out.float(), ref.float()
    # Both round the GEMM1 output to bf16, at different points: a few
    # elements near zero differ by up to ~5% of the output's std. A wrong
    # weight layout gives errors of the size of the outputs everywhere.
    assert (out - ref).norm() < 1e-2 * ref.norm()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=0.1 * ref.std().item())
