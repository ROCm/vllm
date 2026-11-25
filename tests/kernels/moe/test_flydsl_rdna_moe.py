# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection of the rdna_flydsl MoE backend (AITER's FlyDSL W4A16 kernels).
The kernels themselves are tested in AITER."""

import pytest
from compressed_tensors.quantization import QuantizationArgs

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.experts import flydsl_rdna_moe
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt4Static,
    kInt4Static32,
)
from vllm.platforms import current_platform

RDNA_FLYDSL = WNA16MoEBackend.RDNA_FLYDSL


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


@pytest.mark.skipif(
    not _on_gfx115x() or flydsl_rdna_moe.aiter_rdna_moe() is None,
    reason="needs gfx115x and aiter.ops.flydsl.moe_w4a16_rdna",
)
@pytest.mark.parametrize("group_size", [32, 128])
def test_rdna_flydsl_is_the_default(group_size: int):
    """Without an explicit --moe-backend, symmetric 4-bit layers select the
    AITER kernels; layer widths they do not support fall back to another
    backend.
    """
    assert _select(512, group_size) == RDNA_FLYDSL
    if group_size == 32:
        # Not a multiple of the 64-column tile.
        assert _select(544, group_size) != RDNA_FLYDSL


@pytest.mark.skipif(not _on_gfx115x(), reason="needs gfx115x")
def test_rdna_flydsl_needs_aiter(monkeypatch: pytest.MonkeyPatch):
    """Without an importable aiter.ops.flydsl.moe_w4a16_rdna the backend is
    not selected."""
    monkeypatch.setattr(flydsl_rdna_moe, "aiter_rdna_moe", lambda: None)
    assert _select(512, 128) != RDNA_FLYDSL
