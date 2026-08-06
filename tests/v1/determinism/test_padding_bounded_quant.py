# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A dynamic per-tensor amax must not span rows the token does not own.

This is a third property, distinct from the two the neighbouring files test.
Batch invariance is "a token's output does not depend on what was batched with
it"; run-to-run determinism is "the same batch twice agrees". This is neither:
it is that a reduction must not read rows that belong to *no* token in the
batch -- cudagraph padding, or a token-expert slot that was never routed.

Those rows are not merely unused. Under cudagraphs the buffer is a persistent
graph-pool allocation, so a row no producer wrote this step holds the previous
replay's contents; on the MoE path the slots MM1 skipped hold whatever the
shared workspace held. A per-tensor scale is the one granularity whose
reduction crosses them -- per-token and per-group are bounded by construction,
and a static scale has no reduction at all.

Each test carries a positive control that the *unbounded* form does move, so
none of them can pass by being vacuous.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

# A value no activation produces, so a scale that moved when it appeared can
# only have read a row nothing wrote. Not NaN and not Inf: the amax kernels use
# `fmaxf`, which is NaN-blind, and Inf would conflate "the reduction crossed
# this row" with "the arithmetic overflowed".
SENTINEL = 1e30


@pytest.fixture
def vllm_config():
    cfg = VllmConfig()
    with set_current_vllm_config(cfg):
        yield cfg


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a CUDA-alike device"
)
def test_linear_per_tensor_scale_ignores_cudagraph_padding(vllm_config):
    device = current_platform.device_type
    with set_current_vllm_config(vllm_config):
        quant = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)

    n_real, n_pad, hidden = 12, 4, 64
    torch.manual_seed(7)
    x = torch.randn(n_real + n_pad, hidden, device=device, dtype=torch.bfloat16)
    x[n_real:] = SENTINEL

    is_padding = torch.zeros(n_real + n_pad, dtype=torch.bool, device=device)
    is_padding[n_real:] = True

    with set_forward_context(
        None, vllm_config, num_tokens=n_real + n_pad, is_padding_full=is_padding
    ):
        _, bounded = quant(x)

    fp8_max = torch.finfo(current_platform.fp8_dtype()).max
    want = float(x[:n_real].abs().amax()) / fp8_max
    assert float(bounded) == pytest.approx(want, rel=1e-6)

    # Positive control: with no mask the same input takes its scale from the
    # padding rows, which is the behaviour this bound exists to remove.
    _, unbounded = quant(x)
    assert float(unbounded) > 1e26
    assert float(unbounded) > 1e6 * float(bounded)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a CUDA-alike device"
)
def test_linear_per_tensor_scale_unchanged_when_nothing_is_padding(vllm_config):
    """An all-False mask must be a bitwise no-op, not merely a close one."""
    device = current_platform.device_type
    with set_current_vllm_config(vllm_config):
        quant = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)

    torch.manual_seed(11)
    x = torch.randn(16, 64, device=device, dtype=torch.bfloat16)
    is_padding = torch.zeros(16, dtype=torch.bool, device=device)

    with set_forward_context(
        None, vllm_config, num_tokens=16, is_padding_full=is_padding
    ):
        q_masked, s_masked = quant(x)
    q_plain, s_plain = quant(x)

    assert torch.equal(s_masked, s_plain)
    assert torch.equal(q_masked.view(torch.uint8), q_plain.view(torch.uint8))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a CUDA-alike device"
)
def test_moe_a2_scale_ignores_unrouted_slots(vllm_config):
    """The contiguous a2 buffer is one slot per token-expert pair.

    MM1 writes only the routed, local ones and MM2 reads only those; the rest
    hold whatever the workspace held.
    """
    device = current_platform.device_type
    n_slots, hidden, n_live = 24, 64, 18
    torch.manual_seed(13)
    a = torch.randn(n_slots, hidden, device=device, dtype=torch.bfloat16)
    a[n_live:] = SENTINEL

    topk_ids = torch.zeros(n_slots, dtype=torch.int32, device=device)
    topk_ids[n_live:] = -1  # unrouted

    assert int((topk_ids >= 0).sum()) == n_live

    fp8 = current_platform.fp8_dtype()
    _, bounded = moe_kernel_quantize_input(a, None, fp8, False, None, topk_ids=topk_ids)
    _, unbounded = moe_kernel_quantize_input(a, None, fp8, False, None)

    want = float(a[:n_live].abs().amax()) / torch.finfo(fp8).max
    assert float(bounded) == pytest.approx(want, rel=1e-6)
    assert float(unbounded) > 1e26

    # Ids that do not describe these rows must be ignored, not guessed at.
    _, short = moe_kernel_quantize_input(
        a, None, fp8, False, None, topk_ids=topk_ids[:5]
    )
    assert torch.equal(short, unbounded)
    _, fallback = moe_kernel_quantize_input(a, None, fp8, False, None)
    assert torch.equal(fallback, unbounded)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a CUDA-alike device"
)
def test_bound_is_disabled_under_sequence_parallel_moe(vllm_config, monkeypatch):
    """Sequence-parallel MoE shards the tokens; the mask does not.

    Under `use_sequence_parallel_moe` the token dimension is split across TP
    ranks while `ForwardContext.is_padding` stays full-batch, so the mask no
    longer describes these rows.  Slicing it would silently apply the *leading*
    window to whatever shard the rank holds -- and since padding is trailing,
    the shard that actually holds it would get an all-False mask and the bound
    would quietly become a no-op.  Better to leave the reduction unbounded,
    which is at least the documented pre-existing behaviour.

    This pins the wiring: the bound must be *off*, not merely different.
    """
    device = current_platform.device_type
    monkeypatch.setattr(
        type(vllm_config.parallel_config),
        "use_sequence_parallel_moe",
        property(lambda self: True),
    )
    with set_current_vllm_config(vllm_config):
        quant = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)
    assert quant.sequence_parallel_moe is True

    n_real, n_pad, hidden = 12, 4, 64
    torch.manual_seed(19)
    x = torch.randn(n_real + n_pad, hidden, device=device, dtype=torch.bfloat16)
    x[n_real:] = SENTINEL
    is_padding = torch.zeros(n_real + n_pad, dtype=torch.bool, device=device)
    is_padding[n_real:] = True

    with set_forward_context(
        None, vllm_config, num_tokens=n_real + n_pad, is_padding_full=is_padding
    ):
        _, scale = quant(x)
    _, unbounded = quant(x)

    # Disabled: identical to the unmasked path, bit for bit.
    assert torch.equal(scale, unbounded)
    assert float(scale) > 1e26
