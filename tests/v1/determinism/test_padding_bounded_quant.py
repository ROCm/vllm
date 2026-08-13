# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A dynamic per-tensor amax must not span rows the token does not own.

Neither batch invariance nor run-to-run determinism: the rows at issue belong to
*no* token in the batch -- cudagraph padding, or an unrouted token-expert slot --
and hold the previous replay's contents rather than nothing. Per-tensor is the
one granularity whose reduction crosses them.

Each test carries a positive control that the unbounded form does move, so none
of them can pass by being vacuous.
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
    """Sequence-parallel MoE shards the tokens; the mask stays full-batch.

    Slicing it would apply the leading window to whatever shard the rank holds
    -- and since padding is trailing, the shard holding it would get an
    all-False mask and the bound would become a silent no-op. So it must be
    off here, not merely different.
    """
    device = current_platform.device_type
    monkeypatch.setattr(
        type(vllm_config.parallel_config),
        "use_sequence_parallel_moe",
        property(lambda self: True),
    )
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


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a CUDA-alike device"
)
def test_moe_a1_scale_ignores_cudagraph_padding(vllm_config):
    """The MoE's *first* activation quantize is one row per token.

    `topk_ids` bounds the a2 buffer, whose rows are token-expert slots. It says
    nothing about a1, whose rows are tokens -- and the rows a1 must not span are
    the cudagraph padding ones.
    """
    device = current_platform.device_type
    n_real, n_pad, hidden = 12, 4, 64
    torch.manual_seed(23)
    a = torch.randn(n_real + n_pad, hidden, device=device, dtype=torch.bfloat16)
    a[n_real:] = SENTINEL

    is_padding = torch.zeros(n_real + n_pad, dtype=torch.bool, device=device)
    is_padding[n_real:] = True

    fp8 = current_platform.fp8_dtype()
    with set_forward_context(
        None, vllm_config, num_tokens=n_real + n_pad, is_padding_full=is_padding
    ):
        _, bounded = moe_kernel_quantize_input(
            a, None, fp8, False, None, bound_cudagraph_padding=True
        )
        # Opting out must be a true no-op, so the a2 call sites are unaffected.
        _, opted_out = moe_kernel_quantize_input(a, None, fp8, False, None)
    _, unbounded = moe_kernel_quantize_input(a, None, fp8, False, None)

    want = float(a[:n_real].abs().amax()) / torch.finfo(fp8).max
    assert float(bounded) == pytest.approx(want, rel=1e-6)
    assert float(unbounded) > 1e26
    assert torch.equal(opted_out, unbounded)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a CUDA-alike device"
)
def test_mask_shorter_than_the_batch_leaves_the_rest_unbounded(vllm_config):
    """A mask that runs out must not be stretched over the rows it never saw.

    Attributing one row's padding bit to another would clip real activations, so
    the uncovered rows are included unbounded -- as if there were no mask.
    """
    device = current_platform.device_type
    quant = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)

    n_rows, hidden = 16, 64
    torch.manual_seed(29)
    x = torch.randn(n_rows, hidden, device=device, dtype=torch.bfloat16)
    x[n_rows - 1] = SENTINEL  # a real row the short mask never reaches

    short = torch.zeros(4, dtype=torch.bool, device=device)
    with set_forward_context(
        None, vllm_config, num_tokens=n_rows, is_padding_full=short
    ):
        _, scale = quant(x)
    _, unbounded = quant(x)

    assert torch.equal(scale, unbounded)
    assert float(scale) > 1e26
