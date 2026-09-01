# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DynamicInt8LMHeadMethod (per-channel INT8 lm_head on ROCm)."""

import math

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.dynamic_int8_lm_head import (
    DynamicInt8LMHeadMethod,
)
from vllm.scalar_type import scalar_types


def _has_int8_kernel() -> bool:
    """Check if any kernel can handle signed int8 per-channel."""
    try:
        choose_mp_linear_kernel(
            MPLinearLayerConfig(
                full_weight_shape=(128, 256),
                partition_weight_shape=(128, 256),
                weight_type=scalar_types.int8,
                act_type=torch.float16,
                group_size=-1,
                zero_points=False,
                has_g_idx=False,
            )
        )
        return True
    except (ValueError, KeyError):
        return False


DTYPES = [torch.float16, torch.bfloat16]
# (M=vocab, K=hidden) — keep sizes small for fast tests,
# plus one representative real-world shape.
MK_SHAPES = [
    (256, 128),
    (1024, 512),
    (8192, 2560),
]
# 5 is the largest N the wvSplitK_int8 kernel implements; 6 is the first that
# must fall back to F.linear rather than reach the kernel and throw.
N_BATCH = [1, 4, 5, 6]
SEEDS = [0]


def _make_layer(M: int, K: int, dtype: torch.dtype) -> torch.nn.Module:
    """Create a minimal layer with a weight parameter, as create_weights does."""
    method = DynamicInt8LMHeadMethod()
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[M],
        input_size=K,
        output_size=M,
        params_dtype=dtype,
    )
    return method, layer


@pytest.mark.parametrize("n", N_BATCH)
@pytest.mark.parametrize("m,k", MK_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not _has_int8_kernel(), reason="no int8 per-channel kernel")
@torch.inference_mode()
def test_dynamic_int8_lm_head_apply(n, m, k, dtype, seed, default_vllm_config):
    torch.manual_seed(seed)
    method, layer = _make_layer(m, k, dtype)

    # Fill with Xavier-scaled random data to keep magnitudes reasonable.
    xavier = math.sqrt(2 / k)
    layer.weight.data.copy_(
        (torch.rand(m, k, dtype=dtype, device="cpu") * 2 - 1) * xavier
    )
    w_orig = layer.weight.data.clone()

    # Move to GPU, then quantize.
    layer.cuda()
    method.process_weights_after_loading(layer)

    # Verify quantization happened.
    assert layer.weight.dtype == torch.int8, "weight should be INT8 after quantization"
    assert hasattr(layer, "weight_scale"), "weight_scale should be registered"
    assert method._w_orig.dtype == dtype, "_w_orig should keep original dtype"

    # Run apply (exercises wvSplitK_int8 for small N*K).
    x = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    out = method.apply(layer, x)

    assert out.shape == (n, m)
    assert out.dtype == dtype

    # Reference: FP linear with original weights.
    ref = torch.nn.functional.linear(x, w_orig.to(device="cuda", dtype=dtype))

    # Error budget: each weight has up to ±0.5 * scale quantization error,
    # where scale ≈ xavier / 127 ≈ sqrt(2/K) / 127.  Accumulated over K
    # multiply-adds the error grows as ~scale * sqrt(K).  On top of that,
    # FP16/BF16 accumulation adds ~K * eps * |val| error.  The tolerance
    # below is an empirical upper bound that covers both sources across
    # all tested shapes and dtypes.
    atol = torch.finfo(dtype).eps * math.sqrt(k) * 128
    torch.testing.assert_close(out, ref, atol=atol, rtol=5e-2)


@pytest.mark.parametrize("m,k", MK_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not _has_int8_kernel(), reason="no int8 per-channel kernel")
@torch.inference_mode()
def test_dynamic_int8_lm_head_embedding(m, k, dtype, seed, default_vllm_config):
    torch.manual_seed(seed)
    method, layer = _make_layer(m, k, dtype)

    xavier = math.sqrt(2 / k)
    layer.weight.data.copy_(
        (torch.rand(m, k, dtype=dtype, device="cpu") * 2 - 1) * xavier
    )
    w_orig = layer.weight.data.clone()

    layer.cuda()
    method.process_weights_after_loading(layer)

    # Embedding lookup should use original (lossless) weights.
    indices = torch.randint(0, m, (8,), device="cuda")
    emb = method.embedding(layer, indices)

    ref = torch.nn.functional.embedding(indices, w_orig.cuda())
    torch.testing.assert_close(emb, ref, atol=0, rtol=0)


@pytest.mark.parametrize("lm_head_first", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not _has_int8_kernel(), reason="no int8 per-channel kernel")
@torch.inference_mode()
def test_dynamic_int8_lm_head_tie_weights(lm_head_first, dtype, default_vllm_config):
    """A tied lm_head and embedding must share one INT8 copy of the vocab table.

    Both layers get their own DynamicInt8LMHeadMethod and the loader calls
    process_weights_after_loading once per layer, so quantizing them
    independently would build a second INT8 vocab table. The tie must also be
    order-independent: the loader may reach either module first.
    """
    torch.manual_seed(0)
    m, k = 256, 128
    m_embed, embed = _make_layer(m, k, dtype)
    m_head, lm_head = _make_layer(m, k, dtype)

    xavier = math.sqrt(2 / k)
    embed.weight.data.copy_(
        (torch.rand(m, k, dtype=dtype, device="cpu") * 2 - 1) * xavier
    )
    embed.cuda()
    lm_head.cuda()
    embed.quant_method = m_embed

    w_orig = embed.weight.data.clone()
    assert m_head.tie_weights(lm_head, embed) is lm_head
    assert lm_head.weight is embed.weight

    order = (
        ((m_head, lm_head), (m_embed, embed))
        if lm_head_first
        else ((m_embed, embed), (m_head, lm_head))
    )
    for method, layer in order:
        method.process_weights_after_loading(layer)

    # Both sides quantized, sharing one INT8 buffer and one scale tensor.
    assert lm_head.weight.dtype == torch.int8
    assert lm_head.weight.data_ptr() == embed.weight.data_ptr()
    assert lm_head.weight_scale.data_ptr() == embed.weight_scale.data_ptr()
    assert m_head._w_orig.data_ptr() == m_embed._w_orig.data_ptr()

    # Quantizing twice would corrupt the shared table; check it still
    # dequantizes back to the original weights.
    dequant = embed.weight.data.to(dtype) * embed.weight_scale.unsqueeze(1)
    torch.testing.assert_close(dequant, w_orig.cuda(), atol=xavier / 127, rtol=0.05)


@pytest.mark.parametrize("dtype", DTYPES[:1])  # guard logic is dtype-independent
@pytest.mark.skipif(not _has_int8_kernel(), reason="no int8 per-channel kernel")
@torch.inference_mode()
def test_dynamic_int8_lm_head_requantizes_after_reload(dtype, default_vllm_config):
    """Clearing the per-layer guard must let the weight quantize again.

    model_loader/reload/layerwise.py deletes
    ``_already_called_process_weights_after_loading`` and re-runs
    ``process_weights_after_loading`` so a refit re-quantizes. A guard kept on
    the quant method instead of the layer would be invisible to that path,
    leaving an FP16 weight in front of the INT8 kernel.
    """
    torch.manual_seed(0)
    m, k = 256, 128
    method, layer = _make_layer(m, k, dtype)

    xavier = math.sqrt(2 / k)
    layer.weight.data.copy_(
        (torch.rand(m, k, dtype=dtype, device="cpu") * 2 - 1) * xavier
    )
    w_orig = layer.weight.data.clone()
    layer.cuda()

    method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.int8
    assert layer._already_called_process_weights_after_loading is True

    # Reload restores the FP16 parameter recorded at construction, clears the
    # guard, and processes again.
    layer.register_parameter(
        "weight", torch.nn.Parameter(w_orig.cuda(), requires_grad=False)
    )
    delattr(layer, "_already_called_process_weights_after_loading")
    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.int8, "reload must re-quantize the weight"
