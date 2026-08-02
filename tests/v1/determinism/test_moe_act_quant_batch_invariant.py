# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of dynamic per-tensor MoE activation quantization.

A dynamic per-tensor activation scale is an amax over every row the kernel was
handed, so a token's quantized activation -- and its output -- depends on what
else was batched with it. Under ``VLLM_BATCH_INVARIANT`` the scheme is promoted
to per-token, which is a function of the token alone.

The first two tests are kernel-level: they drive ``TritonExperts`` through the
modular kernel directly, so they need no fp8 MoE checkpoint. The amax itself is
deterministic (``scaled_fp8_quant`` reduces with ``atomicMaxFloat``, and max is
order independent), so the same batch twice always agrees -- the needle has to
move *between* batches, and it is deliberately not row 0 of either.

The last one is end to end. No local fp8 MoE checkpoint uses this scheme (the
``-FP8`` convention is static per-tensor and ``-FP8-dynamic`` is per-token,
both of which are already invariant), but ``quantization="fp8"`` reaches it
from any BF16 MoE model: ``_Fp8OnlineMoEBase`` pairs a per-tensor weight key
with ``kFp8DynamicTensorSym``.
"""

import random

import pytest
import torch
from utils import _extract_step_logprobs, skip_if_not_cuda_alike

import vllm.envs as envs
from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.platforms import current_platform

E = 8
TOPK = 2
K = 512
INTERMEDIATE = 512
FILLER = 63
# The needle sits in the middle of the big batch: index 0 is privileged, both
# in the token ordering and in the weight-scale indexing this exercises.
NEEDLE_POS = 37
DTYPE = torch.bfloat16


def _quantize_per_expert(w: torch.Tensor, fp8_dtype: torch.dtype):
    """[E, R, C] -> fp8 with one scale per expert, stored 1-D [E] as vLLM does."""
    amax = w.abs().amax(dim=(1, 2)).to(torch.float32)
    scale = (amax / torch.finfo(fp8_dtype).max).clamp(min=1e-12)
    q = (w.to(torch.float32) / scale.view(-1, 1, 1)).clamp(
        -torch.finfo(fp8_dtype).max, torch.finfo(fp8_dtype).max
    )
    return q.to(fp8_dtype), scale


def _build(device: torch.device):
    fp8_dtype = current_platform.fp8_dtype()
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * INTERMEDIATE, K, device=device, dtype=DTYPE) / 8
    w2 = torch.randn(E, K, INTERMEDIATE, device=device, dtype=DTYPE) / 8
    # Spread the per-expert weight magnitudes: the promoted scheme takes the
    # kernel's per-channel weight-scale branch, where a misshaped per-tensor
    # scale would send every expert to w_scale[0].
    for e in range(E):
        w1[e] *= 2.0 ** (e - 4)
        w2[e] *= 2.0 ** (4 - e)
    w1q, w1s = _quantize_per_expert(w1, fp8_dtype)
    w2q, w2s = _quantize_per_expert(w2, fp8_dtype)

    needle = torch.randn(1, K, device=device, dtype=DTYPE) / 4
    # High-magnitude filler: this is what drags the batch amax, and with it the
    # per-tensor activation scale, away from what the needle alone would give.
    filler = torch.randn(FILLER, K, device=device, dtype=DTYPE) * 8.0
    big = torch.cat([filler[:NEEDLE_POS], needle, filler[NEEDLE_POS:]])

    gen = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.stack(
        [torch.randperm(E, generator=gen)[:TOPK] for _ in range(1 + FILLER)]
    ).to(device)
    big_ids = torch.cat([ids[1 : 1 + NEEDLE_POS], ids[0:1], ids[1 + NEEDLE_POS :]])
    weights = torch.full((1 + FILLER, TOPK), 0.5, device=device, dtype=torch.float32)

    moe_config = FusedMoEConfig(
        num_experts=E,
        experts_per_token=TOPK,
        hidden_dim=K,
        intermediate_size=INTERMEDIATE,
        num_local_experts=E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=DTYPE,
        device=device.type,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=1 + FILLER,
    )
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None
    )
    kernel = FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(), TritonExperts(moe_config, quant_config)
    )

    def run(a, topk_weights, topk_ids):
        return kernel.apply(
            a,
            w1q,
            w2q,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=E,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    small = run(needle, weights[0:1], ids[0:1])[0]
    large = run(big, weights, big_ids)[NEEDLE_POS]
    return kernel.fused_experts, small, large


@skip_if_not_cuda_alike
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="requires fp8 support")
def test_dynamic_per_tensor_moe_act_quant_is_promoted(workspace_init):
    """The needle's output must not depend on the rest of its batch."""
    assert envs.VLLM_BATCH_INVARIANT
    experts, small, large = _build(torch.device(f"{current_platform.device_type}:0"))

    assert experts.per_act_token_quant, (
        "the mode should have promoted the dynamic per-tensor activation scheme"
    )
    torch.testing.assert_close(small, large, atol=0, rtol=0)


@skip_if_not_cuda_alike
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="requires fp8 support")
def test_dynamic_per_tensor_moe_act_quant_moves_without_the_mode(
    monkeypatch, workspace_init
):
    """Without the mode the same setup is batch variant.

    Guards the test above from passing vacuously: if the filler stopped moving
    the activation amax -- different shapes, a different quantization path --
    the invariance assertion would hold for the wrong reason.
    """
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    experts, small, large = _build(torch.device(f"{current_platform.device_type}:0"))

    assert not experts.per_act_token_quant
    assert not torch.equal(small, large)


# 13GB in BF16; ``quantization="fp8"`` halves that on the device but the
# checkpoint is still downloaded and dequantized in host memory.
E2E_MODEL = "allenai/OLMoE-1B-7B-0924"
_PADDING = "Some background context for the question that follows. " * 100


@skip_if_not_cuda_alike
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="requires fp8 support")
@large_gpu_mark(min_gb=32)
def test_online_fp8_moe_generation_is_bitwise_invariant_e2e():
    """A whole MoE model on the promoted path, through the engine.

    Online fp8 quantization of a BF16 MoE is the reachable form of this scheme:
    per-tensor weights with dynamic per-tensor activations, and no calibrated
    activation scale anywhere for the promotion to defer to. Before the
    promotion existed this configuration was refused at startup under the mode.

    Measured on gfx950: 3/3 needle trials bitwise equal with the mode on and
    3/3 differing with it off, so the case is not numerically inert. The
    control is only that -- with the mode off the run also picks different
    attention and linear kernels, so it does not isolate the MoE scale. The
    kernel-level tests above do.
    """
    llm = LLM(
        model=E2E_MODEL,
        quantization="fp8",
        max_num_seqs=16,
        gpu_memory_utilization=0.35,
        max_model_len=4096,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=16, seed=20240919, logprobs=5
    )
    needle = _PADDING + "Write one factual sentence about the moon."
    baseline = llm.generate([needle], sampling, use_tqdm=False)[0]
    base_logprobs, _ = _extract_step_logprobs(baseline)
    assert base_logprobs is not None

    random.seed(12345)
    for _ in range(3):
        batch_size = random.randint(3, 16)
        # Never index 0: that position keeps its token offset between the solo
        # and batched runs, so it can stay invariant on its own.
        pos = random.randint(1, batch_size - 1)
        prompts = [
            needle if i == pos else _PADDING + f"Describe topic number {i}."
            for i in range(batch_size)
        ]
        out = llm.generate(prompts, sampling, use_tqdm=False)[pos]
        logprobs, _ = _extract_step_logprobs(out)
        assert out.outputs[0].token_ids == baseline.outputs[0].token_ids
        assert torch.equal(logprobs, base_logprobs), (
            f"needle at position {pos} of batch {batch_size} moved: "
            f"max |delta| = {(logprobs - base_logprobs).abs().max().item()}"
        )
