# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of dynamic per-tensor MoE activation quantization.

A dynamic per-tensor activation scale is an amax over every row the kernel was
handed, so a token's quantized activation -- and its output -- depends on what
else was batched with it. Under ``VLLM_BATCH_INVARIANT`` the scheme is promoted
to per-token, which is a function of the token alone.

The fp8 tests are kernel-level: they drive ``TritonExperts`` through the
modular kernel directly, so they need no fp8 MoE checkpoint. The amax itself is
deterministic (``scaled_fp8_quant`` reduces with ``atomicMaxFloat``, and max is
order independent), so the same batch twice always agrees -- the needle has to
move *between* batches, and it is deliberately not row 0 of either.

The int8 tests measure the same promotion for the other w8a8 dtype the guard
covers, on the legacy ``fused_experts`` entry, because ``TritonExperts`` cannot
execute int8 at all -- see the two ``test_triton_experts_*`` tests.

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
    int8_w8a8_moe_quant_config,
    maybe_promote_act_quant_for_batch_invariance,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
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


# The int8 arms below run in fp32, not BF16: `ops.scaled_int8_quant`, which is
# what a *dynamic per-tensor* int8 activation scheme calls, rejects BFloat16
# ("expected scalar type Float"), so the mode-off control -- the arm that has to
# move for the mode-on arm to mean anything -- cannot be built in BF16 at all.
# (The mode-on arm can: per-token int8 quantization does accept BF16, so the
# promotion turns a configuration that raises into one that runs.)
INT8_ACT_DTYPE = torch.float32
# Lower than the fp8 arm's 8.0 on purpose. At 8.0 the per-tensor amax is so far
# above the needle that the needle quantizes to all zeros, and the control then
# "moves" only because the batched output is identically zero, which proves
# nothing about the scale.
INT8_FILLER_SCALE = 2.0


def _quantize_per_expert_int8(w: torch.Tensor):
    """[E, R, C] -> int8 with one scale per expert, stored 1-D [E] as vLLM does."""
    amax = w.abs().amax(dim=(1, 2)).to(torch.float32)
    scale = (amax / 127.0).clamp(min=1e-12)
    q = (w.to(torch.float32) / scale.view(-1, 1, 1)).round().clamp(-127, 127)
    return q.to(torch.int8), scale


def _build_int8(device: torch.device, weight_scale_layout: str = "per_tensor"):
    """The int8 twin of ``_build``, on the legacy ``fused_experts`` entry.

    Not on ``TritonExperts``: that class cannot execute int8 at all -- see
    the two ``test_triton_experts_*`` tests -- while this entry applies the
    same promotion and reaches the same Triton kernel with the same weight-scale
    widening, so it is where the promoted int8 scheme is measurable.

    ``weight_scale_layout`` selects what the kernel's per-channel weight-scale
    branch is handed: the 1-D ``[E]`` scale it widens itself, the same values
    materialized as a contiguous ``[E, N]``, or every expert collapsed onto
    ``w_scale[0]`` -- the read the widening exists to prevent.
    """
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * INTERMEDIATE, K, device=device, dtype=torch.float32) / 8
    w2 = torch.randn(E, K, INTERMEDIATE, device=device, dtype=torch.float32) / 8
    for e in range(E):
        w1[e] *= 2.0 ** (e - 4)
        w2[e] *= 2.0 ** (4 - e)
    w1q, w1s = _quantize_per_expert_int8(w1)
    w2q, w2s = _quantize_per_expert_int8(w2)
    if weight_scale_layout == "materialized":
        w1s = w1s.reshape(E, 1).expand(E, 2 * INTERMEDIATE).contiguous()
        w2s = w2s.reshape(E, 1).expand(E, K).contiguous()
    elif weight_scale_layout == "collapsed":
        w1s = w1s[0].reshape(1, 1).expand(E, 2 * INTERMEDIATE).contiguous()
        w2s = w2s[0].reshape(1, 1).expand(E, K).contiguous()
    else:
        assert weight_scale_layout == "per_tensor"

    needle = torch.randn(1, K, device=device, dtype=INT8_ACT_DTYPE) / 4
    filler = (
        torch.randn(FILLER, K, device=device, dtype=INT8_ACT_DTYPE) * INT8_FILLER_SCALE
    )
    big = torch.cat([filler[:NEEDLE_POS], needle, filler[NEEDLE_POS:]])

    gen = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.stack(
        [torch.randperm(E, generator=gen)[:TOPK] for _ in range(1 + FILLER)]
    ).to(device)
    big_ids = torch.cat([ids[1 : 1 + NEEDLE_POS], ids[0:1], ids[1 + NEEDLE_POS :]])
    weights = torch.full((1 + FILLER, TOPK), 0.5, device=device, dtype=torch.float32)

    quant_config = int8_w8a8_moe_quant_config(
        w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None
    )

    def run(a, topk_weights, topk_ids):
        return fused_experts(
            a,
            w1q,
            w2q,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=E,
            expert_map=None,
            quant_config=quant_config,
        )

    # ``fused_experts`` promotes internally; this is the same call it makes.
    promoted = maybe_promote_act_quant_for_batch_invariance(quant_config)
    small = run(needle, weights[0:1], ids[0:1])[0]
    large_all = run(big, weights, big_ids)
    return promoted, small, large_all


@skip_if_not_cuda_alike
def test_dynamic_per_tensor_int8_moe_act_quant_is_promoted(workspace_init):
    """The int8 twin of the fp8 arm: repeatable first, then invariant.

    Repeatability comes first because a path that cannot reproduce itself
    cannot be batch invariant, and calling it variant would understate the
    defect.

    What this does *not* show is that the int8 GEMM is insensitive to reduction
    order. int8 x int8 products accumulate exactly in fp32 for any realistic
    activation: measured on gfx950, the largest partial sum over K=32768 of
    Gaussian activations is ~1.4e6 against the 2^24 where fp32 stops
    representing integers, and forward, reversed, permuted and differently
    tiled reductions over those products are all bitwise equal. Reordering only
    becomes detectable with near-saturated same-sign products (K >= 2048 at
    |a| = |b| = 127). So the activation scale is the only batch-variance channel
    this scheme has, and it is the one measured here.
    """
    assert envs.VLLM_BATCH_INVARIANT
    device = torch.device(f"{current_platform.device_type}:0")
    promoted, small, large_all = _build_int8(device)

    assert promoted.per_act_token_quant, (
        "the mode should have promoted the dynamic per-tensor activation scheme"
    )
    _, small_again, large_again = _build_int8(device)
    torch.testing.assert_close(small, small_again, atol=0, rtol=0)
    torch.testing.assert_close(large_all, large_again, atol=0, rtol=0)
    torch.testing.assert_close(small, large_all[NEEDLE_POS], atol=0, rtol=0)


@skip_if_not_cuda_alike
def test_dynamic_per_tensor_int8_moe_act_quant_moves_without_the_mode(
    monkeypatch, workspace_init
):
    """Without the mode the int8 needle moves, and not by rounding.

    Measured on gfx950 at K=512: all 512 outputs differ, max |delta| 13.5
    against a needle output of magnitude 19.3. Same at K=4096 and K=8192.
    """
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    device = torch.device(f"{current_platform.device_type}:0")
    promoted, small, large_all = _build_int8(device)

    assert not promoted.per_act_token_quant
    assert not torch.equal(small, large_all[NEEDLE_POS])


@skip_if_not_cuda_alike
def test_per_tensor_weight_scale_widening_is_batch_invariant(workspace_init):
    """``_as_per_channel_weight_scale`` is what makes the promoted path work.

    Promotion sends a per-tensor weight scale down the kernel's per-channel
    branch, where a 1-D ``[E]`` scale would have both strides collapse to 0 and
    every expert would read ``w_scale[0]``. The widening returns a stride-0
    ``[E, N]`` view instead. Two things have to hold: the view must compute
    exactly what a materialized ``[E, N]`` scale computes, and it must actually
    change the answer relative to the collapse -- otherwise the first check
    passes for free.
    """
    assert envs.VLLM_BATCH_INVARIANT
    device = torch.device(f"{current_platform.device_type}:0")
    _, widened, widened_all = _build_int8(device)
    _, materialized, materialized_all = _build_int8(device, "materialized")
    _, collapsed, _ = _build_int8(device, "collapsed")

    torch.testing.assert_close(widened, materialized, atol=0, rtol=0)
    torch.testing.assert_close(widened_all, materialized_all, atol=0, rtol=0)
    assert not torch.equal(widened, collapsed), (
        "every expert already read w_scale[0], so the widening is untested here"
    )


class _DeferQuantTritonExperts(TritonExperts):
    """``TritonExperts`` in its quantize-inside-``apply`` configuration.

    In production that is LoRA plus a DP/EP all2all. The scheme, the scales and
    the batch the amax spans are identical either way; only the step that
    quantizes moves.
    """

    @property
    def expects_unquantized_inputs(self) -> bool:
        return self.quant_dtype is not None


def _int8_triton_experts_call(device: torch.device, experts_cls: type[TritonExperts]):
    """A one-token int8 w8a8 call into ``TritonExperts``, ready to raise."""
    torch.manual_seed(0)
    w1q, w1s = _quantize_per_expert_int8(
        torch.randn(E, 2 * INTERMEDIATE, K, device=device, dtype=torch.float32) / 8
    )
    w2q, w2s = _quantize_per_expert_int8(
        torch.randn(E, K, INTERMEDIATE, device=device, dtype=torch.float32) / 8
    )
    moe_config = FusedMoEConfig(
        num_experts=E,
        experts_per_token=TOPK,
        hidden_dim=K,
        intermediate_size=INTERMEDIATE,
        num_local_experts=E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=INT8_ACT_DTYPE,
        device=device.type,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=1,
    )
    experts = experts_cls(
        moe_config,
        int8_w8a8_moe_quant_config(
            w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None
        ),
    )
    assert experts.per_act_token_quant, "promoted, so the undeclared pair is live"
    kernel = FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), experts)

    def call():
        return kernel.apply(
            torch.randn(1, K, device=device, dtype=INT8_ACT_DTYPE),
            w1q,
            w2q,
            torch.full((1, TOPK), 0.5, device=device, dtype=torch.float32),
            torch.tensor([[0, 1]], device=device),
            activation=MoEActivation.SILU,
            global_num_experts=E,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    return call


# Why the promoted int8 scheme is not declared in ``_supports_quant_scheme``:
# ``TritonExperts.__init__`` promotes a dynamic per-tensor int8 config to
# per-token, which leaves it executing ``(kInt8StaticTensorSym,
# kInt8DynamicTokenSym)`` -- a pair it does not list. Declaring that pair would
# be an unmeasured claim in the strongest sense, because the class cannot run
# any int8 w8a8 scheme at all, listed or not. Both of its *declared* int8 rows
# die at the same two places the next two tests pin, and no production config
# sends it int8 w8a8 anyway: ``make_int8_moe_quant_config`` returns an int8
# *w8a16* config whenever the activation scales are dynamic.
#
# These two pin the raising statement, not just the exception type. A bare
# ``pytest.raises`` would also pass on a shape mismatch or a typo in the
# fixture above, which would leave the negative claim resting on an unrelated
# failure. When either stops raising, the declaration question is worth
# reopening -- with the arms above rerun against this class.


@skip_if_not_cuda_alike
def test_triton_experts_rejects_int8_activations_from_the_prepare_step(workspace_init):
    """The prepare step quantizes, and the dtype allowlist has no int8 in it."""
    assert envs.VLLM_BATCH_INVARIANT
    device = torch.device(f"{current_platform.device_type}:0")
    call = _int8_triton_experts_call(device, TritonExperts)

    with pytest.raises(AssertionError) as excinfo:
        call()

    # The assert carries no message, so pin it by its source instead.
    frame = excinfo.traceback[-1]
    statement = " ".join(str(frame.statement).split())
    assert frame.path.name == "triton_moe.py" and frame.name == "apply", (
        f"expected the dtype allowlist in TritonExperts.apply, got {frame.path}"
        f":{frame.lineno + 1} in {frame.name}"
    )
    assert statement.startswith("assert hidden_states.dtype in"), (
        f"raised somewhere else in apply: {statement}"
    )


@skip_if_not_cuda_alike
def test_triton_experts_cannot_type_its_own_int8_activations(workspace_init):
    """Quantizing inside ``apply`` clears the allowlist and dies one step later.

    ``compute_type`` is derived from ``hidden_states`` *after* that variable has
    been rebound to the quantized tensor, so int8 reaches a branch that has no
    case for it.
    """
    assert envs.VLLM_BATCH_INVARIANT
    device = torch.device(f"{current_platform.device_type}:0")
    call = _int8_triton_experts_call(device, _DeferQuantTritonExperts)

    with pytest.raises(ValueError, match="Unsupported compute_type: torch.int8"):
        call()


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
