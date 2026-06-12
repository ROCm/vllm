# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER Triton W4A8 MoE experts (MXFP4 weights + dynamic FP8 activations).

Modular backend driving aiter's Triton ``moe_gemm_a8w4`` kernel for DeepSeek-V4:
routing is done upstream by vLLM (DeepseekV4 noaux_tc), and this class builds
the kernel's RoutingData from the precomputed topk via ``routing_a8w4_from_topk``,
then runs the two-stage GEMM with dynamic per-tensor FP8 activation scales
(DSv4 MoE has no static input scale): stage1 (gate+up, SILU) -> bf16 ->
requant to fp8 -> stage2 (down).

Weight layout: w13 gate/up are interleaved (the kernel's swiglu reads
even=gate / odd=up) then CDNA4-swizzled via ``_swizzle_mxfp4`` — see the
dedicated AITER_TRITON_W4A8 branch in oracle/mxfp4.py. Recipe validated
numerically on gfx950 (scripts/validate_triton_a8w4.py).
"""

import os

import torch
import triton

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)

__all__ = ["AiterTritonW4A8Experts"]

_FP8_MAX = 448.0
# Single-CTA fused_routing_from_topk caps at NK=4096; above it (prefill chunks)
# use a scalable torch argsort + the fast triton ExptData kernel.
_FUSED_ROUTING_NK_MAX = 4096
# Step 1 (leaner pipeline): fold the inter-stage requant into stage1's epilogue
# via out_mx_quant. Toggle off to recover the original explicit 2-kernel requant
# (for A/B accuracy isolation).
_FOLD_REQUANT = os.environ.get("VLLM_DSV4_TRITON_FOLD_REQUANT", "1") == "1"


def _routing_from_topk(topk_weights, topk_ids, n_experts, block_m):
    """Build moe_gemm_a8w4 RoutingData from precomputed topk for any NK."""
    from aiter.ops.triton.moe.moe_routing.routing import routing_a8w4_from_topk

    tw = topk_weights.to(torch.float32).contiguous()
    tid = topk_ids.to(torch.int32).contiguous()
    if tw.numel() <= _FUSED_ROUTING_NK_MAX:
        return routing_a8w4_from_topk(tw, tid, n_experts, block_m)

    # Scalable fallback (prefill): expert-sort via argsort; reuse the fast
    # triton ExptData kernel. gate/scatter form a valid inverse-perm pair.
    from aiter.ops.triton.moe.moe_routing.routing import (
        ExptData,
        RoutingData,
        _compute_expt_data_internal,
        _expt_data_only_kernel,
    )

    n_gates = tw.numel()
    flat_e = tid.flatten()
    order = torch.argsort(flat_e, stable=True)
    topk_indx = order.to(torch.int32)
    gate_indx = torch.argsort(order, stable=True).to(torch.int32)
    gate_scal = tw.flatten()[order]
    hist = torch.bincount(flat_e, minlength=n_experts).to(torch.int32)

    offs_raw, offs_pad, pid_map, blocks1a, BLOCK_A, bm_log2 = (
        _compute_expt_data_internal(n_experts, n_gates, block_m, tw.device)
    )
    _expt_data_only_kernel[(blocks1a,)](
        hist, n_experts, offs_raw, offs_pad, pid_map, pid_map.shape[0],
        n_gates, bm_log2, BLOCK_A, (hist.shape[0] == BLOCK_A), num_warps=1,
    )
    rdata = RoutingData(
        block_m=block_m, gate_scal=gate_scal, expt_hist=hist,
        n_expts_tot=n_experts, n_expts_act=topk_ids.shape[1],
        expt_data=ExptData(hist, offs_raw, offs_pad, pid_map),
    )
    return rdata, topk_indx, gate_indx


class AiterTritonW4A8Experts(mk.FusedMoEExpertsModular):
    """MXFP4 weights + dynamic FP8 activations via aiter Triton moe_gemm_a8w4."""

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.platforms.rocm import on_gfx950

        return rocm_aiter_ops.is_fused_moe_enabled() and on_gfx950()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        # MXFP4 weights; FP8 activation is quantized dynamically in-class.
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUOAI]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_all2all_kernels

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return True  # routing is done upstream; we only consume topk_ids/weights

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self, M, N, K, topk, global_num_experts, local_num_experts,
        expert_tokens_meta, activation,
    ):
        # aiter manages its own workspaces.
        return (0,), (0,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
        from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            should_use_cdna4_mx_scale_swizzle,
        )

        # The weight scale's CDNA4 swizzle is applied at load time only when
        # should_use_cdna4_mx_scale_swizzle() is true (gfx950 AND TP<=2; at
        # TP>=4 the A8W4 dispatch uses BLOCK_K<256 tiles, incompatible with the
        # 256-aligned CDNA4 scale swizzle). The kernel's swizzle_mx_scale arg
        # MUST agree with what _swizzle_mxfp4 actually did — hardcoding
        # "CDNA4_SCALE" at TP>=4 makes the kernel read an un-swizzled (Strided)
        # scale as if CDNA4-swizzled, silently corrupting the per-block weight
        # scales (coherent but ~18pt GSM8K loss at TP=8). Gate it like the
        # non-modular aiter_mxfp4_w4a8_moe reference does.
        swz = "CDNA4_SCALE" if should_use_cdna4_mx_scale_swizzle() else None

        qc = self.quant_config
        E = global_num_experts if global_num_experts > 0 else w1.shape[0]
        M = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        swiglu_limit = (
            qc.gemm1_clamp_limit if qc.gemm1_clamp_limit is not None else 1e30
        )

        # Build kernel RoutingData from vLLM's precomputed (DeepseekV4) topk.
        block_m = max(16, min(triton.next_power_of_2(max(M * topk // E, 1)), 128))
        routing_data, gather_idx, scatter_idx = _routing_from_topk(
            topk_weights, topk_ids, E, block_m
        )

        unpadded_i = self.moe_config.intermediate_size_per_partition_unpadded
        unpadded_h = self.moe_config.hidden_dim_unpadded

        # Stage 1 (gate+up, SILU) — dynamic mxfp8 (per-1x32) activation.
        # mxfp8 is far finer than per-tensor fp8: per-tensor crushes precision on
        # real activations with outlier channels, degrading generation quality.
        x_q, x_sc = downcast_to_mxfp(hidden_states, torch.float8_e4m3fn, axis=-1)
        # Leaner pipeline (decode): fold the inter-stage requant into stage1's
        # epilogue. out_mx_quant=True makes stage1 emit (fp8 e4m3, ue8m0 per-1x32
        # scale) straight from the GEMM write-back — exactly stage2's input — so
        # the standalone _downcast_to_mxfp kernel between the stages is removed
        # (−1 launch + −1 HBM round-trip / layer). The mx-out epilogue requires
        # split_k==1, which holds for the whole fused-routing range (NK<=4096,
        # i.e. decode); prefill (NK>4096, where split_k may be >1) keeps the
        # explicit 2-kernel requant. Same tuned stage1 config either way.
        fold_requant = _FOLD_REQUANT and topk_ids.numel() <= _FUSED_ROUTING_NK_MAX
        stage1 = moe_gemm_a8w4(
            x_q, w1.storage.data, x_sc, qc.w1_precision.weight_scale.storage.data,
            None, None, qc.w1_bias, routing_data, gather_indx=gather_idx,
            swizzle_mx_scale=swz, out_dtype=torch.bfloat16,
            apply_swiglu=True, alpha=1.0, limit=swiglu_limit,
            swiglu_add_residual=False, out_mx_quant=fold_requant,
            unpadded_N=unpadded_i * 2 if unpadded_i else None,
            unpadded_K=unpadded_h,
        )
        if fold_requant:
            i_q, i_sc = stage1  # (fp8 values, ue8m0 per-1x32 scales) from epilogue
        else:
            i_q, i_sc = downcast_to_mxfp(stage1, torch.float8_e4m3fn, axis=-1)

        # Stage 2 (down) with gammas.
        out = moe_gemm_a8w4(
            i_q, w2.storage.data, i_sc, qc.w2_precision.weight_scale.storage.data,
            None, None, qc.w2_bias, routing_data, scatter_indx=scatter_idx,
            gammas=None if apply_router_weight_on_input else routing_data.gate_scal,
            swizzle_mx_scale=swz, out_dtype=torch.bfloat16,
            unpadded_N=unpadded_h, unpadded_K=unpadded_i,
        )
        output.copy_(out.view(output.shape))
