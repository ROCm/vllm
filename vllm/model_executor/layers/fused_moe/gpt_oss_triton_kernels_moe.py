# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG, FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate)
from aiter.ops.triton.moe_routing.routing import routing
from aiter.ops.triton.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.quant_moe import downcast_to_static_fp8


logger = init_logger(__name__)


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    quant_config: Optional[FusedMoEQuantConfig] = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    unpadded_N_w1 = None,
    unpadded_K_w1 = None,
    unpadded_N_w2 = None,
    unpadded_K_w2 = None
) -> torch.Tensor:

    routing_data, gather_idx, scatter_idx = routing(gating_output, topk, sm_first=not renormalize)

    return triton_kernel_fused_experts(
        None,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        activation=activation,
        quant_config=quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        unpadded_N_w1=unpadded_N_w1,
        unpadded_K_w1=unpadded_K_w1,
        unpadded_N_w2=unpadded_N_w2,
        unpadded_K_w2=unpadded_K_w2)


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    activation: str = "silu",
    quant_config: Optional[FusedMoEQuantConfig] = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    a1q_scale: Optional[torch.Tensor] = None,
    unpadded_N_w1 = None,
    unpadded_K_w1 = None,
    unpadded_N_w2 = None,
    unpadded_K_w2 = None
) -> torch.Tensor:
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert (quant_config.w1_bias is None
            or quant_config.w1_bias.dtype == torch.float32)
    assert (quant_config.w2_bias is None
            or quant_config.w2_bias.dtype == torch.float32)

    # Shape check, only check non-mxfp4
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    gammas = routing_data.gate_scal if routing_data else None

    hidden_states = downcast_to_static_fp8(hidden_states, quant_config.w1_precision.flex_ctx.lhs_data.scale)

    intermediate_cache1 = moe_gemm_a8w4(hidden_states, 
        w1.storage.data, 
        None, 
        quant_config.w1_precision.weight_scale.storage.data, 
        quant_config.w1_precision.flex_ctx.lhs_data.scale, 
        quant_config.w2_precision.flex_ctx.lhs_data.scale, 
        quant_config.w1_bias, routing_data, 
        gather_indx=gather_indx, 
        gammas=gammas if apply_router_weight_on_input else None, 
        swizzle_mx_scale="CDNA4_SCALE", 
        out_dtype=torch.float8_e4m3fn, 
        apply_swiglu=True, 
        alpha=swiglu_alpha, 
        limit=swiglu_limit,
        unpadded_N=unpadded_N_w1,
        unpadded_K=unpadded_K_w1)

    intermediate_cache3 = moe_gemm_a8w4(intermediate_cache1, 
        w2.storage.data, 
        None, 
        quant_config.w2_precision.weight_scale.storage.data, 
        quant_config.w2_precision.flex_ctx.lhs_data.scale, 
        None, 
        quant_config.w2_bias, 
        routing_data, 
        scatter_indx=scatter_indx, 
        gammas=None if apply_router_weight_on_input else gammas, 
        swizzle_mx_scale="CDNA4_SCALE",
        unpadded_N=unpadded_N_w2,
        unpadded_K=unpadded_K_w2)

    return intermediate_cache3


class BatchedOAITritonExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config)
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.BatchedExperts,
                mk.FusedMoEActivationFormat.BatchedExperts)

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self, a: torch.Tensor, aq: torch.Tensor, M: int, N: int, K: int,
        topk: int, global_num_experts: int, local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # workspace are allocated inside the kernel
        assert a.dim() == 2
        num_dp = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = self.max_num_tokens
        workspace2 = (0, 0, 0)
        output = (num_experts, max_num_tokens * num_dp, N)
        return (output, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        return triton_kernel_fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            routing_data=None,
            gather_indx=None,
            scatter_indx=None,
            activation=activation,
            quant_config=self.quant_config,
            apply_router_weight_on_input=False,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale)