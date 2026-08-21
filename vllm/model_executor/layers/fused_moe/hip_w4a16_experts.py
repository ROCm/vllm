# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AWQ GEMV-based MoE experts for ROCm (RDNA3/3.5)."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up


class HipW4A16Experts(mk.FusedMoEExpertsModular):
    """MoE experts using the AWQ GEMV HIP kernel for 4-bit weights on ROCm.

    Weights are in AWQ format: [E, K, N/8] int32 (packed 4-bit).
    Scales are [E, K/G, N] fp16.
    Zero-points are [E, K/G, N/8] int32 (AWQ packed).
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        # 0 lets awq_gemv_moe_hip pick split-k with its own heuristic.
        self._split_k_w1 = 0
        self._split_k_w2 = 0

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "HipW4A16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.GELU]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_fi_nvl_two_sided_kernels

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # AWQ format: w1 is [E, K, N/8]
        assert w1.dim() == 3 and w2.dim() == 3
        E = w1.size(0)
        N = w1.size(2) * 8
        K = a1.size(-1)
        assert a1.dim() == 2
        M = a1.size(0)
        topk = topk_ids.size(1)
        return E, M, N, K, topk

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        num_slots = round_up(M * topk, 1)
        workspace1 = (num_slots, max(activation_out_dim, K))
        workspace2 = (num_slots, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

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
        assert w1.dtype == torch.int32
        assert hidden_states.dim() == 2
        assert hidden_states.is_contiguous()

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        activation_out_dim = self.adjust_N_for_activation(N, activation)

        sorted_token_ids, expert_ids, _ = moe_align_block_size(
            topk_ids,
            1,
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

        tw = topk_weights if topk_weights is not None else hidden_states.new_empty(0)

        num_slots = sorted_token_ids.size(0)

        gemm1_out = _resize_cache(workspace2, (num_slots, N))
        act_out = _resize_cache(workspace13, (num_slots, activation_out_dim))
        gemm2_out = _resize_cache(workspace2, (num_slots, K))

        # GEMM 1
        ops.awq_gemv_moe_hip(
            hidden_states,
            w1,
            self.w1_scale,
            self.quant_config.w1_zp,
            gemm1_out,
            sorted_token_ids,
            expert_ids,
            tw.view(-1),
            top_k_num,
            False,
            split_k=self._split_k_w1,
        )

        apply_moe_activation(activation, act_out, gemm1_out)

        # GEMM 2
        ops.awq_gemv_moe_hip(
            act_out,
            w2,
            self.w2_scale,
            self.quant_config.w2_zp,
            gemm2_out,
            sorted_token_ids,
            expert_ids,
            tw.view(-1),
            1,
            not apply_router_weight_on_input,
            split_k=self._split_k_w2,
        )

        ops.moe_sum(gemm2_out.view(num_tokens, top_k_num, K), output)
