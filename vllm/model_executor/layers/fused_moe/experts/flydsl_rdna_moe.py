# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE W4A16 experts on the FlyDSL WMMA kernel for RDNA3 / RDNA3.5."""

import importlib.util

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_flydsl_rdna_moe import (
    BLOCK_M,
    TILE_N,
    flydsl_rdna_moe_gemm,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
    kInt4Static32,
)
from vllm.platforms import current_platform


class FlydslRdnaWNA16Experts(mk.FusedMoEExpertsModular):
    """W4A16 experts backed by ``flydsl_rdna_moe_gemm`` (gfx11).

    ``w1`` / ``w2`` and their scales are in the ``pack_w4a16_weights`` layout;
    zero points are folded into the scale coefficients.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        assert quant_config.use_int4_w4a16, "Supports only int4_w4a16"
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False

        from vllm.platforms.rocm import on_gfx11

        return on_gfx11() and importlib.util.find_spec("flydsl") is not None

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return activation_key is None and weight_key in (kInt4Static, kInt4Static32)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Packed layout [E, N_out / TILE_N, tile stride].
        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        N = w1.size(1) * TILE_N // self.moe_config.w13_num_shards
        return w1.size(0), a1.size(0), N, a1.size(-1), topk_ids.size(1)

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
        # The output is provisioned out of workspace13, so every GEMM buffer
        # lives in workspace2.
        gate_up = N if not activation.is_gated else 2 * N
        return ((M, K), (M * topk * (gate_up + N + K),), (M, K))

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
    ) -> None:
        assert self.w1_scale is not None and self.w2_scale is not None
        assert not apply_router_weight_on_input
        assert expert_map is None, "expert parallelism is not supported"

        E, M, N, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if global_num_experts == -1:
            global_num_experts = E

        gate_up = w1.size(1) * TILE_N
        rows = M * top_k
        widths = (gate_up, N, K)
        sizes = [rows * n for n in widths]
        gate_up_out, act_out, down_out = (
            t.view(rows, n)
            for t, n in zip(workspace2.view(-1)[: sum(sizes)].split(sizes), widths)
        )
        sorted_ids, expert_ids, num_valid = moe_align_block_size(
            topk_ids, BLOCK_M, global_num_experts
        )
        flydsl_rdna_moe_gemm(
            gate_up_out,
            hidden_states,
            w1,
            self.w1_scale,
            sorted_ids,
            expert_ids,
            num_valid,
            top_k,
        )
        self.activation(activation, act_out, gate_up_out)
        flydsl_rdna_moe_gemm(
            down_out, act_out, w2, self.w2_scale, sorted_ids, expert_ids, num_valid, 1
        )

        down_out = down_out.view(M, top_k, K)
        down_out.mul_(topk_weights.view(M, top_k, 1))
        ops.moe_sum(down_out, output)
