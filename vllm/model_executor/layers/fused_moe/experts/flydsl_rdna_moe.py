# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE W4A16 experts on AITER's FlyDSL WMMA kernels for RDNA3.5 (gfx115x)."""

import functools
from types import ModuleType

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
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

logger = init_logger(__name__)


@functools.cache
def aiter_rdna_moe() -> ModuleType | None:
    """``aiter.ops.flydsl.moe_w4a16_rdna``, or None when it is not importable."""
    try:
        from aiter.ops.flydsl import moe_w4a16_rdna
    except Exception as e:
        # aiter can also fail at import time with other errors, e.g. on a
        # Triton it was not built for.
        logger.debug("aiter.ops.flydsl.moe_w4a16_rdna is not importable: %s", e)
        return None
    return moe_w4a16_rdna


def rdna_moe() -> ModuleType:
    module = aiter_rdna_moe()
    assert module is not None, "aiter.ops.flydsl.moe_w4a16_rdna is not importable"
    return module


def fuses_silu_mul(moe_config: FusedMoEConfig) -> bool:
    """Whether GEMM1 applies the activation itself; ``w1`` is then packed in
    ``gate_up_tile_order``."""
    return moe_config.activation == MoEActivation.SILU and (
        moe_config.swiglu_limit is None
    )


class FlydslRdnaWNA16Experts(mk.FusedMoEExpertsModular):
    """W4A16 experts backed by AITER's ``flydsl_rdna_moe`` (gfx115x).

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
        self._silu_mul = fuses_silu_mul(moe_config)
        # Without zero points the packer stores one coefficient per group.
        self._w1_pairs = quant_config.w1_zp is not None
        self._w2_pairs = quant_config.w2_zp is not None
        assert not self._silu_mul or self.activation_config.clamp_limit is None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False

        from vllm.platforms.rocm import on_gfx115x

        return on_gfx115x() and aiter_rdna_moe() is not None

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
            moe_parallel_config.use_ep
            or moe_parallel_config.use_fi_nvl_two_sided_kernels
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
        N = w1.size(1) * rdna_moe().TILE_N // self.moe_config.w13_num_shards
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

        assert self.block_shape is not None
        rdna_moe().flydsl_rdna_moe(
            output,
            hidden_states,
            w1,
            self.w1_scale,
            w2,
            self.w2_scale,
            topk_weights,
            topk_ids,
            workspace2,
            self.block_shape[1],
            global_num_experts,
            N,
            None if self._silu_mul else functools.partial(self.activation, activation),
            self._w1_pairs,
            self._w2_pairs,
        )
