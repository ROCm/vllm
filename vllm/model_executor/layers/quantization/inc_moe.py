# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INC w4a16-sym FusedMoE method on ROCm.

Reuses the HIP HybridW4A16 path: stores `[E, K/8, N]` int32 GPTQ-format
packed weights, then in `process_weights_after_loading` converts to the
ExLlama-shuffled `[E, N, K//8]` layout consumed by
`fused_moe_wvSplitK_int4_gemm` (`csrc/rocm/skinny_gemms_int4.cu`).

The auto-round `auto_round:auto_gptq` packing produces a GPTQ-format
checkpoint with `qweight` / `qzeros` / `scales` suffixes per expert, so
parameters are registered under the GPTQ names (`w*_qweight`, `w*_scales`,
`w*_qzeros`) to match the standard FusedMoE expert-name mapping.
For symmetric quantization the loaded `qzeros` are the GPTQ 7-sentinel
("no zero point") and are dropped before the hybrid kernel sees the
weights.
"""

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.utils import set_weight_attrs


class INCHybridW4A16MoEMethod(FusedMoEMethodBase):
    NUM_BITS = 4
    PACKED_FACTOR = 8  # 32-bit container / 4-bit element

    def __init__(self, moe: FusedMoEConfig, group_size: int):
        super().__init__(moe)
        self.group_size = group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        extra_weight_attrs.update({"is_transposed": True, "quant_method": "group"})
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        # Per-expert GPTQ qweight is [K/8, N] int32; fused w13 is [K/8, 2N].
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.PACKED_FACTOR,
                w13_num_shards * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.PACKED_FACTOR,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size

        # Scales: per-expert [K/group, N] in params_dtype.
        w13_scales = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        set_weight_attrs(w2_scales, {"load_full_w2": False})

        # qzeros: GPTQ-sym stores the 7-sentinel; we accept the load and
        # discard before the kernel runs. Per-expert [K/group, N/8] int32.
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                w13_num_shards * intermediate_size_per_partition // self.PACKED_FACTOR,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.PACKED_FACTOR,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.fused_moe.hybrid_w4a16_moe_helper import (
            setup_hybrid_w4a16_moe,
        )

        # GPTQ-sym qzeros are the 7-sentinel, not real zero points — drop.
        del layer._parameters["w13_qzeros"]
        del layer._parameters["w2_qzeros"]

        # Helper expects compressed-tensors-style names. These start as
        # aliases sharing storage with `w*_qweight` / `w*_scales`; after the
        # helper's repack/transpose they hold new tensors. Delete the
        # originals so the GPTQ-layout copies are freed.
        layer.w13_weight_packed = layer.w13_qweight
        layer.w2_weight_packed = layer.w2_qweight
        layer.w13_weight_scale = layer.w13_scales
        layer.w2_weight_scale = layer.w2_scales

        setup_hybrid_w4a16_moe(self, layer)

        for name in (
            "w13_qweight",
            "w2_qweight",
            "w13_scales",
            "w2_scales",
        ):
            del layer._parameters[name]

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None, (
            "INCHybridW4A16MoEMethod.apply called before "
            "process_weights_after_loading installed the modular kernel."
        )
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )

    @property
    def supports_eplb(self) -> bool:
        return True


def can_use_hybrid_w4a16_moe(weight_bits: int, group_size: int, sym: bool) -> bool:
    """Gate: HIP HybridW4A16 only supports 4-bit symmetric and group_size>0."""
    import vllm.envs as envs
    from vllm.platforms import current_platform

    if not envs.VLLM_MOE_HYBRID_W4A16:
        return False
    if not current_platform.is_rocm():
        return False
    return weight_bits == 4 and sym and group_size > 0
