# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional

import torch

import vllm.model_executor.layers.fused_moe  # noqa
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = ["QuarkMoEMethod", "QuarkW8A8Fp8MoEMethod"]


class QuarkMoEMethod(FusedMoEMethodBase):

    @staticmethod
    def get_moe_method(
            quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
            module: torch.nn.Module,
            layer_name: str) -> "QuarkMoEMethod":
        layer_quant_config = quant_config._find_matched_config(
            layer_name, module)

        if (layer_quant_config.get("output_tensors")
                or layer_quant_config.get("bias")):
            raise NotImplementedError("Currently, Quark models with "
                                      "output_tensors and bias "
                                      "quantized are not supported")
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if quant_config._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8Fp8MoEMethod(weight_config, input_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")


class QuarkW8A8Fp8MoEMethod(QuarkMoEMethod):

    def __init__(self, weight_config: Dict[str, Any], input_config: Dict[str,
                                                                         Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config

        self.weight_qscheme = self.weight_quant.get("qscheme")
        self.input_qscheme = self.input_quant.get("qscheme")
        per_tensor = (self.weight_qscheme == "per_tensor"
                      and self.input_qscheme == "per_tensor")
        per_channel = (self.weight_qscheme == "per_channel"
                       and self.input_qscheme == "per_channel")

        if not (per_tensor or per_channel):
            raise ValueError(
                "For FP8 Fused MoE layers, only per-tensor and per-channel "
                "scales for weights and activations are supported. Found "
                f"{self.weight_qscheme}, {self.input_qscheme}")  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")
        self.is_per_token = self.input_qscheme == "per_channel"
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization.")

        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            is_rocm_aiter_moe_enabled)

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        if self.weight_qscheme == "per_tensor":
            # Allocate 2 scales for w1 and w3 respectively.
            # They are combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, 2, dtype=torch.float32),
                                                  requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSOR quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        elif self.weight_qscheme == "per_channel":
            # quark's scale is 1 dim.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=torch.float32),
                                                  requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, hidden_size, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None.")
            if (not all_close_1d(layer.w13_input_scale)
                    or not all_close_1d(layer.w2_input_scale)):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer. ")
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False)
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False)

        if current_platform.is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = \
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale,
                    layer.w13_input_scale)
            w2_weight, w2_weight_scale, w2_input_scale = \
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w2_weight, layer.w2_weight_scale,
                    layer.w2_input_scale)
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                  requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale,
                                                        requires_grad=False)
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(w13_input_scale,
                                                           requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                 requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                       requires_grad=False)
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(w2_input_scale,
                                                          requires_grad=False)

        # For per-tensor case, Fp8 moe kernel needs single weight scale
        # for w13 per expert. Use max then dequant and requant each expert.
        if self.weight_qscheme == "per_tensor":
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)
        # quark's scale is 1 dim.
        elif self.weight_qscheme == "per_channel":
            if self.is_per_token:
                w13_weight_scale = layer.w13_weight_scale.unsqueeze(-1)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False)
                w2_weight_scale = layer.w2_weight_scale.unsqueeze(-1)
                layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                           requires_grad=False)

        # Property to determine if AITER is used
        if self.rocm_aiter_moe_enabled:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa E501
                rocm_aiter_fused_experts, shuffle_weights)

            # reshaping weights is required for aiter moe kernel.
            shuffled_w13, shuffled_w2 = shuffle_weights(
                layer.w13_weight.data, layer.w2_weight.data)

            layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                 requires_grad=False)

            self.rocm_aiter_fused_experts_func = rocm_aiter_fused_experts
        else:
            from vllm.model_executor.layers.fused_moe import fused_experts
            self.fused_experts_func = fused_experts


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)

        if self.rocm_aiter_moe_enabled:
            return self.rocm_aiter_fused_experts_func(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                use_fp8_w8a8=True,
                per_channel_quant=self.weight_qscheme == "per_channel",
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                expert_map=expert_map)

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=True,
            per_channel_quant=self.weight_qscheme == "per_channel",
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale)