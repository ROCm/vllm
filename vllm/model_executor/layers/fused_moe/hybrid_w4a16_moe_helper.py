# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared setup for the HIP HybridW4A16 MoE path.

Used by both `CompressedTensorsWNA16MoEMethod` and `INCHybridW4A16MoEMethod`
to convert GPTQ-packed `[E, K/8, N]` int32 weights into the ExLlama-shuffled
`[E, N, K//8]` int32 layout consumed by `fused_moe_wvSplitK_int4_gemm`
(`csrc/rocm/skinny_gemms_int4.cu`), and to install the matching
`HybridW4A16MoEExperts` modular kernel on the method.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.utils import replace_parameter


def setup_hybrid_w4a16_moe(method, layer: torch.nn.Module) -> None:
    """Convert weights and install `HybridW4A16MoEExperts` on `method`.

    `method` must expose `.moe` and `.get_fused_moe_quant_config(layer)`.
    `layer` must hold `w13_weight_packed`/`w2_weight_packed` (int32, GPTQ
    `[E, K/8, N]` layout) and `w13_weight_scale`/`w2_weight_scale`
    (`[E, K/G, N]`) as parameters.
    """
    from vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16 import (
        pack_int4_exllama_shuffle,
    )
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.hybrid_w4a16_moe import (
        HybridW4A16MoEExperts,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        unpack_quantized_values_into_int32,
    )
    from vllm.scalar_type import scalar_types

    wtype = scalar_types.uint4

    def convert_weights(w_packed: torch.Tensor) -> torch.Tensor:
        E_dim = w_packed.size(0)
        experts = []
        for e in range(E_dim):
            unpacked = unpack_quantized_values_into_int32(
                w_packed[e], wtype, packed_dim=0
            )
            unpacked_t = unpacked.t().contiguous()
            repacked = pack_int4_exllama_shuffle(unpacked_t)
            experts.append(repacked)
        return torch.stack(experts)

    replace_parameter(
        layer,
        "w13_weight_packed",
        torch.nn.Parameter(
            convert_weights(layer.w13_weight_packed), requires_grad=False
        ),
    )
    replace_parameter(
        layer,
        "w2_weight_packed",
        torch.nn.Parameter(
            convert_weights(layer.w2_weight_packed), requires_grad=False
        ),
    )

    layer.w13_weight_scale = torch.nn.Parameter(
        layer.w13_weight_scale.transpose(1, 2).contiguous(),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        layer.w2_weight_scale.transpose(1, 2).contiguous(),
        requires_grad=False,
    )

    layer.use_hybrid_w4a16_moe = True

    method.moe_quant_config = method.get_fused_moe_quant_config(layer)
    assert method.moe_quant_config is not None
    layer.w13_weight = layer.w13_weight_packed
    layer.w2_weight = layer.w2_weight_packed

    prepare_finalize = maybe_make_prepare_finalize(
        moe=method.moe,
        quant_config=method.moe_quant_config,
        routing_tables=layer._maybe_init_expert_routing_tables(),
        allow_new_interface=True,
        use_monolithic=False,
    )
    assert prepare_finalize is not None
    method.moe_kernel = mk.FusedMoEKernel(
        prepare_finalize,
        HybridW4A16MoEExperts(
            moe_config=method.moe, quant_config=method.moe_quant_config
        ),
        shared_experts=None,
        inplace=not method.moe.disable_inplace,
    )
