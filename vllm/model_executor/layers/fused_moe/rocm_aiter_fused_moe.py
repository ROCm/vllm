# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.utils import (aiter_2stage_moe_enabled, aiter_fp8_block_moe_enabled,
                        aiter_moe_enabled)


def rocm_aiter_fused_experts(
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_fp8_w8a8: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None,
        expert_mask: Optional[torch.Tensor] = None,
        **kwargs  # Ignore additional keyword arguments
) -> torch.Tensor:
    assert aiter_moe_enabled(), "AITER MoE is enabled "
    " only on ROCm platform with VLLM_ROCM_USE_AITER_MOE=1"

    import aiter as rocm_aiter
    import aiter.fused_moe_bf16_asm as rocm_aiter_asm_fmoe

    if aiter_fp8_block_moe_enabled() and use_fp8_w8a8:
        assert w1_scale is not None
        assert w2_scale is not None

        local_E = E = w1.shape[0]
        if expert_mask is not None:
            E = expert_mask.numel()

        topk = topk_ids.shape[1]
        model_dim = w1.shape[-1]
        dtype = hidden_states.dtype
        # The default block sizes are 128 in AITER.
        if block_shape is None:
            block_shape = [128, 128]

        scale_blk_k = block_shape[1]

        (
            sorted_token_ids,
            sorted_weight_buf,
            sorted_expert_ids,
            num_valid_ids,
            out_asm,
        ) = rocm_aiter_asm_fmoe.moe_sorting_ck(topk_ids,
                                               topk_weights,
                                               E,
                                               model_dim,
                                               dtype,
                                               expert_mask=expert_mask)

        a1, a1_scale = per_token_group_quant_fp8(hidden_states, scale_blk_k)
        rocm_aiter.fmoe_fp8_blockscale_g1u1(
            out=out_asm,
            input=a1,
            gate=w1,
            down=w2,
            sorted_token_ids=sorted_token_ids,
            sorted_weight_buf=sorted_weight_buf,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            topk=topk,
            input_scale=a1_scale.t().contiguous(),
            fc1_scale=w1_scale.view(local_E, -1),
            fc2_scale=w2_scale.view(local_E, -1),
            fc_scale_blkn=block_shape[0],
            fc_scale_blkk=block_shape[1],
            fc2_smooth_scale=None,
        )

        return out_asm

    elif use_fp8_w8a8:
        return rocm_aiter_asm_fmoe.asm_moe(hidden_states=hidden_states,
                                           w1=w1,
                                           w2=w2,
                                           topk_weight=topk_weights,
                                           topk_ids=topk_ids,
                                           fc1_scale=w1_scale,
                                           fc2_scale=w2_scale,
                                           fc1_smooth_scale=None,
                                           fc2_smooth_scale=None,
                                           a16=False)
    if aiter_2stage_moe_enabled():
        from aiter.fused_moe_bf16_asm import ck_moe_2stages
        return ck_moe_2stages(a1=hidden_states,
                              w1=w1,
                              w2=w2,
                              topk_weight=topk_weights,
                              topk_ids=topk_ids)
    return rocm_aiter.ck_moe(hidden_states=hidden_states,
                             w1=w1,
                             w2=w2,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids)


def rocm_aiter_topk_softmax(topk_weights: torch.Tensor,
                            topk_indices: torch.Tensor,
                            token_expert_indices: torch.Tensor,
                            gating_output: torch.Tensor,
                            renormalize: bool) -> tuple[torch.Tensor, ...]:
    import aiter as rocm_aiter
    rocm_aiter.topk_softmax(topk_weights, topk_indices, token_expert_indices,
                            gating_output, renormalize)

    return topk_weights, topk_indices


def shuffle_weights(
    *tensors: torch.Tensor, layout: tuple[int] = (16, 16)
) -> tuple[torch.Tensor, ...]:
    """
    Applies shuffle_weight function from AITER to each 
    input tensor and returns them.

    Args:
    *tensors: Variable number of torch.Tensor objects.
    *layout: The layout of the shuffling.
 
    Returns:
    A tuple of shuffled tensors.
    """
    from aiter.ops.shuffle import shuffle_weight

    return tuple(shuffle_weight(tensor, layout=layout) for tensor in tensors)


def expand_weights(*tensors: torch.Tensor,
                   expansion_dims: list[int]) -> tuple[torch.Tensor, ...]:
    """
    Expands the dimensions of input tensors.
 
    Args:
        *tensors: A variable number of torch.Tensor objects.
        expansion_dims: A list of expansion dimensions 
        corresponding to each tensor.
 
    Returns:
        A tuple of tensors with expanded dimensions.
    """

    assert len(tensors) == len(expansion_dims), \
    "Number of tensors must match the number of expansion dimensions."

    return tuple(
        tensor.unsqueeze(-1).unsqueeze(-1).expand((-1, dim, -1))
        for tensor, dim in zip(tensors, expansion_dims))
