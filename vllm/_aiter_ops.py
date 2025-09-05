# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def rocm_aiter_maybe_pad_weight_for_shuffle(
        weight: torch.Tensor,
        layout: tuple[int, int] = (16, 16),
) -> torch.Tensor:
    # pad weights with zeros at the end of the dimension
    # so that the shuffle works
    IN, IK = layout
    BK = IK * 2  # BK = 32
    # BN = IN      # BN = 16

    # Calculate padding needed for last dimension (K dimension)
    last_dim = weight.shape[-1]
    pad_k = (BK - (last_dim % BK)) % BK

    # Pad the weight tensor: (pad_left, pad_right)
    # for each dimension from last to first
    padded_weight = torch.nn.functional.pad(weight, (0, pad_k, 0, 0),
                                            mode='constant',
                                            value=0)
    return padded_weight


def rocm_aiter_tuned_gemm_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

    # This AITER function can be used for
    # - BF16 and FP16 matmul
    #   e.g. vllm/model_executor/layers/linear.py
    # - per-tensor activations + per-tensor weights
    #   e.g. vllm/model_executor/layers/quantization/utils/w8a8_utils.py
    from aiter.tuned_gemm import tgemm as aiter_tgemm

    return aiter_tgemm.mm(input,
                          weight,
                          otype=out_dtype,
                          scale_a=scale_a,
                          scale_b=scale_b,
                          bias=bias)


def rocm_aiter_tuned_gemm_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

    m = input.shape[0]
    n = weight.shape[0]
    if out_dtype is None:
        out_dtype = input.dtype
    return torch.empty((m, n), dtype=out_dtype, device=input.device)


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_tuned_gemm",
        op_func=rocm_aiter_tuned_gemm_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_tuned_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class aiter_ops:

    @staticmethod
    def rocm_aiter_tuned_gemm(
            input: torch.Tensor,  # [M, K]
            weight: torch.Tensor,  # [N, K]
            bias: Optional[torch.Tensor] = None,
            out_dtype: Optional[torch.dtype] = None,
            scale_a: Optional[torch.Tensor] = None,
            scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

        return torch.ops.vllm.rocm_aiter_tuned_gemm(
            input,
            weight,
            bias=bias,
            out_dtype=out_dtype,
            scale_a=scale_a,
            scale_b=scale_b,
        )
