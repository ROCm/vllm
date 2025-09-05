# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache
from typing import Optional

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx9
from vllm.utils import direct_register_custom_op

# helper function


def shuffle_weights(
    *tensors: torch.Tensor, layout: tuple[int, int] = (16, 16)
) -> tuple[torch.Tensor, ...]:
    """
    Applies shuffle_weight function from AITER to each 
    input tensor and returns them.
    
    Rearranges (shuffles) the input tensor/s
    into a specified block layout for optimized computation.

    Args:
        *tensors: Variable number of torch.Tensor objects.
        layout: A pair of integers specifying the 
        block sizes used to divide the tensors during shuffling.
        Default is (16, 16).

    Returns:
    A Tuple of shuffled tensors.
    """
    from aiter.ops.shuffle import shuffle_weight

    return tuple(shuffle_weight(tensor, layout=layout) for tensor in tensors)


def rocm_aiter_maybe_pad_weight_for_shuffle(
    weight: torch.Tensor, layout: tuple[int, int] = (16, 16)) -> torch.Tensor:
    # pad weights with zeros at the end of the dimension so that the shuffle works
    _, IK = layout
    BK = IK * 2  # BK = 32

    # Calculate padding needed for last dimension (K dimension)
    last_dim = weight.shape[-1]
    pad_k = (BK - (last_dim % BK)) % BK
    need_to_pad = pad_k > 0

    if need_to_pad:
        # Pad the weight tensor: (pad_left, pad_right) for each dimension from last to first
        padded_weight = torch.nn.functional.pad(weight, (0, pad_k, 0, 0),
                                                mode='constant',
                                                value=0)
    else:
        padded_weight = weight
    return padded_weight


@cache
def is_rocm_aiter_hipb_gemm_enabled() -> bool:
    return current_platform.is_rocm() \
        and envs.VLLM_ROCM_USE_AITER_HIPB_GEMM \
        and envs.VLLM_ROCM_USE_AITER \
        and on_gfx9()


def rocm_aiter_hipb_gemm_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:
    from aiter import hipb_mm

    # print(f"hipb_gemm_impl")
    # print(f"input: {input.shape}")
    # print(f"weight: {weight.shape}")
    # if bias is not None:
    #     print(f"bias: {bias.shape}")
    # print(f"out_dtype: {out_dtype}")
    # if scale_a is not None:
    #     print(f"scale_a: {scale_a.shape}")
    # if scale_b is not None:
    #     print(f"scale_b: {scale_b.shape}")

    out = hipb_mm(
        input,
        weight,
        solution_index=-1,
        bias=bias,
        out_dtype=out_dtype,
        scaleA=scale_a,
        scaleB=scale_b,
        scaleOut=None,
        swizzle=True,
    )

    return out


def rocm_aiter_hipb_gemm_fake(
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
    from aiter import hipb_create_extension
    hipb_create_extension()

    direct_register_custom_op(
        op_name="rocm_aiter_hipb_gemm",
        op_func=rocm_aiter_hipb_gemm_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_hipb_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
        tags=(torch.Tag.needs_fixed_stride_order, ),
    )

    direct_register_custom_op(
        op_name="rocm_aiter_tuned_gemm",
        op_func=rocm_aiter_tuned_gemm_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_tuned_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class aiter_ops:

    @staticmethod
    def rocm_aiter_hipb_gemm_swizzle(
            input: torch.Tensor,  # [M, K]
            weight: torch.Tensor,  # [N, K]
            bias: Optional[torch.Tensor] = None,
            out_dtype: Optional[torch.dtype] = None,
            scale_a: Optional[torch.Tensor] = None,
            scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:
        # print(f"rocm_aiter_hipb_gemm")
        # print(f"input: {input.shape}")
        # print(f"weight: {weight.shape}")
        # if bias is not None:
        #     print(f"bias: {bias.shape}")
        # print(f"out_dtype: {out_dtype}")
        # if scale_a is not None:
        #     print(f"scale_a: {scale_a.shape}")
        # if scale_b is not None:
        #     print(f"scale_b: {scale_b.shape}")

        assert input.dtype in [
            torch.bfloat16, torch.float8_e4m3fnuz
        ], (f"input dtype: {input.dtype}, only support bfloat16 and float8_e4m3fnuz"
            )

        if input.dim() >= 3:
            inp_view = input.view(-1, input.size(-1))
        else:
            inp_view = input

        padded_inp_view = rocm_aiter_maybe_pad_weight_for_shuffle(inp_view)
        shuffled_weight = shuffle_weights(
            rocm_aiter_maybe_pad_weight_for_shuffle(weight))[0]

        if out_dtype is None:
            out_dtype = input.dtype

        assert out_dtype in [
            torch.bfloat16
        ], (f"out_dtype: {out_dtype}, only support bfloat16 output")

        if scale_b is not None and scale_b.dim() == 2:
            scale_b = scale_b.t()

        out = torch.ops.vllm.rocm_aiter_hipb_gemm(
            padded_inp_view,
            shuffled_weight.t(),
            bias=bias,
            out_dtype=out_dtype,
            scale_a=scale_a,
            scale_b=scale_b,
        )

        if input.dim() >= 3:
            out = out.view(*input.shape[:-1], weight.shape[0])
        if out_dtype is not None:
            out = out.to(out_dtype)
        return out

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
