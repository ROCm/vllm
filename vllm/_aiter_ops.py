# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum
from typing import Optional

import torch

from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


class AITEROpMode(IntEnum):
    NO_AITER = 0
    AITER = 1
    SWIZZLE_AITER = 2


def use_swizzle(n: int, k: int, layout: tuple[int, int]) -> bool:
    return n % layout[0] == 0 and k % (layout[1] * 2) == 0


def use_swizzle_gemm(n: int, k: int, dtype: torch.dtype) -> bool:

    multiple_of: int = 64

    if dtype == current_platform.fp8_dtype():
        multiple_of = 128

    return n % multiple_of == 0 and k % multiple_of == 0


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

    @staticmethod
    def gemm_a8w8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        import aiter as rocm_aiter

        return rocm_aiter.gemm_a8w8_blockscale(A,
                                               B,
                                               As,
                                               Bs,
                                               dtype=output_dtype)

    @staticmethod
    def gemm_a8w8_blockscale_bpreshuffle(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        import aiter as rocm_aiter
        return rocm_aiter.gemm_a8w8_blockscale_bpreshuffle(A,
                                                           B,
                                                           As,
                                                           Bs,
                                                           dtype=output_dtype)
