# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def can_shuffle(n: int, k: int, layout: tuple[int, int]) -> bool:
    IN, IK = layout
    BK = IK * 2
    return (n % IN == 0) and (k % BK == 0)


def calc_padding_to_multiples_of_32(k: int) -> int:
    return -k % 32


def rocm_aiter_per_tensor_quant_impl(
    x: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.quant import per_tensor_quant_hip

    return per_tensor_quant_hip(x, scale, dtype)


def rocm_aiter_per_tensor_quant_fake(
    x: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, dtype=dtype), torch.empty(
        1, dtype=torch.float32, device=x.device
    )


def rocm_aiter_per_token_quant_impl(
    out: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    from aiter.ops.quant import dynamic_per_token_scaled_quant

    dynamic_per_token_scaled_quant(
        out,
        x,
        scale,
        scale_ub=None,
        shuffle_scale=False,
        num_rows=None,
        num_rows_factor=1,
    )


def rocm_aiter_per_token_quant_fake(
    out: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    pass


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_per_tensor_quant",
        op_func=rocm_aiter_per_tensor_quant_impl,
        fake_impl=rocm_aiter_per_tensor_quant_fake,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_per_token_quant",
        op_func=rocm_aiter_per_token_quant_impl,
        fake_impl=rocm_aiter_per_token_quant_fake,
        mutates_args=["out", "scale"],
    )

    from aiter.tuned_gemm import tgemm as aiter_tgemm
else:
    aiter_tgemm = None


class aiter_ops:
    _initialized = False

    @classmethod
    def initialize(cls) -> None:
        # Add a safeguard so that
        # aiter_ops can still be imported
        # on non-ROCm platforms and called
        # without causing errors
        if not current_platform.is_rocm():
            return
        if cls._initialized:
            return
        from aiter import hipb_create_extension

        hipb_create_extension()
        cls._initialized = True

    @staticmethod
    def rocm_aiter_tuned_gemm(
        input: torch.Tensor,  # [M, K]
        weight: torch.Tensor,  # [N, K]
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        scale_a: torch.Tensor | None = None,
        scale_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return aiter_tgemm.mm(
            input, weight, otype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias
        )

    def rocm_aiter_per_tensor_quant(
        x: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.rocm_aiter_per_tensor_quant(x, scale, dtype)

    def rocm_aiter_per_token_quant(
        x: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_shape = x.shape
        out = torch.empty(x.shape, dtype=dtype, device=x.device)
        if scale is None:
            scale = torch.empty(
                (*out_shape[:-1], 1), dtype=torch.float32, device=x.device
            )

        torch.ops.vllm.rocm_aiter_per_token_quant(
            out,
            x,
            scale,
        )
        return out, scale

    def hip_bpreshuffle_gemm(
        input: torch.Tensor,  # [M, K]
        weight: torch.Tensor,  # [K, N]
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        scale_a: torch.Tensor | None = None,
        scale_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out_dtype is None:
            out_dtype = torch.bfloat16

        assert out_dtype == torch.bfloat16, (
            f"hip_bpreshuffle_gemm only supports bfloat16 output dtype"
            f", you have passed in {out_dtype}"
        )
        
        # Pad input K dimension to multiples of 32
        num_pad = calc_padding_to_multiples_of_32(input.shape[-1])
        input = F.pad(input, (0, num_pad), "constant", 0)
        
        if input.dim() >= 3:
            inp_view = input.view(-1, input.size(-1))
            batched = True
        else:
            inp_view = input
            batched = False

        from aiter import hipb_mm

        output = hipb_mm(
            inp_view,
            weight,
            solution_index=-1,
            bias=bias,
            out_dtype=out_dtype,
            scaleA=scale_a,
            scaleB=scale_b,
            scaleOut=None,
            bpreshuffle=True,
        )

        if batched:
            output = output.view(*input.shape[:-1], weight.shape[1])

        return output
