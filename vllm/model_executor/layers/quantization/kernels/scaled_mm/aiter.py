# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Sequence

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

from .cutlass import CutlassInt8ScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearLayerConfig,
)


class AiterInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "Requires ROCm."

        if compute_capability is not None and compute_capability < 90:
            return False, "requires compute capability 90 and above."

        try:
            import aiter  # noqa: F401 # deliberately attempt to import aiter
        except Exception:
            return False, "requires `aiter` to be installed."

        if not rocm_aiter_ops.is_linear_enabled():
            return (
                False,
                "requires setting `VLLM_ROCM_USE_AITER=1` "
                + "and `VLLM_ROCM_USE_AITER_LINEAR=1`. "
                + "`VLLM_ROCM_USE_AITER_LINEAR` default is True.",
            )
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "supports symmetric quantization only."
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        `AiterInt8ScaledMMLinearKernel` implements a fused version of
            `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
        where scale_a * a and scale_b * b are implemented using numpy-style
        broadcasting.
        Currently only support per-tensor-per-tensor GEMM
        and per-token-per-channel GEMM through AITER
        w8a8 scaled gemm. `AiterInt8ScaledMMLinearKernel` also does not support
        ATIER block scaled GEMM and mix-precision GEMM.
        """
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        assert symmetric, (
            "AiterInt8ScaledMMLinearKernel only supports symmetric quantization."
        )
        x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, i_zp, symmetric=symmetric)

        assert x_zp is None, (
            "AiterInt8ScaledMMLinearKernel only supports symmetric quantization."
        )
        out_dtype = x.dtype

        assert w_q.shape[0] % 16 == 0 and w_q.shape[1] % 16 == 0
        assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
        assert bias is None or bias.shape[0] == w_q.shape[1] and bias.dtype == out_dtype

        m = x_q.shape[0]  # a
        n = w_q.shape[1]  # b

        per_tensor_scale_a = x_s.numel() == 1
        per_tensor_scale_b = w_s.numel() == 1
        per_token_scale_a = x_s.numel() == m
        per_channel_scale_b = w_s.numel() == n

        # @TODO:
        # Maybe broadcast the per-tensor-scale into per-channel-scale
        # if one of the scale is a per-channel-scale.
        # For now, it only supports:
        # - per-tensor-per-tensor a8w8 scaled GEMM, and
        # - per-token-per-channel a8w8 scaled GEMM
        assert (per_tensor_scale_a and per_tensor_scale_b) or (
            per_token_scale_a and per_channel_scale_b
        ), (
            "Currently only support per-tensor-per-tensor GEMM "
            + " and per-token-per-channel GEMM through AITER"
            " w8a8 scaled gemm. `AiterInt8ScaledMMLinearKernel` "
            + "does not support AITER block scaled GEMM."
        )

        # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects
        # a to be [M, K]
        # b to be [N, K]
        # CutlassInt8ScaledMMLinearKernel prepare weight `w_q` in [K, N] format
        return rocm_aiter_ops.gemm_a8w8(x_q, w_q.t(), x_s, w_s, bias, out_dtype)


class AiterFP8RowWiseShuffledScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)
        self.gemm_kernel = self._scaled_mm_fallback

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."

        from vllm.platforms.rocm import on_mi3xx

        if not on_mi3xx():
            return False, "requires MI3xx."

        if compute_capability is not None and compute_capability < 94:
            return False, "requires compute capability 94 and above."

        try:
            import aiter  # noqa: F401 # deliberately attempt to import aiter
        except Exception:
            return False, "requires `aiter` to be installed."

        if not rocm_aiter_ops.is_linear_enabled():
            return (
                False,
                "requires setting `VLLM_ROCM_USE_AITER=1` "
                + "and `VLLM_ROCM_USE_AITER_LINEAR=1`. "
                + "`VLLM_ROCM_USE_AITER_LINEAR` default is True.",
            )

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if c.out_dtype == torch.float16:
            # hipblaslt rowwise _scaled_mm only supports BFloat16
            return False, "supports BFloat16 output data type only."

        if per_tensor_activation_scales or per_tensor_weight_scales:
            return False, "cannot be used with per tensor activation and weight scales."

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layout = (16, 16)
        use_swizzle_gemm = rocm_aiter_ops.can_shuffle(
            *layer.weight.shape, layout=(128, 64)
        )
        weight = layer.weight
        if use_swizzle_gemm:
            weight = torch.nn.Parameter(
                rocm_aiter_ops.shuffle_weight(weight.t(), layout=layout),
                requires_grad=False,
            )
            self.gemm_kernel = self._scaled_mm_shuffled
        else:
            # keep the weight as (K, N)
            weight = torch.nn.Parameter(weight, requires_grad=False)
            self.gemm_kernel = self._scaled_mm_fallback

        layer.weight = torch.nn.Parameter(weight.data, requires_grad=False)

    def _scaled_mm_fallback(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        output = torch._scaled_mm(
            A,
            B,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs.t(),
            bias=bias,
        )
        return torch.narrow(output, 0, 0, output_shape[0]).view(*output_shape)

    def _scaled_mm_shuffled(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        output_shape = [*A.shape[:-1], B.shape[0]]
        output = rocm_aiter_ops.gemm_a8w8_bpreshuffle(
            A,
            B,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs,
        )
        if bias is not None:
            output = output + bias
        return output.view(*output_shape)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        return self.gemm_kernel(
            A=A,
            B=B,
            out_dtype=out_dtype,
            As=As,
            Bs=Bs,
            bias=bias,
            output_shape=output_shape,
        )
