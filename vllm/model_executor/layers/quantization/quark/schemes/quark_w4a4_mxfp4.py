# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    OCP_MX_BLOCK_SIZE, per_token_group_quant_mxfp4)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

try:
    from aiter.ops.triton.gemm_afp4wfp4 import (
        gemm_afp4wfp4, gemm_afp4wfp4_preshuffled_scales)
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    if envs.VLLM_TRITON_FP4_GEMM_USE_ASM:
        from aiter import gemm_a4w4_asm
        from aiter.utility.fp4_utils import (dynamic_mxfp4_quant as
                                             dynamic_mxfp4_quant_asm)

    from vllm.utils import direct_register_custom_op

    def gemm_a4w4_asm_fake(
        A: torch.Tensor,  # A:[M, K/2] f4x2
        B: torch.Tensor,  # B:[N, K/2] f4x2
        A_scale: torch.Tensor,  # A_scale:[M, K/32] e8m0 paded
        B_scale: torch.Tensor,  # B_scale:[N, K/32] e8m0 paded
        out: torch.Tensor,  # Out:[M, N] bf16
        bias: torch.Tensor,  # bias:[1, N] f32
        alpha: Optional[float] = 1.0,
        beta: Optional[float] = 0.0,
        bpreshuffle: Optional[bool] = True,
    ) -> torch.Tensor:
        pass

    direct_register_custom_op(
        op_name="gemm_a4w4_asm",
        op_func=gemm_a4w4_asm,
        mutates_args=["out"],
        fake_impl=gemm_a4w4_asm_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    def dynamic_mxfp4_quant_fake(
            x: torch.Tensor,
            scaling_mode: str = "even",
            shuffle: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        M, N = x.shape
        MXFP4_QUANT_BLOCK_SIZE = 32
        x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
        scaleN_valid = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
        scaleN = triton.cdiv(scaleN_valid, 8) * 8
        blockscale_e8m0 = torch.empty(
            (
                triton.cdiv(M, 256) * 256,
                scaleN,
            ),
            dtype=torch.uint8,
            device=x.device,
        )
        return (x_fp4, blockscale_e8m0)

    direct_register_custom_op(
        op_name="dynamic_mxfp4_quant_asm" \
            if envs.VLLM_TRITON_FP4_GEMM_USE_ASM else \
                "dynamic_mxfp4_quant",
        op_func=dynamic_mxfp4_quant_asm \
            if envs.VLLM_TRITON_FP4_GEMM_USE_ASM else \
                dynamic_mxfp4_quant,
        mutates_args=[],
        fake_impl=dynamic_mxfp4_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    def gemm_afp4wfp4_preshuffled_scales_proxy(
        x: torch.Tensor,
        w: torch.Tensor,
        x_scales: torch.Tensor,
        w_scales: torch.Tensor,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        y: Optional[torch.Tensor] = None,
        NUM_KSPLIT: Optional[int] = None,
        BLOCK_SIZE_K: Optional[int] = None,
        SPLITK_BLOCK_SIZE: Optional[int] = None,
        BLOCK_SIZE_N: Optional[int] = None,
    ) -> None:
        config = None
        if NUM_KSPLIT or BLOCK_SIZE_K or SPLITK_BLOCK_SIZE or BLOCK_SIZE_N:
            config = {"NUM_KSPLIT": NUM_KSPLIT} if NUM_KSPLIT else {}
            config = {"BLOCK_SIZE_K": BLOCK_SIZE_K} if BLOCK_SIZE_K else {}
            config = {
                "SPLITK_BLOCK_SIZE": SPLITK_BLOCK_SIZE
            } if SPLITK_BLOCK_SIZE else {}
            config = {"BLOCK_SIZE_N": BLOCK_SIZE_N} if BLOCK_SIZE_N else {}

        gemm_afp4wfp4_preshuffled_scales(x, w, x_scales, w_scales, dtype, y,
                                         config)

    def gemm_afp4wfp4_preshuffled_scales_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        x_scales: torch.Tensor,
        w_scales: torch.Tensor,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        y: Optional[torch.Tensor] = None,
        NUM_KSPLIT: Optional[int] = None,
        BLOCK_SIZE_K: Optional[int] = None,
        SPLITK_BLOCK_SIZE: Optional[int] = None,
        BLOCK_SIZE_N: Optional[int] = None,
    ):
        pass

    from vllm.utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="gemm_afp4wfp4_preshuffled_scales_proxy",
        op_func=gemm_afp4wfp4_preshuffled_scales_proxy,
        mutates_args=["y"],
        fake_impl=gemm_afp4wfp4_preshuffled_scales_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    def gemm_afp4wfp4_proxy(
        x: torch.Tensor,
        w: torch.Tensor,
        x_scales: torch.Tensor,
        w_scales: torch.Tensor,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        y: Optional[torch.Tensor] = None,
        NUM_KSPLIT: Optional[int] = None,
        BLOCK_SIZE_K: Optional[int] = None,
        SPLITK_BLOCK_SIZE: Optional[int] = None,
    ) -> None:
        config = None
        if NUM_KSPLIT or BLOCK_SIZE_K or SPLITK_BLOCK_SIZE:
            config = {"NUM_KSPLIT": NUM_KSPLIT} if NUM_KSPLIT else {}
            config = {"BLOCK_SIZE_K": BLOCK_SIZE_K} if BLOCK_SIZE_K else {}
            config = {
                "SPLITK_BLOCK_SIZE": SPLITK_BLOCK_SIZE
            } if SPLITK_BLOCK_SIZE else {}

        gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, y, config)

    def gemm_afp4wfp4_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        x_scales: torch.Tensor,
        w_scales: torch.Tensor,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        y: Optional[torch.Tensor] = None,
        NUM_KSPLIT: Optional[int] = None,
        BLOCK_SIZE_K: Optional[int] = None,
        SPLITK_BLOCK_SIZE: Optional[int] = None,
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="gemm_afp4wfp4_proxy",
        op_func=gemm_afp4wfp4_proxy,
        mutates_args=["y"],
        fake_impl=gemm_afp4wfp4_fake,
        dispatch_key=current_platform.dispatch_key,
    )

except ImportError:
    dynamic_mxfp4_quant = gemm_afp4wfp4 = None

__all__ = ["QuarkW4A4MXFP4"]


class QuarkW4A4MXFP4(QuarkScheme):

    def __init__(self, weight_quant_spec: dict[str, Any],
                 input_quant_spec: dict[str, Any]):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.emulate = not current_platform.supports_mx()
        if not self.emulate and (dynamic_mxfp4_quant is None
                                 or gemm_afp4wfp4 is None):
            # Currently need these kernels if not emulating
            raise NotImplementedError(
                f"{self.__class__.__name__} requires AITER to be installed "
                "for non-emulation mode! Please refer to "
                "https://github.com/ROCm/aiter for installation details.")

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                          requires_grad=False)

        if self.emulate:
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                    requires_grad=False)
            try:
                from quark.torch.export.nn.modules import realquantizer
                from quark.torch.quantization.config.config import (
                    QuantizationSpec)
            except ImportError as err:
                raise ImportError(
                    "The package `amd-quark` is required to use AMD Quark "
                    "MX-FP4 models. Please install it with `pip install "
                    "amd-quark`.") from err

            weight_quant_spec = QuantizationSpec.from_dict(
                self.weight_quant_spec)

            weight_quantizer = realquantizer.get_real_quantizer(
                qspec=weight_quant_spec,
                quantizer=None,
                real_quantized=True,
                reorder=False,
                float_dtype=self.out_dtype,
                scale_shape=layer.weight_scale.shape,
                zero_point_shape=None,
            )
            weight_quantizer.scale.data = layer.weight_scale.data

            if not envs.VLLM_QUARK_EMU_MEM_OPT:
                layer.weight = torch.nn.Parameter(
                    weight_quantizer(layer.weight.data).to(self.out_dtype),
                    requires_grad=False,
                )
            else:
                self.weight_quantizer = weight_quantizer
            layer.weight_scale = None

            # This call is necessary to release the scales memory.
            torch.cuda.empty_cache()
        else:
            if envs.VLLM_TRITON_FP4_GEMM_USE_ASM:
                weight_scale_shuffle = layer.weight_scale.data
                sm, sn = weight_scale_shuffle.shape
                weight_scale_shuffle = weight_scale_shuffle.view(
                    sm // 32, 2, 16, sn // 8, 2, 4, 1)
                weight_scale_shuffle = weight_scale_shuffle.permute(
                    0, 3, 5, 2, 4, 1, 6).contiguous()
                weight_scale_shuffle = weight_scale_shuffle.view(sm, sn)
                layer.weight_scale = torch.nn.Parameter(weight_scale_shuffle,
                                                        requires_grad=False)
            else:
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.data.T.contiguous(),
                    requires_grad=False)

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None,
                      x_scales: torch.Tensor = None) -> torch.Tensor:

        if self.emulate:
            if envs.VLLM_QUARK_EMU_MEM_OPT:
                dq_w = self.weight_quantizer(layer.weight).to(self.out_dtype)
            else:
                dq_w = layer.weight
            qdq_x, _ = per_token_group_quant_mxfp4(x, OCP_MX_BLOCK_SIZE)
            return F.linear(qdq_x, dq_w, bias)
        else:
            M = x.shape[0]
            if envs.VLLM_TRITON_FP4_GEMM_USE_ASM and M > 128:
                if x_scales is None:
                    x_q, x_s = torch.ops.vllm.dynamic_mxfp4_quant_asm(
                        x, shuffle=True)
                else:
                    x_q = x
                    x_s = x_scales

                y = torch.empty((M + 255) // 256 * 256,
                                layer.weight.shape[0],
                                device=x_q.device,
                                dtype=self.out_dtype)
                #asm_bias = torch.empty_like(y)
                torch.ops.vllm.gemm_a4w4_asm(x_q,
                                             layer.weight,
                                             x_s,
                                             layer.weight_scale,
                                             y,
                                             y,
                                             bpreshuffle=False)

                return y[:M]
            elif envs.VLLM_TRITON_FP4_GEMM_USE_ASM:
                if x_scales is None:
                    x_q, x_s = torch.ops.vllm.dynamic_mxfp4_quant_asm(
                        x, shuffle=(M >= 32))
                    x_s = x_s.view(torch.uint8)
                else:
                    x_q = x
                    x_s = x_scales
                if M >= 32:
                    sm, sn = x_s.shape
                    x_s = x_s.view(sm // 32, sn * 32)
                y = torch.empty(x_q.shape[0],
                                layer.weight.shape[0],
                                device=x_q.device,
                                dtype=self.out_dtype)

                smw, snw = layer.weight_scale.shape
                torch.ops.vllm.gemm_afp4wfp4_preshuffled_scales_proxy(
                    x_q, layer.weight.T, x_s,
                    layer.weight_scale.view(smw // 32, snw * 32),
                    self.out_dtype, y)
                return y
            else:
                if x_scales is None:
                    x_q, x_s = torch.ops.vllm.dynamic_mxfp4_quant(x)
                else:
                    x_q = x
                    x_s = x_scales
                y = torch.empty(x_q.shape[0],
                                layer.weight.shape[0],
                                device=x_q.device,
                                dtype=self.out_dtype)

                torch.ops.vllm.gemm_afp4wfp4_proxy(x_q, layer.weight.T, x_s,
                                                   layer.weight_scale.T,
                                                   self.out_dtype, y)

                return y
