# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from contextlib import nullcontext

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    awq_pack,
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    permute_param_layout_,
)

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class HipW4A16LinearKernel(MPLinearKernel):
    w_zp_name: str | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 0 means "auto" for awq_gemv_hip; keep default consistent.
        self._split_k: int = 0

    @classmethod
    def get_min_capability(cls) -> int:
        # RDNA3/3.5 (gfx11xx) reports major=11, minor=0+ -> capability 110+
        return 110

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        ok, err = cls._validate_config_invariants(c)
        if not ok:
            return False, err
        import vllm._custom_ops as ops

        if ops.hip_w4a16_linear_kernel_apply_weights is None:
            return False, (
                "HipW4A16LinearKernel requires the "
                "hip_w4a16_linear_kernel_apply_weights op "
                "(not available in this build)"
            )

        if c.out_type not in (None, torch.float16):
            return False, "HipW4A16LinearKernel supports float16 output only"

        if c.weight_type.is_signed() or c.weight_type.size_bits != 4:
            return False, ("HipW4A16LinearKernel supports unsigned 4-bit weights only")

        if c.act_type != torch.float16:
            return False, "HipW4A16LinearKernel supports float16 activations only"

        if c.group_size < 16 or c.group_size % 16 != 0:
            return False, (
                "HipW4A16LinearKernel requires group size that is a positive "
                f"multiple of 16, got {c.group_size}"
            )

        full_k, full_n = c.full_weight_shape
        part_k, part_n = c.partition_weight_shape
        if (
            (full_k <= 0)
            or (full_n <= 0)
            or (part_k <= 0)
            or (part_n <= 0)
            or (full_k % part_k != 0)
            or (full_n % part_n != 0)
        ):
            return (
                False,
                "Partition weight shape must divide full weight shape "
                f"(full={c.full_weight_shape}, part={c.partition_weight_shape})",
            )

        if part_k % c.group_size != 0:
            return (
                False,
                f"Input features ({part_k}) must be divisible by group size "
                f"({c.group_size})",
            )

        pack_factor = 32 // c.weight_type.size_bits
        if part_n % pack_factor != 0:
            return (
                False,
                f"Output features ({part_n}) must be divisible by pack_factor "
                f"({pack_factor})",
            )

        if c.has_g_idx:
            return False, "HipW4A16LinearKernel does not support g_idx reordering"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._validate_layer_invariants(layer)
        c = self.config
        K = c.partition_weight_shape[0]
        N = c.partition_weight_shape[1]
        num_groups = 1 if c.group_size == -1 else K // c.group_size
        # 0 lets awq_gemv_hip pick split-k with its own heuristic, and without
        # a target factor to divide into there is nothing to pad K for.
        self._split_k = 0
        pack_factor = 32 // c.weight_type.size_bits

        if not c.zero_points:
            # Synthesize zero-points for symmetric quantized checkpoints so
            # the HIP AWQ kernel can run.  Each nibble is set to the weight
            # type's bias (e.g. 8 for uint4b8) so that dequantization
            # correctly computes (raw_value - bias) * scale.
            device = getattr(layer, self.w_q_name).device
            bias = c.weight_type.bias
            packed_bias = 0
            for i in range(pack_factor):
                packed_bias |= bias << (c.weight_type.size_bits * i)
            # Convert to signed int32 range for torch.full
            if packed_bias > 0x7FFFFFFF:
                packed_bias -= 0x100000000
            zeros = torch.full(
                (N // pack_factor, num_groups),
                packed_bias,
                dtype=torch.int32,
                device=device,
            )
            if self.w_zp_name is None:
                self.w_zp_name = "weight_zero_point"
            if getattr(layer, self.w_zp_name, None) is None:
                layer.register_parameter(
                    self.w_zp_name,
                    BasevLLMParameter(data=zeros, weight_loader=lambda *_a, **_k: None),
                )
            self.config.zero_points = True

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            # Normalize to (K//pack, N) with packed_dim=0 regardless of
            # whether the source is compressed-tensors or AWQ.
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            w_unpacked = unpack_quantized_values_into_int32(
                x.data, self.config.weight_type, packed_dim=0
            ).contiguous()
            return awq_pack(w_unpacked, c.weight_type.size_bits, w_unpacked.shape[0], N)

        def transform_w_zp(x):
            assert isinstance(x, BasevLLMParameter)
            packed_dim = getattr(x, "packed_dim", 0)
            w_unpacked = (
                unpack_quantized_values_into_int32(
                    x.data, self.config.weight_type, packed_dim=packed_dim
                )
                .t()
                .contiguous()
            )
            return awq_pack(w_unpacked, c.weight_type.size_bits, w_unpacked.shape[0], N)

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            # Normalize to (G, N) with input_dim=0 regardless of source.
            permute_param_layout_(x, input_dim=0, output_dim=1)
            return x.data.contiguous()

        self._transform_param(layer, self.w_q_name, transform_w_q)
        if self.config.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import vllm._custom_ops as ops

        w_q, w_s, w_zp, _ = self._get_weight_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (self.config.partition_weight_shape[1],)
        M = x_2d.shape[0]
        K = self.config.partition_weight_shape[0]
        N = self.config.partition_weight_shape[1]
        split_k = self._split_k
        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"hip_w4a16 {M}x{N}x{K}")
        )
        with ctx:
            output = ops.hip_w4a16_linear_kernel_apply_weights(
                x_2d, w_q, w_s, w_zp, K, split_k
            )

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
