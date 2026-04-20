# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""W8A8 INT8 skinny GEMM for ROCm.

Uses the wvSplitK_w8a8 kernel for small batch sizes (N<=5) where
activations fit in LDS. Falls back to the Triton scaled_mm kernel
for larger batches.

This mirrors the pattern used by HipW8A16LinearKernel for W8A16.
"""

from contextlib import nullcontext

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units

from .ScaledMMLinearKernel import Int8ScaledMMLinearLayerConfig
from .triton import TritonInt8ScaledMMLinearKernel

# INT8 activations in LDS: 1 byte each, full LDS capacity
# gfx9: 64KB, gfx95x: 160KB (kernel checks at runtime)
LDS_CAPACITY_BYTES = 64 * 1024


class ROCmInt8SkinnyGemmLinearKernel(TritonInt8ScaledMMLinearKernel):
    """W8A8 per-channel int8 skinny GEMM for ROCm.

    Uses the wvSplitK_w8a8 kernel for small batch sizes where both
    int8 activations and weights fit the LDS constraint. Falls back
    to TritonInt8ScaledMMLinearKernel for larger batches.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."

        if not envs.VLLM_ROCM_USE_SKINNY_GEMM:
            return False, "requires VLLM_ROCM_USE_SKINNY_GEMM to be enabled."

        try:
            if not hasattr(torch.ops, "_rocm_C") or not hasattr(
                torch.ops._rocm_C, "wvSplitK_w8a8"
            ):
                return False, "wvSplitK_w8a8 op not available in this build."
        except Exception:
            return False, "ROCm ops not available."

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
        w_q, w_s, i_s, i_zp, _ = self._get_layer_params(layer)

        # Quantize activations to int8 (static or dynamic)
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=True
        )
        assert x_zp is None

        out_dtype = x.dtype
        m = x_q.shape[0]  # batch size
        k = w_q.shape[0]  # K dimension (weights are [K, N] from Triton)
        n = w_q.shape[1]  # output features

        per_tensor_scale_a = x_s.numel() == 1

        # Skinny GEMM fast path: small batch with per-tensor activation scale
        if (
            per_tensor_scale_a
            and m <= 5
            and k % 16 == 0
            and n % 16 == 0
            and k * m <= LDS_CAPACITY_BYTES
        ):
            # wvSplitK_w8a8 expects:
            #   in_a = weights [N_out, K] int8
            #   in_b = activations [batch, K] int8
            #   w_scale = per-channel weight scale [N_out] fp16/bf16
            #   a_scale = per-tensor activation scale (scalar) float32
            w_t = w_q.t()  # [K, N] -> [N, K]

            # Prepare per-channel weight scale in output dtype
            per_tensor_scale_b = w_s.numel() == 1
            if per_tensor_scale_b:
                w_scale_chan = w_s.to(out_dtype).expand(n).contiguous()
            else:
                w_scale_chan = w_s.to(out_dtype).contiguous()

            # Activation scale must be float32 scalar
            a_scale = x_s.to(torch.float32).reshape(1)

            ctx = (
                nullcontext()
                if torch.compiler.is_compiling()
                else torch.profiler.record_function(f"wvSplitK_w8a8 {m}x{n}x{k}")
            )
            with ctx:
                return ops.wvSplitK_w8a8(
                    w_t,
                    x_q,
                    w_scale_chan,
                    a_scale,
                    num_compute_units(),
                    bias,
                )

        # Fallback to Triton for larger batches or per-token scales
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
            triton_scaled_mm,
        )

        return triton_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=out_dtype, bias=bias
        )
