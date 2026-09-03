# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
import os
from contextlib import nullcontext
from importlib.util import find_spec
from typing import Final

import torch

from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_CONCH_SUPPORTED_WEIGHT_TYPES: Final = [
    scalar_types.uint4,
    scalar_types.uint8,
    scalar_types.uint4b8,
    scalar_types.uint8b128,
]
_CONCH_SUPPORTED_GROUP_SIZES: Final = [-1, 128]

# Shape context for the monkey-patched _get_tuning_parameters.
# Set by apply_weights() before calling into conch; read by the patched
# function.  Safe because model execution is single-threaded per worker.
_current_m_dim: int = 0
_current_n_dim: int = 0


@functools.lru_cache(maxsize=1)
def _is_rdna3() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx11

    return on_gfx11()


def _make_config(m: int, n: int, k: int, warps: int, stages: int) -> dict[str, int]:
    return {
        "cxpr_block_size_m": m,
        "cxpr_block_size_n": n,
        "cxpr_block_size_k": k,
        "cxpr_group_size_m": max(m // 8, 1),
        "num_warps": warps,
        "num_stages": stages,
    }


def _parse_tune_env(value: str) -> dict[str, int]:
    m, n, k, w, s = (int(x) for x in value.split(","))
    return _make_config(m, n, k, w, s)


def _rdna3_tuning_parameters() -> dict[str, int]:
    """RDNA 3.5 tuning for the conch mixed-precision GEMM kernel.

    Replaces conch's default ``_get_tuning_parameters`` via monkey-patch.
    Reads the current problem shape from module-level ``_current_m_dim``
    and ``_current_n_dim`` which are set by ``apply_weights`` before each
    call into conch.

    Environment variable overrides (format: ``M,N,K,warps,stages``):
      ``CONCH_TUNE``          – override all calls
      ``CONCH_TUNE_DECODE``   – override decode path only (m_dim <= 16)
      ``CONCH_TUNE_PREFILL``  – override prefill path only (m_dim > 16)
    Per-phase overrides take priority over ``CONCH_TUNE``.
    """
    m_dim = _current_m_dim
    n_dim = _current_n_dim
    is_decode = m_dim <= 16

    phase_env = os.environ.get(
        "CONCH_TUNE_DECODE" if is_decode else "CONCH_TUNE_PREFILL", ""
    )
    if phase_env:
        return _parse_tune_env(phase_env)

    global_env = os.environ.get("CONCH_TUNE", "")
    if global_env:
        return _parse_tune_env(global_env)

    if is_decode:
        # Tuned on Qwen3-4B w4a16 / gfx1151: 39.6 tok/s, TPOT 25.2 ms
        return _make_config(16, 16, 64, warps=1, stages=1)

    # Per-shape prefill configs tuned on Qwen3-4B w4a16 / gfx1151 (M=128):
    #   N >= 8192 (gate_up_proj-like):  128x128x16_w4 @ 45.1 GB/s
    #   N >= 3072 (qkv_proj-like):       64x128x32_w4 @ 39.9 GB/s
    #   N <  3072 (o_proj / down_proj):   64x64x32_w2  @ 34-39 GB/s
    if n_dim >= 8192:
        return _make_config(128, 128, 16, warps=4, stages=1)
    if n_dim >= 3072:
        return _make_config(64, 128, 32, warps=4, stages=1)
    return _make_config(64, 64, 32, warps=2, stages=1)


def _install_conch_tuning() -> None:
    """Monkey-patch conch's ``_get_tuning_parameters`` once."""
    import conch.kernels.quantization.gemm as _conch_mod

    if getattr(_conch_mod, "_vllm_tuning_installed", False):
        return
    _conch_mod._get_tuning_parameters = _rdna3_tuning_parameters  # type: ignore[attr-defined]
    _conch_mod._vllm_tuning_installed = True  # type: ignore[attr-defined]


class ConchLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        ok, err = cls._validate_config_invariants(c)
        if not ok:
            return False, err
        if c.weight_type not in _CONCH_SUPPORTED_WEIGHT_TYPES:
            error_msg = (
                f"Weight type ({c.weight_type}) not supported by "
                "ConchLinearKernel, supported types are: "
                f"{_CONCH_SUPPORTED_WEIGHT_TYPES}"
            )
            return False, error_msg

        if c.group_size not in _CONCH_SUPPORTED_GROUP_SIZES:
            error_msg = (
                f"Group size ({c.group_size}) not supported by "
                "ConchLinearKernel, supported group sizes are: "
                f"{_CONCH_SUPPORTED_GROUP_SIZES}"
            )
            return False, error_msg

        if c.has_g_idx:
            return (
                False,
                "Activation reordering (g_idx) is not supported by ConchLinearKernel",
            )

        if find_spec("conch") is None:
            error_msg = (
                "conch-triton-kernels is not installed, please "
                "install it via `pip install conch-triton-kernels` "
                "and try again!"
            )
            return False, error_msg

        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    #  `weight_zero_point` is: {input_dim = 1, output_dim = 0, packed_dim = 0}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._validate_layer_invariants(layer)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = x.data.contiguous()
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x

        def transform_w_zp(x):
            # Zero points are stored PACKED as [N//pack_factor, K//G]
            # The Conch kernel expects UNPACKED zeros: [K//G, N]
            # We need to unpack and reorder
            assert isinstance(x, BasevLLMParameter)
            packed = x.data  # shape: [N//pack_factor, K//G], dtype: int32

            # Determine packing based on weight bit width
            size_bits = self.config.weight_type.size_bits
            pack_factor = 32 // size_bits  # 8 for 4-bit, 4 for 8-bit
            mask = (1 << size_bits) - 1  # 0xF for 4-bit, 0xFF for 8-bit

            n_packed, k_groups = packed.shape
            n_full = n_packed * pack_factor

            # Unpack using vectorized bitwise ops
            # shifts = [0, size_bits, 2*size_bits, ...] for each packed position
            shifts = torch.arange(
                0, 32, size_bits, dtype=torch.int32, device=packed.device
            )
            # packed: [N//pack_factor, K//G] -> [N//pack_factor, K//G, 1]
            # shifts: [pack_factor] -> [1, 1, pack_factor]
            # Result: [N//pack_factor, K//G, pack_factor]
            unpacked = (packed.unsqueeze(-1) >> shifts) & mask

            # Permute to [K//G, N//pack_factor, pack_factor] then reshape to [K//G, N]
            unpacked = unpacked.permute(1, 0, 2).reshape(k_groups, n_full)

            x.data = unpacked.to(torch.uint8).contiguous()

            # Update metadata - zeros are no longer packed
            if hasattr(x, "_input_dim"):
                x._input_dim = 0
            if hasattr(x, "_output_dim"):
                x._output_dim = 1
            if hasattr(x, "_packed_factor"):
                x._packed_factor = 1
            return x

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        if self.config.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)
        elif self.w_zp_name is not None:
            layer.register_parameter(self.w_zp_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global _current_m_dim, _current_n_dim

        from conch.ops.quantization.gemm import mixed_precision_gemm

        if _is_rdna3():
            _install_conch_tuning()

        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        # Map channelwise group_size=-1 to the actual input dimension K.
        # The conch kernel computes stride_mul = block_k / group_size;
        # passing -1 produces a negative stride that reads out-of-bounds
        # scale values for all K-blocks after the first.
        group_size = self.config.group_size
        if group_size == -1:
            group_size = x.shape[-1]

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (self.config.partition_weight_shape[1],)

        M = x_2d.shape[0]
        K = x_2d.shape[1]
        N = w_q.shape[1]

        _current_m_dim = M
        _current_n_dim = N

        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(
                f"conch_gemm {M}x{N}x{K} w{self.config.weight_type.size_bits}"
            )
        )
        with ctx:
            output = mixed_precision_gemm(
                x=x_2d,
                w_q_packed=w_q.data,
                w_s=w_s.data,
                w_zp=w_zp.data if w_zp is not None else None,
                weight_size_bits=self.config.weight_type.size_bits,
                weight_bias=self.config.weight_type.bias,
                group_size=group_size,
            )

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
