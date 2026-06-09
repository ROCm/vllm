# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER FlyDSL W4A8 MoE experts (MXFP4 weights + FP8 activations).

This drives AITER's FlyDSL a8w4 MoE kernels the way ATOM does: the same
``aiter.fused_moe`` 2-stage path used by the ``AITER_MXFP4_BF16`` backend, but
with ``gate_mode=INTERLEAVE``. On gfx950 that routes the per-1x32 MXFP4 path to
the FP8-activation FlyDSL kernels (``flydsl_moe*_afp8_wfp4_..._gui``) for
M >= AITER_BF16_FP8_MOE_BOUND, falling back to BF16 activations for small M
(decode). AITER performs the activation FP8 quant and inter-stage requant
internally; the caller only supplies interleaved MXFP4 weights/scales.

Weight layout differs from the SEPARATED/BF16 path: w1's gate/up halves are
interleaved (``shuffle_weight_a16w4(w1, 16, gate_up=True)``); see the dedicated
AITER_FLYDSL_W4A8 branch in oracle/mxfp4.py.
"""

import os

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import AiterExperts
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)

logger = init_logger(__name__)

# Under gate_mode=INTERLEAVE, aiter picks fp8 activation only for
# M >= AITER_BF16_FP8_MOE_BOUND; below it, bf16 activation routes to a cktile
# kernel that rejects the interleaved MXFP4 / e8m0 scales ("Unsupported
# scales/output dtype"). The DSv4 FlyDSL configs are fp8 for ALL token counts,
# so force fp8 everywhere by setting the bound to 0.
_AITER_BF16_FP8_BOUND_ENV = "AITER_BF16_FP8_MOE_BOUND"

__all__ = ["AiterFlyDslW4A8Experts"]


def _force_flydsl_fp8_all_m() -> None:
    """Force the FP8 FlyDSL path for all token counts (incl. decode).

    Idempotent and process-global; invoked when this backend is selected.
    Honors an explicit user-provided bound rather than overriding it.
    """
    cur = os.environ.get(_AITER_BF16_FP8_BOUND_ENV)
    if cur == "0":
        return
    if cur is not None:
        logger.warning_once(
            "aiter_flydsl wants %s=0 (force FP8 FlyDSL for all M), but it is "
            "already set to %s; leaving the user value. Small-M decode may hit "
            "the unsupported bf16/cktile path.",
            _AITER_BF16_FP8_BOUND_ENV,
            cur,
        )
        return
    os.environ[_AITER_BF16_FP8_BOUND_ENV] = "0"
    logger.info_once(
        "aiter_flydsl backend: set %s=0 to force FP8 FlyDSL MoE for all token "
        "counts (decode + prefill).",
        _AITER_BF16_FP8_BOUND_ENV,
    )


def _flydsl_available() -> bool:
    try:
        from aiter.ops.flydsl.utils import is_flydsl_available

        return bool(is_flydsl_available())
    except Exception:
        return False


class AiterFlyDslW4A8Experts(AiterExperts):
    """MXFP4 weights + FP8 activations via AITER FlyDSL kernels (gfx950)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force the FlyDSL gate-up-interleaved (gui) path, which selects the
        # FP8-activation a8w4 kernels on gfx950. Read by AiterExperts.apply.
        self._aiter_gate_mode = "interleave"
        # gate_mode=interleave + bound=0 => fp8 FlyDSL for all M (avoids the
        # bf16/cktile path that errors on interleaved MXFP4 scales at decode).
        _force_flydsl_fp8_all_m()

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.platforms.rocm import on_gfx950

        return (
            rocm_aiter_ops.is_fused_moe_enabled()
            and on_gfx950()
            and _flydsl_available()
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # MXFP4 weights only; FP8 activation is applied internally by the
        # FlyDSL kernel (dynamic per-1x32), so no separate activation key.
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # FlyDSL afp8_wfp4 path uses SILU-gated (DeepSeek-V4) or SwiGLU-OAI.
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUOAI]
