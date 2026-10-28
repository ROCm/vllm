# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K2.5 / K2.6 (``kimi_k25``) platform entry point.

The vision tower and projector are shared. On ROCm the Kimi-K2 text backbone
runs on the hardware-isolated ``vllm.models.deepseek_v2`` implementation; other
platforms keep the generic model.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING or not current_platform.is_rocm():
    from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration
else:
    from .amd.model import (  # type: ignore[assignment]
        KimiK25ForConditionalGeneration,
    )

__all__ = ["KimiK25ForConditionalGeneration"]
