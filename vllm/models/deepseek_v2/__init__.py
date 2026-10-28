# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V2 / V3 (``deepseek_v2``) platform entry point.

Serves the MLA + MoE architecture shared by DeepSeek-V2, DeepSeek-V3/R1 and
Kimi-K2 (``kimi_k2``, which reuses ``DeepseekV3Config``).

ROCm uses the hardware-isolated implementation under ``amd/``, which owns its
MLA layer end to end. Other platforms keep the generic implementation.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

# The generic branch is the static default that type-checkers see; the ROCm
# branch overrides it at runtime (kept type-compatible via type: ignore).
if TYPE_CHECKING or not current_platform.is_rocm():
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2ForCausalLM,
        DeepseekV3ForCausalLM,
    )
else:
    from .amd.model import (  # type: ignore[assignment]
        DeepseekV2ForCausalLM,
        DeepseekV3ForCausalLM,
    )
    from .amd.mtp import DeepSeekMTP  # type: ignore[assignment]

__all__ = [
    "DeepSeekMTP",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
]
