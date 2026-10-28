# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K2.5 / K2.6 for ROCm: the shared vision stack over the ROCm Kimi-K2
text backbone (``vllm.models.deepseek_v2``)."""

from vllm.model_executor.models.kimi_k25 import (
    KimiK25ForConditionalGeneration as _KimiK25ForConditionalGenerationBase,
)


class KimiK25ForConditionalGeneration(_KimiK25ForConditionalGenerationBase):
    # The registry routes this architecture to vllm.models.deepseek_v2, whose
    # ROCm DeepseekV3ForCausalLM is the Kimi-K2 backbone (DeepseekV3Config).
    language_model_architecture = "DeepseekV3ForCausalLM"
