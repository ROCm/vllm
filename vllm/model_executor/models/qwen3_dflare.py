# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 adapter for AngelSlim DFlare draft checkpoints."""

from vllm.model_executor.models.gemma4_dflare import (
    DFlareGemma4ForCausalLM,
    DFlareGemma4Model,
)


class DFlareQwen3Model(DFlareGemma4Model):
    """DFlare body configured for Qwen3 embedding and rotary conventions."""


class DFlareQwen3ForCausalLM(DFlareGemma4ForCausalLM):
    """vLLM wrapper for AngelSlim Qwen3 DFlare checkpoints."""

    model_cls = DFlareQwen3Model
