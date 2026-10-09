# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 adapter for AngelSlim DFlare draft checkpoints."""

from vllm.model_executor.models.dflare import DFlareForCausalLM, DFlareModel


class DFlareQwen3Model(DFlareModel):
    """DFlare body configured for Qwen3 embedding and rotary conventions."""


class DFlareQwen3ForCausalLM(DFlareForCausalLM):
    """vLLM wrapper for AngelSlim Qwen3 DFlare checkpoints."""

    model_cls = DFlareQwen3Model
