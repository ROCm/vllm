# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.v1.spec_decode.utils import PADDING_SLOT_ID


def compact_dflare_context(
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | Sequence[torch.Tensor | None] | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | list[torch.Tensor | None] | None,
]:
    """Remove rejected rows from packed DFlare context inputs."""
    if context_slot_mapping is None:
        return context_states, context_positions, None

    if isinstance(context_slot_mapping, Sequence):
        reference_mapping = next(
            (mapping for mapping in context_slot_mapping if mapping is not None),
            None,
        )
        if reference_mapping is None:
            return context_states, context_positions, list(context_slot_mapping)
        valid_context = reference_mapping != PADDING_SLOT_ID
        return (
            context_states[valid_context],
            context_positions[valid_context],
            [
                mapping[valid_context] if mapping is not None else None
                for mapping in context_slot_mapping
            ],
        )

    valid_context = context_slot_mapping != PADDING_SLOT_ID
    return (
        context_states[valid_context],
        context_positions[valid_context],
        context_slot_mapping[valid_context],
    )
