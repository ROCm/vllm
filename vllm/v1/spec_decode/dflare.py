# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any, Protocol, cast

import torch
from transformers import Qwen3Config
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID


class _ContextKVModel(Protocol):
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None: ...


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


class DFlareProposer(DFlashProposer):
    """DFlare proposer with rejected context removed before KV projection."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflare"
        SpecDecodeBaseProposer.__init__(
            self,
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        self.runner = runner

        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        self.max_padded_query_tokens = max(
            self.max_query_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self.max_positions = self.max_num_tokens + self.max_padded_query_tokens
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_padded_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_padded_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.arange = torch.arange(
            self.max_positions + 1,
            device=device,
            dtype=torch.int32,
        )
        self.parallel_drafting_hidden_state_tensor = None
        # M-RoPE targets (Qwen3-VL / Qwen3.5): (3, tokens) context positions
        # and the draft query positions that continue them.
        self._mrope_context: torch.Tensor | None = None
        self._mrope_query_buffer: torch.Tensor | None = None

        from vllm.model_executor.models.qwen3_dflash import (
            dflash_has_any_non_causal,
        )

        self.dflash_causal = not dflash_has_any_non_causal(
            cast(Qwen3Config, self.draft_model_config.hf_config)
        )
        self._has_rejected_context = False

    @override
    def load_model(self, target_model: torch.nn.Module) -> None:
        text_config = self.vllm_config.model_config.hf_text_config
        rope_parameters = dict(getattr(text_config, "rope_parameters", None) or {})
        if rope_parameters.get("mrope_section"):
            # KV slot positions come from the runner's first num_context
            # tokens, which match the draft context only in padded mode.
            if self.speculative_config.disable_padded_drafter_batch:
                raise NotImplementedError(
                    "DFlare with an M-RoPE target only supports padded drafter "
                    "batch. Please unset disable_padded_drafter_batch in the "
                    "speculative_config."
                )
            # The draft takes the target's M-RoPE layout (sections, interleaving,
            # partial rotary dimension) so its positions line up with the
            # target's, but keeps its own rope_theta: that is what the draft
            # checkpoint was trained with.
            draft_config = self.speculative_config.draft_model_config.hf_config
            partial_rotary_factor = float(
                rope_parameters.get(
                    "partial_rotary_factor",
                    getattr(text_config, "partial_rotary_factor", 1.0),
                )
            )
            layout = {
                key: rope_parameters[key]
                for key in ("mrope_section", "mrope_interleaved")
                if key in rope_parameters
            }
            draft_config.rope_parameters = {
                **(getattr(draft_config, "rope_parameters", None) or {}),
                **layout,
                "partial_rotary_factor": partial_rotary_factor,
            }
            draft_config.partial_rotary_factor = partial_rotary_factor
            # Compiled and CUDA-graph draft forwards read positions from a
            # fixed address, so query M-RoPE lives in one persistent buffer.
            # The extra column keeps row slices non-contiguous like the
            # target's M-RoPE buffer.
            self._mrope_query_buffer = torch.zeros(
                (3, self.max_padded_query_tokens + 1),
                dtype=torch.int64,
                device=self.positions.device,
            )
        super().load_model(target_model)

    def _get_positions(self, num_tokens: int) -> torch.Tensor:
        if self._mrope_query_buffer is not None:
            return self._mrope_query_buffer[:, :num_tokens]
        return super()._get_positions(num_tokens)

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        self._has_rejected_context = num_rejected_tokens_gpu is not None
        self._mrope_context = None
        num_context = target_token_ids.shape[0]
        # KV slots are indexed by token position; only RoPE uses M-RoPE.
        slot_positions = target_positions
        if target_positions.ndim == 2:
            query_buffer = self._mrope_query_buffer
            if self.runner is None or query_buffer is None:
                raise RuntimeError(
                    "DFlare got M-RoPE target positions but was not set up for "
                    "them; it needs the V1 GPU model runner and a target config "
                    "with rope_parameters.mrope_section."
                )
            mrope_context = target_positions[:, :num_context].contiguous()
            self._mrope_context = mrope_context
            slot_positions = self.runner.positions[:num_context].contiguous()

            query_per_req = 1 + self.num_speculative_tokens
            batch_size = cad.batch_size()
            ends = cad.query_start_loc[1 : batch_size + 1].to(torch.long)
            if num_rejected_tokens_gpu is not None:
                # Rejected draft tokens stay at the end of each request's
                # context; the bonus token follows the last accepted one.
                ends = ends - num_rejected_tokens_gpu[:batch_size].to(torch.long)
            last = mrope_context[:, ends - 1]
            offsets = torch.arange(query_per_req, device=last.device, dtype=last.dtype)
            query_mrope = (last[:, :, None] + 1 + offsets).reshape(
                3, batch_size * query_per_req
            )
            query_buffer[:, : query_mrope.shape[1]].copy_(query_mrope)
        return super().set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=slot_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        num_context = self._dflash_num_context
        context_states = self._dflash_hidden_states
        context_positions = self._context_positions_buffer[:num_context]
        context_slots: torch.Tensor | list[torch.Tensor | None] | None = (
            self._context_slot_mapping_buffer[:num_context]
        )
        if self._mrope_context is not None:
            context_positions = self._mrope_context
        if self._has_rejected_context:
            if self._mrope_context is not None and isinstance(
                context_slots, torch.Tensor
            ):
                valid_context = context_slots != PADDING_SLOT_ID
                context_states = context_states[valid_context]
                context_positions = context_positions[:, valid_context]
                context_slots = context_slots[valid_context]
            else:
                context_states, context_positions, context_slots = (
                    compact_dflare_context(
                        context_states,
                        context_positions,
                        context_slots,
                    )
                )
        model = cast(_ContextKVModel, self.model)
        model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slots,
        )
        return (
            {
                "input_ids": self.input_ids[:num_input_tokens],
                "positions": self._get_positions(num_input_tokens),
                "inputs_embeds": None,
            },
            num_input_tokens,
        )
