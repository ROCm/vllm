# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only DeepSeek-V2/V3 (and Kimi-K2) model -- AMD ROCm implementation.

Runs eagerly (no ``torch.compile``), so the fusions Inductor would otherwise
find are called explicitly: the q/kv latent RMSNorms in the MLA layer, and the
TP all-reduce + residual add + RMSNorm between sublayers. Row-parallel outputs
(``o_proj``, the dense MLP and the MoE) therefore stay unreduced and are summed
by the next fused norm.
"""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    _can_use_aiter_fused_ar_rms,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM as _DeepseekV2ForCausalLMBase,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLP,
    DeepseekV2MoE,
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.sequence import IntermediateTensors

from .mla import DeepseekV2MLAAttention

_KV_SCALE_SUFFIXES = (".q_scale", ".k_scale", ".v_scale", ".prob_scale")


def allreduce_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``norm(all_reduce(hidden_states), residual)`` with the fastest ROCm path.

    Prefers AITER's one-shot fused all-reduce + RMSNorm, which the shared
    ``fused_allreduce_rms_norm`` (flashinfer-only) does not dispatch to, and
    otherwise defers to it (explicit all-reduce + fused add-RMSNorm on ROCm).
    """
    if get_tensor_model_parallel_world_size() > 1 and _can_use_aiter_fused_ar_rms(
        hidden_states
    ):
        return rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()(
            input_=hidden_states,
            residual=residual,
            weight=norm.weight,
            epsilon=norm.variance_epsilon,
            gemma_norm=False,
        )
    return fused_allreduce_rms_norm(hidden_states, residual, norm)


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            reduce_results=False,
        )

        # Set when the MoE combine already all-reduced its output, so ranks > 0
        # contribute zeros to the next fused norm's TP sum.
        self.zero_mlp_output = False
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                # aiter applies routed_scaling_factor internally
                apply_routed_scale_to_output=not rocm_aiter_ops.is_fused_moe_enabled(),
            )
            moe_cfg = self.mlp.experts.moe_config
            if moe_cfg.moe_parallel_config.use_mori_kernels:
                # Validated on MoRI only; other combines keep skipping the
                # final reduce.
                self.zero_mlp_output = get_tensor_model_parallel_rank() > 0
            else:
                moe_cfg.skip_final_all_reduce = True
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=False,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the layer's unreduced output and the residual stream.

        ``hidden_states`` is the previous layer's unreduced output (or the
        embeddings for the first layer, with ``residual=None``).
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        hidden_states = self.mlp(hidden_states)
        if self.zero_mlp_output:
            hidden_states = torch.zeros_like(hidden_states)
        return hidden_states, residual


class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        if config.model_type == "deepseek" or not vllm_config.model_config.use_mla:
            raise ValueError(
                "vllm.models.deepseek_v2 (ROCm) implements MLA only; DeepSeek-V1 "
                "and MLA-disabled runs use vllm.model_executor.models.deepseek_v2."
            )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            DeepseekV2MoE,
            "mlp",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.aux_hidden_state_layers = tuple[int, ...]()
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(self._full_hidden(hidden_states, residual))
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            # The next stage's first layer reduces the partial hidden states.
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, residual = allreduce_rms_norm(hidden_states, residual, self.norm)
        if self.end_layer in self.aux_hidden_state_layers:
            # The fused norm's new residual is the full pre-norm hidden state.
            aux_hidden_states.append(residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _full_hidden(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        """The full (TP-reduced) hidden state entering a layer.

        Between layers ``hidden_states`` is an unreduced partial sum, so the
        Eagle-3 aux states reduce it explicitly.
        """
        if residual is None:
            return hidden_states
        return tensor_model_parallel_all_reduce(hidden_states) + residual

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if self.is_fused_shared_expert_enabled
                else 0
            ),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        # The MLA layer owns its KV-cache scales directly (``self_attn.k_scale``),
        # while the checkpoint-name remap targets an ``attn`` submodule.
        kv_scale_aliases = {
            name.replace(".self_attn.", ".self_attn.attn."): name
            for name in params_dict
            if ".self_attn." in name and name.endswith(_KV_SCALE_SUFFIXES)
        }
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if get_spec_layer_idx_from_weight_name(self.config, name) is not None:
                continue  # skip spec decode layers for main model
            is_fusion_moe_shared_experts_layer = (
                self.is_fused_shared_expert_enabled and ("mlp.shared_experts" in name)
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint. Skip it
                # here, before renaming, so the expert mapping below sees the
                # original name.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # DeepSeek-V2-Lite has no q-LoRA, so no fused_qkv_a_proj.
                if param_name == "fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                # When AITER fusion_shared_experts is enabled the checkpoint
                # provides a single widened shared_experts tensor per layer with
                # no expert index. Split it evenly and route the slices into the
                # expert slots appended after the routed experts.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # gate/up are ColumnParallel (dim 0), down is RowParallel (dim 1).
                    split_dim = (
                        1
                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                        else 0
                    )
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                        if loaded_weight.ndim == 1:
                            weight_to_load = loaded_weight[chunk_slice]
                        elif split_dim == 0:
                            weight_to_load = loaded_weight[chunk_slice, :]
                        else:
                            weight_to_load = loaded_weight[:, chunk_slice]
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping  # type: ignore[assignment]
                        if weight_name not in chunk_name:
                            continue
                        is_expert_weight = True
                        name_mapped = chunk_name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # Ask for success, since otherwise we may skip experts
                        # with other available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # An expert weight not mapped to this rank.
                            continue
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        remapped_name = maybe_remap_kv_scale_name(
                            name, params_dict | dict.fromkeys(kv_scale_aliases)
                        )
                        if remapped_name is None:
                            continue
                        name = kv_scale_aliases.get(remapped_name, remapped_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        loader = getattr(param, "weight_loader", default_weight_loader)
                        loader(param, loaded_weight)
            if name is not None and not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)
        return loaded_params


class DeepseekV2ForCausalLM(_DeepseekV2ForCausalLMBase):
    model_cls = DeepseekV2Model  # type: ignore[assignment]

    def set_moe_parameters(self):
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, DeepseekV2MoE):
                # Pick the last one, since the first layers may be dense.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass
