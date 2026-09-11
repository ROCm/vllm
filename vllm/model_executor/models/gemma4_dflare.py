# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 DFlare draft model for vLLM's parallel speculator path."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    Qwen3MLP,
    _resolve_layer_attention,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.multimodal.inputs import NestedTensors


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_angelslim_rope(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
    layout: str = "legacy",
    cos_sin_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the rotary layout used by AngelSlim DFlare training."""
    if cos_sin_cache is None:
        head_dim = hidden_states.shape[-1]
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )
        frequencies = torch.einsum("n,d->nd", positions.float(), inv_freq)
        base_cos = frequencies.cos()
        base_sin = frequencies.sin()
    else:
        cached = cos_sin_cache[positions].to(hidden_states.dtype)
        base_cos, base_sin = cached.chunk(2, dim=-1)
    if layout == "neox":
        cos = torch.cat((base_cos, base_cos), dim=-1)
        sin = torch.cat((base_sin, base_sin), dim=-1)
    elif layout == "legacy":
        cos = base_cos.repeat_interleave(2, dim=-1)
        sin = base_sin.repeat_interleave(2, dim=-1)
    else:
        raise ValueError(f"Unsupported DFlare RoPE layout: {layout}")
    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)
    while cos.dim() < hidden_states.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    return hidden_states * cos + _rotate_half(hidden_states) * sin


class DFlareGemma4Attention(DFlashQwen3Attention):
    """DFlash query attention with dedicated context K/V projections."""

    def __init__(
        self,
        hidden_size: int,
        target_hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool,
        sliding_window: int | None,
        causal: bool,
        is_neox_style: bool,
        rope_layout: str,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_parameters=rope_parameters,
            max_position=max_position,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.target_k_proj = ReplicatedLinear(
            target_hidden_size,
            self.kv_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.target_k_proj",
        )
        self.target_v_proj = ReplicatedLinear(
            target_hidden_size,
            self.kv_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.target_v_proj",
        )
        self.dflare_rope_theta = float(rope_parameters.get("rope_theta", 10000.0))
        self.dflare_rope_layout = rope_layout

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        )
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        )
        q = _apply_angelslim_rope(
            q,
            positions,
            self.dflare_rope_theta,
            self.dflare_rope_layout,
            self.rotary_emb.cos_sin_cache,
        ).view(q_shape)
        k = _apply_angelslim_rope(
            k,
            positions,
            self.dflare_rope_theta,
            self.dflare_rope_layout,
            self.rotary_emb.cos_sin_cache,
        ).view(k_shape)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlareGemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        self.self_attn = DFlareGemma4Attention(
            hidden_size=config.hidden_size,
            target_hidden_size=getattr(
                config,
                "target_hidden_size",
                config.hidden_size,
            ),
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            sliding_window=sliding_window,
            causal=causal,
            is_neox_style=getattr(config, "is_neox_style", True),
            rope_layout=(getattr(config, "dflare_config", None) or {}).get(
                "rope_layout", "legacy"
            ),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=prefix + ".self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
            )
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DFlareGemma4Model(DFlashQwen3Model):
    """DFlash execution shell with DFlare context fusion and projections."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self._dflare_rope_theta = float(
            self.config.rope_parameters.get("rope_theta", 10000.0)
        )
        self._dflare_rope_layout = (
            getattr(self.config, "dflare_config", None) or {}
        ).get("rope_layout", "legacy")
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)
        self.use_aux_hidden_state = False
        current_config = get_current_vllm_config()
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.register_buffer(
            "embedding_scale",
            torch.tensor(
                int(
                    getattr(
                        self.config,
                        "target_embedding_size",
                        self.config.hidden_size,
                    )
                )
                ** 0.5,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )
        dflare_config = getattr(self.config, "dflare_config", None) or {}
        self.mask_token_id = dflare_config.get("mask_token_id")
        self.mask_embedding = nn.Parameter(
            torch.zeros(
                self.config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            ),
            requires_grad=False,
        )
        self.has_separate_mask_embedding = False
        self.layers = nn.ModuleList(
            [
                DFlareGemma4DecoderLayer(
                    current_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    cache_config=current_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.hidden_norm = RMSNorm(
            getattr(
                self.config,
                "target_hidden_size",
                self.config.hidden_size,
            ),
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        target_layer_ids = dflare_config.get(
            "target_layer_ids",
            getattr(self.config, "eagle_aux_hidden_state_layer_ids", None),
        )
        if not target_layer_ids:
            raise ValueError("DFlare requires dflare_config.target_layer_ids")
        self.target_layer_ids = list(target_layer_ids)
        self.target_hidden_size = getattr(
            self.config,
            "target_hidden_size",
            self.config.hidden_size,
        )
        self.layer_fusion_weights = nn.Parameter(
            torch.zeros(self.config.num_hidden_layers, len(self.target_layer_ids))
        )
        for draft_idx in range(self.config.num_hidden_layers):
            target_idx = min(
                len(self.target_layer_ids) - 1,
                round(
                    draft_idx
                    * len(self.target_layer_ids)
                    / self.config.num_hidden_layers
                ),
            )
            self.layer_fusion_weights.data[draft_idx, target_idx] = 2.0
        target_embedding_size = int(
            getattr(self.config, "target_embedding_size", self.config.hidden_size)
        )
        target_head_hidden_size = int(
            getattr(self.config, "target_head_hidden_size", target_embedding_size)
        )
        self.input_projection = (
            nn.Linear(target_embedding_size, self.config.hidden_size, bias=False)
            if target_embedding_size != self.config.hidden_size
            else None
        )
        self.output_projection = (
            nn.Linear(
                self.config.hidden_size,
                target_head_hidden_size,
                bias=False,
            )
            if target_head_hidden_size != self.config.hidden_size
            else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = super().embed_input_ids(input_ids) * self.embedding_scale
        if self.input_projection is not None:
            embeddings = self.input_projection(embeddings)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = super().forward(input_ids, positions, input_embeds)
        if self.output_projection is not None:
            hidden_states = self.output_projection(hidden_states)
        return hidden_states

    def _apply_context_rope_override(
        self,
        all_k_normed: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        num_layers, num_context, num_heads, head_dim = all_k_normed.shape
        token_major = all_k_normed.permute(1, 0, 2, 3).reshape(
            num_context,
            num_layers * num_heads,
            head_dim,
        )
        token_major = _apply_angelslim_rope(
            token_major,
            context_positions,
            self._dflare_rope_theta,
            self._dflare_rope_layout,
            self._rope_cos_sin_cache,
        )
        return token_major.view(
            num_context,
            num_layers,
            num_heads,
            head_dim,
        ).permute(1, 0, 2, 3).contiguous()

    def _build_fused_kv_buffers(self) -> None:
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._fused_target_kv_weight: torch.Tensor | None = None
        self._fused_target_kv_bias: torch.Tensor | None = None
        target_projections_are_dense = all(
            isinstance(
                getattr(layer.self_attn.target_k_proj, "weight", None),
                torch.Tensor,
            )
            and isinstance(
                getattr(layer.self_attn.target_v_proj, "weight", None),
                torch.Tensor,
            )
            and layer.self_attn.target_k_proj.weight.dtype
            not in (torch.int32, torch.uint8)
            for layer in self.layers
        )
        if target_projections_are_dense:
            self._fused_target_kv_weight = torch.stack(
                [
                    torch.cat(
                        (
                            attention.target_k_proj.weight.data,
                            attention.target_v_proj.weight.data,
                        ),
                        dim=0,
                    )
                    for attention in layers_attn
                ],
                dim=0,
            ).contiguous()
            if attn0.target_k_proj.bias is not None:
                self._fused_target_kv_bias = torch.stack(
                    [
                        torch.cat(
                            (
                                attention.target_k_proj.bias.data,
                                attention.target_v_proj.bias.data,
                            ),
                            dim=0,
                        )
                        for attention in layers_attn
                    ],
                    dim=0,
                ).contiguous()
        self._k_norm_weights = torch.stack(
            [attention.k_norm.weight.data for attention in layers_attn],
            dim=0,
        ).contiguous()
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        self._attn_layers = [attention.attn for attention in layers_attn]

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = len(self.target_layer_ids) * self.target_hidden_size
        if context_states.shape[-1] != expected:
            raise ValueError(
                f"DFlare expects {expected} target hidden features, "
                f"received {context_states.shape[-1]}"
            )
        target = context_states.view(
            num_ctx,
            len(self.target_layer_ids),
            self.target_hidden_size,
        )
        fusion = torch.softmax(self.layer_fusion_weights, dim=-1).to(target.dtype)
        fused = torch.einsum("nth,lt->nlh", target, fusion)

        if self._fused_target_kv_weight is not None:
            fused_flat = fused.reshape(-1, self.target_hidden_size)
            variance = fused_flat.float().pow(2).mean(dim=-1, keepdim=True)
            normed_flat = (
                fused_flat.float()
                * torch.rsqrt(variance + self._rms_norm_eps)
                * self._hidden_norm_weight.float()
            ).to(fused_flat.dtype)
            normed = normed_flat.view_as(fused)
            all_kv = torch.bmm(
                normed.permute(1, 0, 2),
                self._fused_target_kv_weight.transpose(1, 2),
            )
            if self._fused_target_kv_bias is not None:
                all_kv = all_kv + self._fused_target_kv_bias[:, None, :]
            all_kv = (
                all_kv.view(
                    num_layers,
                    num_ctx,
                    2,
                    num_kv_heads,
                    head_dim,
                )
                .permute(2, 0, 1, 3, 4)
                .contiguous()
            )
            return all_kv[0], all_kv[1]

        all_k = []
        all_v = []
        for layer_idx, layer in enumerate(self.layers):
            context = torch.empty_like(fused[:, layer_idx])
            ops.rms_norm(
                context,
                fused[:, layer_idx],
                self._hidden_norm_weight,
                self._rms_norm_eps,
            )
            key, _ = layer.self_attn.target_k_proj(context)
            value, _ = layer.self_attn.target_v_proj(context)
            all_k.append(key.view(num_ctx, num_kv_heads, head_dim))
            all_v.append(value.view(num_ctx, num_kv_heads, head_dim))
        return torch.stack(all_k), torch.stack(all_v)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def translate(source_weights):
            for name, weight in source_weights:
                name = name.replace(".attention.", ".self_attn.")
                name = name.replace(".k_proj_target", ".target_k_proj")
                name = name.replace(".v_proj_target", ".target_v_proj")
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    name = name.replace(
                        f".{projection}.",
                        f".mlp.{projection}.",
                    )
                yield name, weight

        return super().load_weights(translate(weights))


class DFlareGemma4ForCausalLM(DFlashQwen3ForCausalLM):
    """vLLM draft-model wrapper for AngelSlim DFlare checkpoints."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        draft_hidden_size = getattr(
            self.config,
            "draft_hidden_size",
            self.config.hidden_size,
        )
        original_hidden_size = self.config.hidden_size
        self.config.hidden_size = draft_hidden_size
        try:
            self.model = DFlareGemma4Model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
                start_layer_id=vllm_config.model_config.get_num_layers(
                    vllm_config.parallel_config
                ),
            )
        finally:
            # Keep the concatenated target width visible to the proposer while
            # the constructed draft layers retain their own hidden size.
            self.config.hidden_size = original_hidden_size

        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = vllm_config.model_config.get_vocab_size()
        target_head_hidden_size = int(
            getattr(self.config, "target_head_hidden_size", draft_hidden_size)
        )
        self.lm_head = self._make_lm_head(
            self.config.draft_vocab_size,
            target_head_hidden_size,
            vllm_config,
            prefix,
        )
        self.logits_processor = self._make_logits_processor(
            self.config.draft_vocab_size,
            vllm_config,
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
            self.has_own_lm_head = True
        else:
            self.draft_id_to_target_id = None
            self.has_own_lm_head = False

    @staticmethod
    def _make_lm_head(vocab_size, hidden_size, vllm_config, prefix):
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        return ParallelLMHead(
            vocab_size,
            hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    @staticmethod
    def _make_logits_processor(vocab_size, vllm_config):
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        return LogitsProcessor(
            vocab_size,
            scale=getattr(
                vllm_config.speculative_config.draft_model_config.hf_config,
                "logit_scale",
                1.0,
            ),
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = len(self.model.target_layer_ids) * self.model.target_hidden_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"DFlare expects {expected} concatenated target features, "
                f"received {hidden_states.shape[-1]}"
            )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = []
        head_weights = []
        includes_draft_id_mapping = False
        for name, weight in weights:
            if (
                name in ("d2t", "draft_id_to_target_id")
                or name.endswith(".d2t")
                or name.endswith(".draft_id_to_target_id")
            ):
                if self.draft_id_to_target_id is None:
                    raise ValueError(
                        "Checkpoint contains a reduced-vocabulary d2t mapping "
                        "but draft_vocab_size equals the target vocabulary"
                    )
                head_weights.append(("draft_id_to_target_id", weight))
                includes_draft_id_mapping = True
            elif "lm_head" in name:
                head_weights.append((name, weight))
            else:
                model_weights.append((name, weight))
        if (
            self.draft_id_to_target_id is not None
            and not includes_draft_id_mapping
        ):
            raise ValueError(
                "Reduced-vocabulary DFlare checkpoint is missing its "
                "draft-to-target token mapping"
            )
        self.model.load_weights(model_weights)
        if head_weights:
            AutoWeightsLoader(self).load_weights(head_weights)
        self.model._build_fused_kv_buffers()

