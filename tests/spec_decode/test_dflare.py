# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4_dflare import (
    DFlareGemma4ForCausalLM,
    DFlareGemma4Model,
    _apply_angelslim_rope,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.configs.dflare import (
    apply_dflare_scheduling_alias,
    dflash_config_from_dflare,
)
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig
from vllm.v1.spec_decode.dflare import compact_dflare_context
from vllm.v1.worker.gpu.spec_decode import init_speculator


def _speculators_config():
    return {
        "speculators_model_type": "dflare",
        "speculators_config": {
            "proposal_methods": [{"speculative_tokens": 15}],
            "verifier": {"name_or_path": "google/gemma-4-E4B-it"},
        },
        "transformer_layer_config": {
            "model_type": "qwen3",
            "architectures": ["Gemma4ForCausalLM"],
            "vocab_size": 262144,
            "hidden_size": 23040,
            "draft_hidden_size": 2560,
            "target_hidden_size": 2560,
            "num_hidden_layers": 7,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 10240,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
        },
        "draft_vocab_size": 262144,
        "target_hidden_size": 2560,
        "aux_hidden_state_layer_ids": [1, 6, 11, 16, 21, 26, 31, 36, 39],
        "mask_token_id": 4,
    }


def test_dflare_speculators_config_conversion():
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _speculators_config()
    )

    assert config["architectures"] == ["DFlareDraftModel"]
    assert config["dflare_config"]["mask_token_id"] == 4
    assert config["dflare_config"]["target_layer_ids"] == [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        38,
    ]
    assert config["dflash_config"]["use_aux_hidden_state"] is False


def test_dflare_model_is_registered():
    assert "DFlareDraftModel" in ModelRegistry.get_supported_archs()


def test_angelslim_legacy_rope_layout():
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    actual = _apply_angelslim_rope(
        hidden,
        positions=torch.tensor([1]),
        theta=100.0,
        layout="legacy",
    )
    expected = torch.tensor(
        [
            [
                math.cos(1.0) - 3.0 * math.sin(1.0),
                2.0 * math.cos(1.0) - 4.0 * math.sin(1.0),
                3.0 * math.cos(0.1) + math.sin(0.1),
                4.0 * math.cos(0.1) + 2.0 * math.sin(0.1),
            ]
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("layout", ["legacy", "neox"])
def test_angelslim_rope_cache_matches_computed_frequencies(layout):
    torch.manual_seed(0)
    hidden = torch.randn(2, 3, 8)
    positions = torch.tensor([1, 7])
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    frequencies = torch.einsum(
        "n,d->nd",
        torch.arange(8, dtype=torch.float32),
        inv_freq,
    )
    cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)

    cached = _apply_angelslim_rope(
        hidden,
        positions,
        theta,
        layout,
        cache,
    )
    computed = _apply_angelslim_rope(hidden, positions, theta, layout)

    torch.testing.assert_close(cached, computed)


def test_compact_dflare_context_removes_rejected_rows():
    states = torch.arange(12).view(4, 3)
    positions = torch.tensor([5, 6, 7, 8])
    slots = [
        torch.tensor([10, -1, 12, -1]),
        torch.tensor([20, -1, 22, -1]),
    ]

    compact_states, compact_positions, compact_slots = compact_dflare_context(
        states,
        positions,
        slots,
    )

    torch.testing.assert_close(compact_states, states[[0, 2]])
    torch.testing.assert_close(compact_positions, positions[[0, 2]])
    assert compact_slots is not None
    torch.testing.assert_close(compact_slots[0], torch.tensor([10, 12]))
    torch.testing.assert_close(compact_slots[1], torch.tensor([20, 22]))


class _Projection(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = None

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states @ self.weight.T, None


class _Attention(nn.Module):
    def __init__(self, key_weight: torch.Tensor, value_weight: torch.Tensor):
        super().__init__()
        self.target_k_proj = _Projection(key_weight)
        self.target_v_proj = _Projection(value_weight)


class _Layer(nn.Module):
    def __init__(self, key_weight: torch.Tensor, value_weight: torch.Tensor):
        super().__init__()
        self.self_attn = _Attention(key_weight, value_weight)


class _ContextCacheModel(DFlareGemma4Model):
    test_keys: torch.Tensor
    test_values: torch.Tensor

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.test_keys, self.test_values

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        return all_k

    def _apply_context_rope_override(
        self,
        all_k_normed: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        return all_k_normed + 10


def test_fused_context_kv_matches_layer_loop(monkeypatch):
    model = object.__new__(DFlareGemma4Model)
    nn.Module.__init__(model)
    model.target_layer_ids = [0, 1]
    model.target_hidden_size = 4
    model.layer_fusion_weights = nn.Parameter(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    model._hidden_norm_weight = torch.ones(4)
    model._rms_norm_eps = 1e-6

    torch.manual_seed(0)
    layers = [
        _Layer(torch.randn(4, 4), torch.randn(4, 4)),
        _Layer(torch.randn(4, 4), torch.randn(4, 4)),
    ]
    model.layers = nn.ModuleList(layers)
    model._fused_target_kv_weight = torch.stack(
        [
            torch.cat(
                (
                    layer.self_attn.target_k_proj.weight,
                    layer.self_attn.target_v_proj.weight,
                )
            )
            for layer in layers
        ]
    )
    model._fused_target_kv_bias = None

    def rms_norm(out, hidden_states, weight, epsilon):
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        out.copy_(
            (
                hidden_states.float() * torch.rsqrt(variance + epsilon) * weight.float()
            ).to(hidden_states.dtype)
        )

    monkeypatch.setattr(
        "vllm.model_executor.models.gemma4_dflare.ops.rms_norm",
        rms_norm,
    )
    context_states = torch.randn(3, 8)
    fused = model._project_context_kv(context_states, 3, 2, 2, 2)

    model._fused_target_kv_weight = None
    loop = model._project_context_kv(context_states, 3, 2, 2, 2)

    torch.testing.assert_close(fused[0], loop[0])
    torch.testing.assert_close(fused[1], loop[1])


def test_dflare_weight_names_are_translated(monkeypatch):
    captured = {}

    def capture_weights(self, weights):
        captured["weights"] = list(weights)
        return {"loaded"}

    monkeypatch.setattr(DFlashQwen3Model, "load_weights", capture_weights)
    model = object.__new__(DFlareGemma4Model)
    nn.Module.__init__(model)
    weights = [
        ("layers.0.attention.k_proj_target.weight", torch.zeros(1)),
        ("layers.0.attention.v_proj_target.weight", torch.zeros(1)),
        ("layers.0.gate_proj.weight", torch.zeros(1)),
    ]

    result = model.load_weights(weights)

    assert result == {"loaded"}
    assert [name for name, _ in captured["weights"]] == [
        "layers.0.self_attn.target_k_proj.weight",
        "layers.0.self_attn.target_v_proj.weight",
        "layers.0.mlp.gate_proj.weight",
    ]


def test_dflare_v2_speculator_dispatch(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dflare.speculator.DFlareSpeculator",
        lambda config, device: sentinel,
    )
    config = cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=SimpleNamespace(method="dflare"),
        ),
    )

    assert init_speculator(config, torch.device("cpu")) is sentinel


def test_context_cache_path_is_owned_by_gemma_dflare():
    model = object.__new__(_ContextCacheModel)
    nn.Module.__init__(model)
    model._num_attn_layers = 1
    model._num_kv_heads = 1
    model._head_dim = 2

    keys = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    values = torch.tensor([[[[5.0, 6.0]], [[7.0, 8.0]]]])
    model.test_keys = keys
    model.test_values = values

    calls = []

    class _AttentionImpl:
        def do_kv_cache_update(self, attention, key, value, cache, slots):
            calls.append((attention, key, value, cache, slots))

    attention = SimpleNamespace(impl=_AttentionImpl(), kv_cache=object())
    model._attn_layers = [attention]
    slots = torch.tensor([12, 13])

    model.precompute_and_store_context_kv(
        torch.zeros(2, 4),
        torch.tensor([3, 4]),
        slots,
    )

    assert len(calls) == 1
    assert calls[0][0] is attention
    torch.testing.assert_close(calls[0][1], keys[0] + 10)
    torch.testing.assert_close(calls[0][2], values[0])
    assert calls[0][3] is attention.kv_cache
    torch.testing.assert_close(calls[0][4], slots)


def test_reduced_vocab_requires_draft_id_mapping():
    model = SimpleNamespace(
        draft_id_to_target_id=nn.Parameter(
            torch.zeros(4, dtype=torch.long),
            requires_grad=False,
        ),
        config=SimpleNamespace(draft_vocab_size=4),
        target_vocab_size=8,
    )

    with pytest.raises(ValueError, match="missing.*draft-to-target"):
        DFlareGemma4ForCausalLM.load_weights(
            cast(DFlareGemma4ForCausalLM, model),
            [("lm_head.weight", torch.zeros(4, 4))],
        )


def test_reduced_vocab_rejects_duplicate_target_ids():
    model = SimpleNamespace(
        draft_id_to_target_id=nn.Parameter(
            torch.zeros(4, dtype=torch.long),
            requires_grad=False,
        ),
        config=SimpleNamespace(draft_vocab_size=4),
        target_vocab_size=8,
    )

    with pytest.raises(ValueError, match="unique"):
        DFlareGemma4ForCausalLM.load_weights(
            cast(DFlareGemma4ForCausalLM, model),
            [
                ("d2t", torch.tensor([0, -1, 0, 0])),
                ("lm_head.weight", torch.zeros(4, 4)),
            ],
        )


def test_reduced_vocab_requires_lm_head():
    model = SimpleNamespace(
        draft_id_to_target_id=nn.Parameter(
            torch.zeros(4, dtype=torch.long),
            requires_grad=False,
        ),
        config=SimpleNamespace(draft_vocab_size=4),
        target_vocab_size=8,
    )

    with pytest.raises(ValueError, match="missing lm_head"):
        DFlareGemma4ForCausalLM.load_weights(
            cast(DFlareGemma4ForCausalLM, model),
            [("d2t", torch.tensor([0, 0, 0, 0]))],
        )


def test_dflare_scheduling_alias_is_shared():
    dflare_config = {
        "mask_token_id": 7,
        "target_layer_ids": [0, 1],
        "causal": False,
    }
    aliased = dflash_config_from_dflare(dflare_config)
    assert aliased["use_aux_hidden_state"] is False
    assert aliased["mask_token_id"] == 7
    assert dflare_config.get("use_aux_hidden_state") is None

    config = {"dflare_config": dflare_config}
    apply_dflare_scheduling_alias(config)
    assert config["dflash_config"] == aliased
    config["dflash_config"]["mask_token_id"] = 99
    apply_dflare_scheduling_alias(config)
    assert config["dflash_config"]["mask_token_id"] == 99


def test_causal_lm_keeps_concatenated_hidden_size(monkeypatch):
    captured = {}

    class _FakeDraftModel:
        def __init__(self, *, vllm_config, prefix: str = "", start_layer_id: int = 0):
            hf_config = vllm_config.speculative_config.draft_model_config.hf_config
            captured["hidden_size"] = hf_config.hidden_size
            captured["draft_hidden_size"] = hf_config.draft_hidden_size
            self.target_layer_ids = [0, 1]
            self.target_hidden_size = 4

    monkeypatch.setattr(
        "vllm.model_executor.models.gemma4_dflare.DFlareGemma4Model",
        _FakeDraftModel,
    )
    monkeypatch.setattr(
        DFlareGemma4ForCausalLM,
        "_make_lm_head",
        staticmethod(lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        DFlareGemma4ForCausalLM,
        "_make_logits_processor",
        staticmethod(lambda *args, **kwargs: object()),
    )
    hf_config = SimpleNamespace(
        hidden_size=12,
        draft_hidden_size=4,
        draft_vocab_size=8,
        target_head_hidden_size=4,
        logit_scale=1.0,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        ),
        model_config=SimpleNamespace(
            get_num_layers=lambda _parallel: 1,
            get_vocab_size=lambda: 8,
        ),
        parallel_config=object(),
    )

    DFlareGemma4ForCausalLM(vllm_config=cast(VllmConfig, vllm_config))

    assert captured["hidden_size"] == 12
    assert captured["draft_hidden_size"] == 4
    assert hf_config.hidden_size == 12


def test_dflare_pipeline_converts_compacts_and_fuses(monkeypatch):
    path = Path(__file__).parents[2] / "tools" / "convert_angelslim_dflare.py"
    spec = importlib.util.spec_from_file_location("convert_angelslim_dflare", path)
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    converted = converter.build_vllm_config(
        {
            "model_type": "qwen3",
            "vocab_size": 16,
            "hidden_size": 4,
            "target_hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "block_size": 4,
            "dflare_config": {
                "target_layer_ids": [0, 1],
                "mask_token_id": 7,
            },
        }
    )
    serving = SpeculatorsConfig.extract_transformers_pre_trained_config(
        {
            "speculators_model_type": "dflare",
            "speculators_config": {
                "proposal_methods": [{"speculative_tokens": 3}],
                "verifier": {"name_or_path": "google/gemma-4-E4B-it"},
            },
            "transformer_layer_config": {
                "model_type": converted["model_type"],
                "architectures": ["Gemma4ForCausalLM"],
                "vocab_size": converted["vocab_size"],
                "hidden_size": converted["hidden_size"],
                "draft_hidden_size": converted["draft_hidden_size"],
                "target_hidden_size": converted["target_hidden_size"],
                "num_hidden_layers": converted["num_hidden_layers"],
                "num_attention_heads": converted["num_attention_heads"],
                "num_key_value_heads": converted["num_key_value_heads"],
                "intermediate_size": converted["intermediate_size"],
                "head_dim": converted["head_dim"],
                "rms_norm_eps": converted["rms_norm_eps"],
            },
            "draft_vocab_size": converted["draft_vocab_size"],
            "target_hidden_size": converted["target_hidden_size"],
            "aux_hidden_state_layer_ids": converted["eagle_aux_hidden_state_layer_ids"],
            "mask_token_id": converted["dflare_config"]["mask_token_id"],
        }
    )

    assert converted["model_type"] == "qwen3"
    assert converted["hidden_size"] == 8
    assert converted["draft_hidden_size"] == 4
    assert converted["dflare_config"]["mask_token_id"] == 7
    assert serving["dflash_config"]["use_aux_hidden_state"] is False
    assert serving["dflare_config"]["mask_token_id"] == 7

    layer0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    layer1 = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    rejected = torch.zeros(2, 8)
    states = torch.stack(
        [
            torch.cat((layer0[0], layer1[0])),
            rejected[0],
            torch.cat((layer0[1], layer1[1])),
            rejected[1],
        ]
    )
    positions = torch.tensor([3, 4, 5, 6])
    slots = [torch.tensor([10, -1, 12, -1])]
    compact_states, compact_positions, compact_slots = compact_dflare_context(
        states,
        positions,
        slots,
    )
    torch.testing.assert_close(
        compact_states,
        torch.stack(
            [
                torch.cat((layer0[0], layer1[0])),
                torch.cat((layer0[1], layer1[1])),
            ]
        ),
    )
    torch.testing.assert_close(compact_positions, torch.tensor([3, 5]))
    assert compact_slots is not None
    torch.testing.assert_close(compact_slots[0], torch.tensor([10, 12]))

    model = object.__new__(DFlareGemma4Model)
    nn.Module.__init__(model)
    model.target_layer_ids = converted["dflare_config"]["target_layer_ids"]
    model.target_hidden_size = converted["target_hidden_size"]
    model.layer_fusion_weights = nn.Parameter(
        torch.tensor([[1.0e6, -1.0e6], [-1.0e6, 1.0e6]])
    )
    identity = torch.eye(4)
    doubled = 2.0 * identity
    model.layers = nn.ModuleList(
        [
            _Layer(identity.clone(), doubled.clone()),
            _Layer(identity.clone(), doubled.clone()),
        ]
    )
    model._fused_target_kv_weight = None
    model._fused_target_kv_bias = None
    model._hidden_norm_weight = torch.ones(4)
    model._rms_norm_eps = 1e-6

    def identity_norm(out, hidden_states, weight, epsilon):
        out.copy_(hidden_states)

    monkeypatch.setattr(
        "vllm.model_executor.models.gemma4_dflare.ops.rms_norm",
        identity_norm,
    )
    keys, values = model._project_context_kv(compact_states, 2, 2, 2, 2)

    torch.testing.assert_close(
        keys[0],
        layer0.view(2, 2, 2),
    )
    torch.testing.assert_close(
        keys[1],
        layer1.view(2, 2, 2),
    )
    torch.testing.assert_close(values[0], 2.0 * keys[0])
    torch.testing.assert_close(values[1], 2.0 * keys[1])
