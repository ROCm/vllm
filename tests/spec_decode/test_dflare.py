import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.gemma4_dflare import (
    DFlareGemma4ForCausalLM,
    DFlareGemma4Model,
    _apply_angelslim_rope,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig
from vllm.v1.spec_decode.utils import compact_dflash_context


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


def test_compact_dflash_context_removes_rejected_rows():
    states = torch.arange(12).view(4, 3)
    positions = torch.tensor([5, 6, 7, 8])
    slots = [
        torch.tensor([10, -1, 12, -1]),
        torch.tensor([20, -1, 22, -1]),
    ]

    compact_states, compact_positions, compact_slots = compact_dflash_context(
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


class _Layer(nn.Module):
    def __init__(self, key_weight: torch.Tensor, value_weight: torch.Tensor):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.target_k_proj = _Projection(key_weight)
        self.self_attn.target_v_proj = _Projection(value_weight)


def test_fused_context_kv_matches_layer_loop(monkeypatch):
    model = object.__new__(DFlareGemma4Model)
    nn.Module.__init__(model)
    model.target_layer_ids = [0, 1]
    model.target_hidden_size = 4
    model.layer_fusion_weights = nn.Parameter(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    )
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
                hidden_states.float()
                * torch.rsqrt(variance + epsilon)
                * weight.float()
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


def test_reduced_vocab_requires_draft_id_mapping():
    model = SimpleNamespace(
        draft_id_to_target_id=nn.Parameter(
            torch.zeros(4, dtype=torch.long),
            requires_grad=False,
        )
    )

    with pytest.raises(ValueError, match="missing.*draft-to-target"):
        DFlareGemma4ForCausalLM.load_weights(
            model,
            [("lm_head.weight", torch.zeros(4, 4))],
        )
