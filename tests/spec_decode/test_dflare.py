from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig


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
