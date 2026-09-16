# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest


def _load_converter():
    path = Path(__file__).parents[2] / "tools" / "convert_angelslim_dflare.py"
    spec = importlib.util.spec_from_file_location("convert_angelslim_dflare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_preserves_slim_architecture():
    converter = _load_converter()
    source = {
        "vocab_size": 262144,
        "hidden_size": 2560,
        "target_hidden_size": 2560,
        "intermediate_size": 10240,
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "block_size": 16,
        "torch_dtype": "bfloat16",
        "model_type": "qwen3",
        "dflare_config": {
            "target_layer_ids": [1, 21, 39],
            "mask_token_id": 4,
        },
    }

    result = converter.build_vllm_config(source)

    assert result["model_type"] == "qwen3"
    assert result["num_hidden_layers"] == 3
    assert result["num_target_layers"] == 3
    assert result["layer_types"] == ["full_attention"] * 3
    assert result["hidden_size"] == 3 * 2560
    assert result["draft_hidden_size"] == 2560
    assert result["dflare_config"]["target_layer_ids"] == [0, 1, 2]
    assert result["eagle_aux_hidden_state_layer_ids"] == [2, 22, 40]
    assert result["num_speculative_tokens"] == 15
    assert result["dflare_config"]["mask_token_id"] == 4
    assert result["dflash_config"] == {
        **result["dflare_config"],
        "use_aux_hidden_state": False,
    }


def test_converter_requires_model_type():
    converter = _load_converter()
    source = {
        "hidden_size": 32,
        "target_hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "dflare_config": {"target_layer_ids": [0], "mask_token_id": 4},
    }

    with pytest.raises(ValueError, match="model_type"):
        converter.build_vllm_config(source)


def test_converter_requires_mask_token_id():
    converter = _load_converter()
    source = {
        "model_type": "qwen3",
        "hidden_size": 32,
        "target_hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "dflare_config": {"target_layer_ids": [0]},
    }

    with pytest.raises(ValueError, match="mask_token_id"):
        converter.build_vllm_config(source)


def test_converter_allows_explicit_target_layer_override():
    converter = _load_converter()
    source = {
        "vocab_size": 128,
        "hidden_size": 32,
        "target_hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "block_size": 4,
        "model_type": "gemma4",
        "dflare_config": {"target_layer_ids": [1, 3, 5], "mask_token_id": 9},
    }

    result = converter.build_vllm_config(source, target_layer_ids=[2, 7])

    assert result["num_target_layers"] == 2
    assert result["hidden_size"] == 64
    assert result["eagle_aux_hidden_state_layer_ids"] == [3, 8]
    assert result["model_type"] == "gemma4"
    assert result["dflare_config"]["mask_token_id"] == 9
