# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest


def _load_converter():
    path = Path(__file__).parents[2] / "tools" / "convert_angelslim_dflash.py"
    spec = importlib.util.spec_from_file_location("convert_angelslim_dflash", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_builds_qwen3_dflash_contract():
    converter = _load_converter()
    result = converter.build_vllm_config(
        {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope_parameters": {"rope_theta": 1000000},
            "block_size": 16,
            "dflash_config": {
                "mask_token_id": 151669,
                "target_layer_ids": [1, 9, 17, 25, 33],
            },
        }
    )

    assert result["architectures"] == ["DFlashDraftModel"]
    assert result["rope_theta"] == 1000000
    assert result["num_speculative_tokens"] == 15
    assert result["dflash_config"]["target_layer_ids"] == [0, 1, 2, 3, 4]
    assert result["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]


def test_converter_requires_mask_token():
    converter = _load_converter()
    with pytest.raises(ValueError, match="mask_token_id"):
        converter.build_vllm_config(
            {
                "vocab_size": 8,
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "dflash_config": {"target_layer_ids": [0]},
            }
        )
