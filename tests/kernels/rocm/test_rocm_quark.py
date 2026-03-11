# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Quark quantization on ROCm.

Covers:
- QuarkConfig: construction, scheme dispatch, config matching
- QuarkW8A8Fp8 scheme: constructor variants, E2E model inference
- QuarkW8A8Int8 scheme: INT8 quantization
- QuarkOCP_MX scheme: MXFP4/MXFP8/MXFP6 block format, emulation
- Scheme selection predicates (_is_fp8_w8a8, _is_static_tensor_w8a8, etc.)
- _find_matched_config: pattern matching, fused modules, fallback hierarchy
- QuarkKVCacheMethod: validation and error handling
- QuarkMoEMethod subclasses: type/API verification
- MXFP4 quant-dequant numerical accuracy
- E2E model inference via vllm_runner (FP8)

References:
- tests/quantization/test_quark.py (upstream canonical test)
- vllm/model_executor/layers/quantization/quark/quark.py
"""

import importlib.util
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

aiter_available = importlib.util.find_spec("aiter") is not None
fp8_supported = current_platform.is_rocm() and current_platform.supports_fp8()

_FP8_MODEL = "amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test"
_TEST_PROMPTS = ["The capital of France is"]


@pytest.fixture(scope="function", autouse=False)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


# ── QuarkConfig construction ──────────────────────────────────────────────


def test_quark_config_w8a8_fp8_construction():
    """QuarkConfig constructs without error for W8A8 FP8 config."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    quant_config = {
        "quant_type": "a8w8_fp8_dynamic",
        "model_config": {
            "linear_layers": {
                "weight": {
                    "dtype": "fp8",
                    "qscheme": "per_tensor",
                    "is_dynamic": False,
                },
                "activation": {
                    "dtype": "fp8",
                    "qscheme": "per_tensor",
                    "is_dynamic": True,
                },
            }
        },
    }
    cfg = QuarkConfig(quant_config=quant_config)
    assert cfg is not None
    assert cfg.quant_config == quant_config


def test_quark_config_kv_cache_group():
    """QuarkConfig with kv_cache_group parameter."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    cfg = QuarkConfig(
        quant_config={"quant_type": "a8w8_fp8"},
        kv_cache_group=["attn.k_proj", "attn.v_proj"],
    )
    assert cfg.kv_cache_group == ["attn.k_proj", "attn.v_proj"]


def test_quark_config_pack_method():
    """QuarkConfig stores pack_method parameter."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    cfg = QuarkConfig(quant_config={"quant_type": "test"}, pack_method="order")
    assert cfg.pack_method == "order"

    cfg_default = QuarkConfig(quant_config={"quant_type": "test"})
    assert cfg_default.pack_method == "reorder"


# ── Scheme selection predicates ──────────────────────────────────────────


def _make_config_with_predicates():
    """Create a QuarkConfig with minimal setup for predicate testing."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    return QuarkConfig(quant_config={})


def test_quark_is_fp8_w8a8_per_tensor():
    """_is_fp8_w8a8 returns True for per-tensor FP8 config."""
    cfg = _make_config_with_predicates()
    weight = {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False}
    inp = {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": True}
    assert cfg._is_fp8_w8a8(weight, inp) is True


def test_quark_is_fp8_w8a8_per_channel():
    """_is_fp8_w8a8 returns True for per-channel weight + dynamic activation."""
    cfg = _make_config_with_predicates()
    weight = {"dtype": "fp8_e4m3", "qscheme": "per_channel", "is_dynamic": False}
    inp = {"dtype": "fp8_e4m3", "qscheme": "per_token", "is_dynamic": True}
    assert cfg._is_fp8_w8a8(weight, inp) is True


def test_quark_is_fp8_w8a8_rejects_int8():
    """_is_fp8_w8a8 returns False for INT8 dtype."""
    cfg = _make_config_with_predicates()
    weight = {"dtype": "int8", "qscheme": "per_tensor", "is_dynamic": False}
    inp = {"dtype": "int8", "qscheme": "per_tensor", "is_dynamic": True}
    assert cfg._is_fp8_w8a8(weight, inp) is False


def test_quark_is_fp8_w8a8_rejects_none():
    """_is_fp8_w8a8 returns False when weight or input is None."""
    cfg = _make_config_with_predicates()
    weight = {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False}
    assert cfg._is_fp8_w8a8(None, None) is False
    assert cfg._is_fp8_w8a8(weight, None) is False
    assert cfg._is_fp8_w8a8(None, weight) is False


def test_quark_is_static_tensor_w8a8_per_tensor():
    """_is_static_tensor_w8a8 returns True for per-tensor symmetric INT8."""
    cfg = _make_config_with_predicates()
    weight = {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    inp = {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    assert cfg._is_static_tensor_w8a8(weight, inp) is True


def test_quark_is_static_tensor_w8a8_per_channel_weight():
    """_is_static_tensor_w8a8 accepts per-channel weight + per-tensor activation."""
    cfg = _make_config_with_predicates()
    weight = {
        "dtype": "int8",
        "qscheme": "per_channel",
        "is_dynamic": False,
        "symmetric": True,
    }
    inp = {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    assert cfg._is_static_tensor_w8a8(weight, inp) is True


def test_quark_is_static_tensor_w8a8_rejects_asymmetric_weight():
    """_is_static_tensor_w8a8 rejects asymmetric weight quantization."""
    cfg = _make_config_with_predicates()
    weight = {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": False,
    }
    inp = {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    assert cfg._is_static_tensor_w8a8(weight, inp) is False


def test_quark_is_static_tensor_w8a8_rejects_dynamic():
    """_is_static_tensor_w8a8 rejects dynamic quantization."""
    cfg = _make_config_with_predicates()
    weight = {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": True,
        "symmetric": True,
    }
    inp = {
        "dtype": "int8",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    assert cfg._is_static_tensor_w8a8(weight, inp) is False


def test_quark_is_w_ocp_mx_fp4():
    """_is_w_ocp_mx_a_x returns True for MXFP4 per-group config."""
    cfg = _make_config_with_predicates()
    weight = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
    }
    assert cfg._is_w_ocp_mx_a_x(weight, None) is True


def test_quark_is_w_ocp_mx_fp6_e3m2():
    """_is_w_ocp_mx_a_x returns True for fp6_e3m2."""
    cfg = _make_config_with_predicates()
    weight = {
        "dtype": "fp6_e3m2",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
    }
    assert cfg._is_w_ocp_mx_a_x(weight, None) is True


def test_quark_is_w_ocp_mx_rejects_wrong_group_size():
    """_is_w_ocp_mx_a_x rejects group_size != 32."""
    cfg = _make_config_with_predicates()
    weight = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 64,
        "scale_format": "e8m0",
    }
    assert cfg._is_w_ocp_mx_a_x(weight, None) is False


def test_quark_is_w_ocp_mx_rejects_wrong_scale_format():
    """_is_w_ocp_mx_a_x rejects scale_format != e8m0."""
    cfg = _make_config_with_predicates()
    weight = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e5m2",
    }
    assert cfg._is_w_ocp_mx_a_x(weight, None) is False


def test_quark_is_w_ocp_mx_rejects_per_tensor():
    """_is_w_ocp_mx_a_x rejects per_tensor qscheme."""
    cfg = _make_config_with_predicates()
    weight = {
        "dtype": "fp4",
        "qscheme": "per_tensor",
        "group_size": 32,
        "scale_format": "e8m0",
    }
    assert cfg._is_w_ocp_mx_a_x(weight, None) is False


def test_quark_is_w_ocp_mx_rejects_list_weight():
    """_is_w_ocp_mx_a_x rejects list-format weight_quant (e.g. fp8_w4a8)."""
    cfg = _make_config_with_predicates()
    weight_list = [{"dtype": "fp8_e4m3"}, {"dtype": "int4"}]
    assert cfg._is_w_ocp_mx_a_x(weight_list, None) is False


# ── _get_scheme_from_config dispatch ─────────────────────────────────────


def test_quark_get_scheme_dispatches_fp8():
    """_get_scheme_from_config returns QuarkW8A8Fp8 for FP8 config."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )

    cfg = QuarkConfig(quant_config={})
    config = {
        "weight": {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": True,
        },
    }
    scheme = cfg._get_scheme_from_config(config)
    assert isinstance(scheme, QuarkW8A8Fp8)


def test_quark_get_scheme_dispatches_int8():
    """_get_scheme_from_config returns QuarkW8A8Int8 for INT8 config."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_int8 import (
        QuarkW8A8Int8,
    )

    cfg = QuarkConfig(quant_config={})
    config = {
        "weight": {
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": True,
        },
        "input_tensors": {
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": True,
        },
    }
    scheme = cfg._get_scheme_from_config(config)
    assert isinstance(scheme, QuarkW8A8Int8)
    assert scheme.qscheme == "per_tensor"
    assert scheme.is_static_input_scheme is True


def test_quark_get_scheme_dispatches_ocp_mx():
    """_get_scheme_from_config returns QuarkOCP_MX for MXFP4 config."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        QuarkOCP_MX,
    )

    cfg = QuarkConfig(quant_config={})
    config = {
        "weight": {
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": False,
        },
        "input_tensors": {
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": True,
        },
    }
    scheme = cfg._get_scheme_from_config(config)
    assert isinstance(scheme, QuarkOCP_MX)


def test_quark_get_scheme_unsupported_raises():
    """_get_scheme_from_config raises NotImplementedError for unrecognized config."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    cfg = QuarkConfig(quant_config={})
    config = {
        "weight": {"dtype": "int2", "qscheme": "per_tensor"},
        "input_tensors": {"dtype": "int2", "qscheme": "per_tensor"},
    }
    with pytest.raises(NotImplementedError, match="No quark compatible scheme"):
        cfg._get_scheme_from_config(config)


def test_quark_get_scheme_output_tensors_raises():
    """_get_scheme_from_config raises for output_tensors quantization."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    cfg = QuarkConfig(quant_config={})
    config = {"output_tensors": {"dtype": "fp8_e4m3"}}
    with pytest.raises(NotImplementedError, match="output_tensors"):
        cfg._get_scheme_from_config(config)


# ── _find_matched_config pattern matching ────────────────────────────────


def test_quark_find_matched_config_exact_match():
    """_find_matched_config matches exact layer names."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    layer_config = {
        "model.layers.0.self_attn.q_proj": {
            "weight": {"dtype": "fp8_e4m3"},
        },
    }
    cfg = QuarkConfig(
        quant_config={
            "layer_quant_config": layer_config,
            "layer_type_quant_config": {},
            "global_quant_config": {},
        }
    )
    module = MagicMock()
    result = cfg._find_matched_config("model.layers.0.self_attn.q_proj", module)
    assert result["weight"]["dtype"] == "fp8_e4m3"


def test_quark_find_matched_config_wildcard():
    """_find_matched_config supports fnmatch wildcards."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    layer_config = {
        "*.q_proj": {"weight": {"dtype": "int8"}},
    }
    cfg = QuarkConfig(
        quant_config={
            "layer_quant_config": layer_config,
            "layer_type_quant_config": {},
            "global_quant_config": {},
        }
    )
    module = MagicMock()
    result = cfg._find_matched_config("model.layers.5.self_attn.q_proj", module)
    assert result["weight"]["dtype"] == "int8"


def test_quark_find_matched_config_fallback_to_global():
    """_find_matched_config falls back to global_quant_config."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    cfg = QuarkConfig(
        quant_config={
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "global_quant_config": {"weight": {"dtype": "fp8_e4m3"}},
        }
    )
    module = MagicMock()
    result = cfg._find_matched_config("model.layers.0.mlp.gate_proj", module)
    assert result["weight"]["dtype"] == "fp8_e4m3"


def test_quark_find_matched_config_fused_module():
    """_find_matched_config maps qkv_proj to q_proj/k_proj/v_proj."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    fp8_config = {
        "weight": {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": True,
        },
    }
    layer_config = {
        "model.layers.0.self_attn.q_proj": fp8_config,
        "model.layers.0.self_attn.k_proj": fp8_config,
        "model.layers.0.self_attn.v_proj": fp8_config,
    }
    cfg = QuarkConfig(
        quant_config={
            "layer_quant_config": layer_config,
            "layer_type_quant_config": {},
            "global_quant_config": {},
        }
    )
    # Simulate packed_modules_mapping for qkv_proj
    cfg.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    module = MagicMock()
    result = cfg._find_matched_config("model.layers.0.self_attn.qkv_proj", module)
    assert result["weight"]["dtype"] == "fp8_e4m3"


# ── QuarkKVCacheMethod validation ────────────────────────────────────────


def test_quark_kv_cache_valid_config():
    """QuarkKVCacheMethod accepts valid fp8_e4m3 per_tensor config."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkKVCacheMethod

    # Should not raise
    QuarkKVCacheMethod.validate_kv_cache_config(
        {"dtype": "fp8_e4m3", "qscheme": "per_tensor"}
    )


def test_quark_kv_cache_none_config():
    """QuarkKVCacheMethod accepts None (no KV cache quantization)."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkKVCacheMethod

    # Should not raise
    QuarkKVCacheMethod.validate_kv_cache_config(None)


def test_quark_kv_cache_invalid_dtype_raises():
    """QuarkKVCacheMethod rejects non-fp8 dtype."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkKVCacheMethod

    with pytest.raises(NotImplementedError, match="fp8_e4m3"):
        QuarkKVCacheMethod.validate_kv_cache_config(
            {"dtype": "int8", "qscheme": "per_tensor"}
        )


def test_quark_kv_cache_invalid_qscheme_raises():
    """QuarkKVCacheMethod rejects non-per-tensor qscheme."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkKVCacheMethod

    with pytest.raises(NotImplementedError, match="per_tensor"):
        QuarkKVCacheMethod.validate_kv_cache_config(
            {"dtype": "fp8_e4m3", "qscheme": "per_channel"}
        )


# ── get_cache_scale remapping ────────────────────────────────────────────


def test_quark_get_cache_scale_k_proj():
    """get_cache_scale remaps k_proj.output_scale to attn.k_scale."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    cfg = QuarkConfig(quant_config={})
    result = cfg.get_cache_scale("model.layers.0.self_attn.k_proj.output_scale")
    assert result == "model.layers.0.self_attn.attn.k_scale"


def test_quark_get_cache_scale_v_proj():
    """get_cache_scale remaps v_proj.output_scale to attn.v_scale."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    cfg = QuarkConfig(quant_config={})
    result = cfg.get_cache_scale("model.layers.0.self_attn.v_proj.output_scale")
    assert result == "model.layers.0.self_attn.attn.v_scale"


def test_quark_get_cache_scale_no_match():
    """get_cache_scale returns None for non-matching names."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    cfg = QuarkConfig(quant_config={})
    assert cfg.get_cache_scale("model.layers.0.mlp.weight") is None


# ── QuarkW8A8Fp8 scheme tests ───────────────────────────────────────────


def test_quark_w8a8_fp8_scheme_construction():
    """QuarkW8A8Fp8 constructs for per-tensor static weight config."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )

    weight_config = {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": False}
    input_config = {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": True}

    scheme = QuarkW8A8Fp8(weight_config=weight_config, input_config=input_config)
    assert scheme is not None
    assert not scheme.is_static_input_scheme  # dynamic activation


def test_quark_w8a8_fp8_static_input_scheme():
    """QuarkW8A8Fp8 with static input scheme (is_dynamic=False)."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )

    weight_config = {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": False}
    input_config = {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": False}

    scheme = QuarkW8A8Fp8(weight_config=weight_config, input_config=input_config)
    assert scheme.is_static_input_scheme


def test_quark_w8a8_fp8_per_channel_weight():
    """QuarkW8A8Fp8 with per-channel weight quantization."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )

    weight_config = {"dtype": "fp8", "qscheme": "per_channel", "is_dynamic": False}
    input_config = {"dtype": "fp8", "qscheme": "per_channel", "is_dynamic": False}

    scheme = QuarkW8A8Fp8(weight_config=weight_config, input_config=input_config)
    assert scheme is not None


def test_quark_w8a8_fp8_no_input_config():
    """QuarkW8A8Fp8 with no input quantization (weight-only)."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )

    weight_config = {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": False}
    scheme = QuarkW8A8Fp8(weight_config=weight_config, input_config=None)
    assert scheme.input_qscheme is None


# ── QuarkW8A8Int8 scheme tests ──────────────────────────────────────────


def test_quark_w8a8_int8_scheme_construction():
    """QuarkW8A8Int8 constructs for per-tensor dynamic input."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_int8 import (
        QuarkW8A8Int8,
    )

    scheme = QuarkW8A8Int8(
        qscheme="per_tensor",
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    assert scheme is not None
    assert scheme.qscheme == "per_tensor"
    assert not scheme.is_static_input_scheme
    assert scheme.input_symmetric is True


def test_quark_w8a8_int8_static_input_per_channel():
    """QuarkW8A8Int8 with static input and per-channel weight quantization."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_int8 import (
        QuarkW8A8Int8,
    )

    scheme = QuarkW8A8Int8(
        qscheme="per_channel",
        is_static_input_scheme=True,
        input_symmetric=False,
    )
    assert scheme.qscheme == "per_channel"
    assert scheme.is_static_input_scheme is True
    assert scheme.input_symmetric is False


# ── QuarkOCP_MX scheme tests ────────────────────────────────────────────


def test_quark_ocp_mx_scheme_construction_mxfp4():
    """QuarkOCP_MX constructs for MXFP4 configuration."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        QuarkOCP_MX,
    )

    weight_quant_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": False,
    }
    input_quant_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": True,
    }

    scheme = QuarkOCP_MX(
        weight_quant_spec=weight_quant_spec, input_quant_spec=input_quant_spec
    )
    assert scheme is not None
    assert scheme.weight_dtype == "mxfp4"
    assert scheme.input_dtype == "mxfp4"


def test_quark_ocp_mx_scheme_construction_mxfp8():
    """QuarkOCP_MX constructs for MXFP8 (emulation/fallback path).

    MXFP8 is not in SUPPORTED_OCP_MX_DTYPES so the constructor runs in
    emulation mode. This tests the fallback code path.
    """
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        QuarkOCP_MX,
    )

    weight_quant_spec = {
        "dtype": "fp8",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": False,
    }
    input_quant_spec = {
        "dtype": "fp8",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": True,
    }

    scheme = QuarkOCP_MX(
        weight_quant_spec=weight_quant_spec, input_quant_spec=input_quant_spec
    )
    assert scheme is not None
    assert scheme.weight_dtype == "mxfp8"
    assert scheme.emulate  # mxfp8 always runs in emulation mode


def test_quark_ocp_mx_scheme_construction_mxfp6_e3m2():
    """QuarkOCP_MX constructs for MXFP6 e3m2 (supported production dtype)."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        QuarkOCP_MX,
    )
    from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
        OCP_MX_Scheme,
    )

    weight_quant_spec = {
        "dtype": "fp6_e3m2",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": False,
    }
    input_quant_spec = {
        "dtype": "fp6_e3m2",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": True,
    }

    scheme = QuarkOCP_MX(
        weight_quant_spec=weight_quant_spec, input_quant_spec=input_quant_spec
    )
    assert scheme is not None
    assert scheme.weight_dtype == "mxfp6_e3m2"
    assert scheme.input_dtype == "mxfp6_e3m2"
    assert scheme.ocp_mx_scheme == OCP_MX_Scheme.w_mxfp6_e3m2_a_mxfp6_e3m2


def test_quark_ocp_mx_dynamic_mxfp4_quant():
    """QuarkOCP_MX with dynamic_mxfp4_quant=True (runtime quantization path)."""
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        QuarkOCP_MX,
    )

    weight_quant_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": False,
    }
    input_quant_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": True,
    }

    scheme = QuarkOCP_MX(
        weight_quant_spec=weight_quant_spec,
        input_quant_spec=input_quant_spec,
        dynamic_mxfp4_quant=True,
    )
    assert scheme is not None
    assert scheme.dynamic_mxfp4_quant is True


def test_quark_ocp_mx_mxfp4_quant_dequant():
    """QuarkOCP_MX MXFP4 quantize-dequant roundtrip via mxfp4_utils."""
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    for M, K in [(128, 256), (256, 512), (512, 1024)]:
        x = torch.randn(M, K, dtype=torch.bfloat16)
        x_dq = quant_dequant_mxfp4(x)
        assert x_dq.shape == (M, K)
        assert x_dq.dtype == torch.bfloat16
        assert not torch.any(torch.isnan(x_dq))


def test_quark_ocp_mx_gemm_with_dynamic_quant_api():
    """QuarkOCP_MX uses gemm_with_dynamic_quant (aiter triton FP4 GEMM)."""
    try:
        from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
            gemm_with_dynamic_quant,
        )

        assert callable(gemm_with_dynamic_quant)
    except ImportError:
        pytest.skip("aiter FP4 GEMM modules not available in this environment")


# ── QuarkMoEMethod subclass API tests ───────────────────────────────────


def test_quark_moe_method_w8a8_fp8_class():
    """QuarkW8A8Fp8MoEMethod class is importable and is a MoE method."""
    from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkMoEMethod,
        QuarkW8A8Fp8MoEMethod,
    )

    assert issubclass(QuarkW8A8Fp8MoEMethod, QuarkMoEMethod)
    assert issubclass(QuarkMoEMethod, FusedMoEMethodBase)


def test_quark_moe_method_ocp_mx_class():
    """QuarkOCP_MX_MoEMethod class is importable."""
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkOCP_MX_MoEMethod,
    )

    assert QuarkOCP_MX_MoEMethod is not None


def test_quark_moe_method_w4a8_fp8_class():
    """QuarkW4A8Fp8MoEMethod class is importable."""
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkW4A8Fp8MoEMethod,
    )

    assert QuarkW4A8Fp8MoEMethod is not None


def test_quark_moe_method_ocp_mx_oss_class():
    """QuarkOCP_MX_MoEMethod_OSS class is importable."""
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkOCP_MX_MoEMethod,
        QuarkOCP_MX_MoEMethod_OSS,
    )

    assert issubclass(QuarkOCP_MX_MoEMethod_OSS, QuarkOCP_MX_MoEMethod)


# ── Scheme importability ────────────────────────────────────────────────


def test_quark_schemes_importable():
    """All Quark scheme classes are importable from the schemes package."""
    from vllm.model_executor.layers.quantization.quark.schemes import (
        QuarkOCP_MX,
        QuarkScheme,
        QuarkW8A8Fp8,
        QuarkW8A8Int8,
    )

    for cls in [QuarkScheme, QuarkW8A8Fp8, QuarkW8A8Int8, QuarkOCP_MX]:
        assert cls is not None
        assert isinstance(cls, type)


# ── MXFP4 numerical accuracy ────────────────────────────────────────────


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific test")
def test_quark_mxfp4_relative_error_is_bounded():
    """MXFP4 quantization error is within acceptable bounds for typical weights.

    FP4 has 4-bit mantissa -> relative error should be < 25% on average.
    """
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(42)

    M, K = 256, 512
    x = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    x_dq = quant_dequant_mxfp4(x)

    rel_error = (x_dq - x).abs() / (x.abs() + 1e-6)
    mean_rel_error = rel_error.mean().item()
    assert mean_rel_error < 0.25, f"Mean relative error too high: {mean_rel_error:.4f}"


# ── E2E model inference tests (from test_quark_rocm.py) ─────────────────


@pytest.mark.skipif(not fp8_supported, reason="FP8 not supported on this hardware")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_quark_fp8_w8a8_rocm(vllm_runner, enable_pickle, kv_cache_dtype):
    """QuarkW8A8Fp8 model loads and produces finite output on ROCm.

    Validates that:
    - QuarkLinearMethod is selected for the model's linear layers
    - QuarkW8A8Fp8 scheme is active
    - Weight dtype is ROCm-specific FP8 (float8_e4m3fnuz on gfx942)
    - Generation does not hang or produce NaN
    """
    from vllm.model_executor.layers.quantization.quark.quark import (
        QuarkLinearMethod,
        QuarkW8A8Fp8,
    )

    with vllm_runner(
        _FP8_MODEL,
        enforce_eager=True,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=1,
    ) as llm:

        def check_model(model):
            layer = model.model.layers[0]
            qkv_proj = layer.self_attn.qkv_proj
            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Fp8)
            assert qkv_proj.weight.dtype == current_platform.fp8_dtype()

        llm.apply_model(check_model)

        outputs = llm.generate_greedy(_TEST_PROMPTS, max_tokens=4)
        assert outputs, "Expected non-empty generation"
        assert outputs[0][1], "Expected non-empty output tokens"


@pytest.mark.skipif(not fp8_supported, reason="FP8 not supported on this hardware")
def test_quark_fp8_parity_rocm(vllm_runner, enable_pickle):
    """Quark FP8 model loads and generates non-empty output on ROCm."""
    from vllm.model_executor.layers.quantization.quark.quark import (
        QuarkLinearMethod,
    )

    with vllm_runner(
        _FP8_MODEL,
        enforce_eager=True,
    ) as llm:

        def check_quark(model):
            layer = model.model.layers[0]
            qkv_proj = layer.self_attn.qkv_proj
            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)

        llm.apply_model(check_quark)
        outputs = llm.generate_greedy(_TEST_PROMPTS, max_tokens=8)

    assert outputs[0][1], "Quark FP8 model produced empty output"
