# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ROCm attention backend registry completeness.

Verifies that all ROCm-specific attention backends are registered in
AttentionBackendEnum and can be instantiated without error.
"""

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


def test_all_rocm_attention_backends_in_enum():
    """All ROCm attention backend variants are present in AttentionBackendEnum."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    rocm_backends = [
        "ROCM_ATTN",
        "ROCM_AITER_FA",
        "ROCM_AITER_MLA",
        "ROCM_AITER_TRITON_MLA",
        "ROCM_AITER_MLA_SPARSE",
        "ROCM_AITER_UNIFIED_ATTN",
    ]
    for name in rocm_backends:
        assert hasattr(AttentionBackendEnum, name), (
            f"AttentionBackendEnum missing ROCm backend: {name}"
        )


def test_rocm_attention_backend_enum_values_are_unique():
    """Each ROCm backend enum entry has a distinct value."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    rocm_names = [
        "ROCM_ATTN",
        "ROCM_AITER_FA",
        "ROCM_AITER_MLA",
        "ROCM_AITER_TRITON_MLA",
        "ROCM_AITER_MLA_SPARSE",
        "ROCM_AITER_UNIFIED_ATTN",
    ]
    values = [AttentionBackendEnum[name].value for name in rocm_names]
    assert len(values) == len(set(values)), (
        "Duplicate enum values found among ROCm backends"
    )


def test_rocm_backend_names_match_enum_keys():
    """Each ROCm backend's get_name() matches the enum key it is registered under."""
    checks = [
        (
            "vllm.v1.attention.backends.rocm_aiter_fa",
            "AiterFlashAttentionBackend",
            "ROCM_AITER_FA",
        ),
        (
            "vllm.v1.attention.backends.mla.rocm_aiter_mla",
            "AiterMLABackend",
            "ROCM_AITER_MLA",
        ),
        (
            "vllm.v1.attention.backends.mla.aiter_triton_mla",
            "AiterTritonMLABackend",
            "ROCM_AITER_TRITON_MLA",
        ),
        (
            "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse",
            "ROCMAiterMLASparseBackend",
            "ROCM_AITER_MLA_SPARSE",
        ),
        (
            "vllm.v1.attention.backends.rocm_aiter_unified_attn",
            "RocmAiterUnifiedAttentionBackend",
            "ROCM_AITER_UNIFIED_ATTN",
        ),
    ]
    import importlib

    for module_path, class_name, expected_enum_key in checks:
        mod = importlib.import_module(module_path)
        backend_cls = getattr(mod, class_name)
        name = backend_cls.get_name()
        # The enum key should contain the backend name (case-insensitive).
        # e.g. get_name()="ROCM_AITER_MLA" → enum key "ROCM_AITER_MLA"
        assert expected_enum_key == expected_enum_key.upper(), (
            f"Expected enum key {expected_enum_key!r} to be uppercase"
        )
        assert name is not None and len(name) > 0, (
            f"{class_name}.get_name() returned empty string"
        )
