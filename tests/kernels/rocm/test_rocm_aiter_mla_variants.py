# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ROCm AITER MLA backend variants not covered by test_rocm_aiter_mla.py.

Covers:
- AiterTritonMLABackend (ROCM_AITER_TRITON_MLA): backend name, is_mla(), importability
- ROCMAiterMLASparseBackend (ROCM_AITER_MLA_SPARSE): name, is_sparse(), is_mla(),
  supported dtypes, block sizes
- AttentionBackendEnum values for ROCm MLA variants
- Backend metadata classes existence
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


# ── AiterTritonMLABackend tests ───────────────────────────────────────────


def test_aiter_triton_mla_importable():
    """AiterTritonMLABackend is importable from its module."""
    from vllm.v1.attention.backends.mla.aiter_triton_mla import (
        AiterTritonMLABackend,
    )

    assert AiterTritonMLABackend is not None


def test_aiter_triton_mla_backend_name():
    """AiterTritonMLABackend.get_name() returns 'AITER_TRITON_MLA'."""
    from vllm.v1.attention.backends.mla.aiter_triton_mla import (
        AiterTritonMLABackend,
    )

    assert AiterTritonMLABackend.get_name() == "AITER_TRITON_MLA"


def test_aiter_triton_mla_is_mla():
    """AiterTritonMLABackend.is_mla() returns True."""
    from vllm.v1.attention.backends.mla.aiter_triton_mla import (
        AiterTritonMLABackend,
    )

    assert AiterTritonMLABackend.is_mla() is True


def test_aiter_triton_mla_impl_cls():
    """AiterTritonMLABackend.get_impl_cls() returns a non-None class."""
    from vllm.v1.attention.backends.mla.aiter_triton_mla import (
        AiterTritonMLABackend,
        AiterTritonMLAImpl,
    )

    assert AiterTritonMLABackend.get_impl_cls() is AiterTritonMLAImpl


def test_aiter_triton_mla_enum_value():
    """AttentionBackendEnum.ROCM_AITER_TRITON_MLA exists."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert hasattr(AttentionBackendEnum, "ROCM_AITER_TRITON_MLA")
    # Verify it can be accessed without error
    val = AttentionBackendEnum.ROCM_AITER_TRITON_MLA
    assert val is not None


def test_aiter_triton_mla_supported_dtypes():
    """AiterTritonMLABackend inherits FP16 and BF16 support from AiterMLABackend."""
    from vllm.v1.attention.backends.mla.aiter_triton_mla import (
        AiterTritonMLABackend,
    )

    assert torch.float16 in AiterTritonMLABackend.supported_dtypes
    assert torch.bfloat16 in AiterTritonMLABackend.supported_dtypes


def test_aiter_triton_mla_supported_kv_cache_dtypes():
    """AiterTritonMLABackend inherits KV cache dtype support from AiterMLABackend."""
    from vllm.v1.attention.backends.mla.aiter_triton_mla import (
        AiterTritonMLABackend,
    )

    kv_dtypes = AiterTritonMLABackend.supported_kv_cache_dtypes
    assert "auto" in kv_dtypes
    assert "fp8" in kv_dtypes or "fp8_e4m3" in kv_dtypes


def test_aiter_triton_mla_block_size():
    """AiterTritonMLABackend uses block_size=1 inherited from AiterMLABackend."""
    from vllm.v1.attention.backends.mla.aiter_triton_mla import (
        AiterTritonMLABackend,
    )

    assert 1 in AiterTritonMLABackend.get_supported_kernel_block_sizes()


# ── AiterMLABackend (ASM) dtype / head-size / block-size checks ───────────


def test_aiter_mla_asm_supported_dtypes():
    """AiterMLABackend (ASM) supports float16 and bfloat16."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    assert torch.float16 in AiterMLABackend.supported_dtypes
    assert torch.bfloat16 in AiterMLABackend.supported_dtypes


def test_aiter_mla_asm_supported_kv_cache_dtypes():
    """AiterMLABackend (ASM) declares FP8 KV cache support."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    kv_dtypes = AiterMLABackend.supported_kv_cache_dtypes
    assert "auto" in kv_dtypes
    assert "fp8" in kv_dtypes or "fp8_e4m3" in kv_dtypes


def test_aiter_mla_asm_block_size():
    """AiterMLABackend (ASM) uses block_size=1 (each page holds 1 KV token)."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    assert AiterMLABackend.get_supported_kernel_block_sizes() == [1]


def test_aiter_mla_asm_head_sizes_unconstrained():
    """AiterMLABackend.get_supported_head_sizes() returns [] (accepts any head_size).

    The ASM kernel constraints are on num_heads (16 or 128), not head_size.
    Returning [] signals 'any head_size' to the vllm backend selection logic.
    """
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    assert AiterMLABackend.get_supported_head_sizes() == []


# ── ROCMAiterMLASparseBackend tests ───────────────────────────────────────


def test_rocm_aiter_mla_sparse_importable():
    """ROCMAiterMLASparseBackend is importable from its module."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert ROCMAiterMLASparseBackend is not None


def test_rocm_aiter_mla_sparse_backend_name():
    """ROCMAiterMLASparseBackend.get_name() returns 'ROCM_AITER_MLA_SPARSE'."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert ROCMAiterMLASparseBackend.get_name() == "ROCM_AITER_MLA_SPARSE"


def test_rocm_aiter_mla_sparse_is_sparse():
    """ROCMAiterMLASparseBackend.is_sparse() returns True."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert ROCMAiterMLASparseBackend.is_sparse() is True


def test_rocm_aiter_mla_sparse_is_mla():
    """ROCMAiterMLASparseBackend.is_mla() returns True."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert ROCMAiterMLASparseBackend.is_mla() is True


def test_rocm_aiter_mla_sparse_supported_dtypes():
    """ROCMAiterMLASparseBackend supports float16 and bfloat16."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert torch.float16 in ROCMAiterMLASparseBackend.supported_dtypes
    assert torch.bfloat16 in ROCMAiterMLASparseBackend.supported_dtypes


def test_rocm_aiter_mla_sparse_supported_kv_cache_dtypes():
    """ROCMAiterMLASparseBackend supports 'auto' and 'bfloat16' KV cache dtypes."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    kv_dtypes = ROCMAiterMLASparseBackend.supported_kv_cache_dtypes
    assert "auto" in kv_dtypes
    assert "bfloat16" in kv_dtypes


def test_rocm_aiter_mla_sparse_block_sizes():
    """ROCMAiterMLASparseBackend block size is [1] (token-level)."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    block_sizes = ROCMAiterMLASparseBackend.get_supported_kernel_block_sizes()
    assert block_sizes == [1]


def test_rocm_aiter_mla_sparse_head_sizes_unconstrained():
    """ROCMAiterMLASparseBackend.get_supported_head_sizes()
    returns [] (any head_size)."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert ROCMAiterMLASparseBackend.get_supported_head_sizes() == []


def test_rocm_aiter_mla_sparse_enum_value():
    """AttentionBackendEnum.ROCM_AITER_MLA_SPARSE exists."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert hasattr(AttentionBackendEnum, "ROCM_AITER_MLA_SPARSE")
    val = AttentionBackendEnum.ROCM_AITER_MLA_SPARSE
    assert val is not None


def test_rocm_aiter_mla_sparse_metadata_importable():
    """ROCMAiterMLASparseMetadata dataclass is importable."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseMetadata,
    )

    assert ROCMAiterMLASparseMetadata is not None


def test_rocm_aiter_mla_sparse_metadata_builder_importable():
    """ROCMAiterMLASparseMetadataBuilder is importable."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseMetadataBuilder,
    )

    assert ROCMAiterMLASparseMetadataBuilder is not None
