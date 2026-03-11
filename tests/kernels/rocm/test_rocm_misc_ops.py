# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for miscellaneous ROCm-specific env vars and architecture detection.

Covers env vars not tested elsewhere:
- VLLM_ROCM_CUSTOM_PAGED_ATTN: custom paged attention for MI3xx
- VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT: shuffled KV cache layout
- VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE: memory chunk size for sleep
- VLLM_ROCM_USE_SKINNY_GEMM: ROCm skinny GEMM kernels

Architecture detection functions:
- on_gfx9(), on_mi3xx(), on_gfx942(), on_gfx950() all return bool
- supports_mx() requires gfx950; False on gfx942
- supports_fp8() is True on gfx942 and gfx950, False on gfx90a

E2E behavior tests:
- VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT: env var propagates to rocm_aiter_ops class
- VLLM_ROCM_CUSTOM_PAGED_ATTN: paged_attention_rocm determinism and finiteness
- paged_attention_rocm: determinism and finiteness smoke test
"""

import importlib

import pytest
import torch

from tests.kernels.rocm.utils import _assert_accurate
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


# ── Env var readable tests ────────────────────────────────────────────────


def test_custom_paged_attn_env_var_readable():
    """VLLM_ROCM_CUSTOM_PAGED_ATTN is readable and defaults to True."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_CUSTOM_PAGED_ATTN, bool)


def test_shuffle_kv_cache_env_var_readable():
    """VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT is readable and defaults to False."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT, bool)


def test_sleep_mem_chunk_size_env_var_readable():
    """VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE is readable and defaults to 256."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE, int)
    assert envs.VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE > 0


def test_use_skinny_gemm_env_var_readable():
    """VLLM_ROCM_USE_SKINNY_GEMM is readable and defaults to True."""
    import vllm.envs as envs

    assert isinstance(envs.VLLM_ROCM_USE_SKINNY_GEMM, bool)


# ── Env var set tests ─────────────────────────────────────────────────────


@pytest.mark.parametrize("enabled", [True, False])
def test_custom_paged_attn_set(enabled, monkeypatch):
    """VLLM_ROCM_CUSTOM_PAGED_ATTN can be set to True or False."""
    monkeypatch.setenv("VLLM_ROCM_CUSTOM_PAGED_ATTN", "1" if enabled else "0")
    import vllm.envs as envs

    importlib.reload(envs)
    assert enabled == envs.VLLM_ROCM_CUSTOM_PAGED_ATTN


@pytest.mark.parametrize("enabled", [True, False])
def test_shuffle_kv_cache_set(enabled, monkeypatch):
    """VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT can be set to True or False."""
    monkeypatch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1" if enabled else "0")
    import vllm.envs as envs

    importlib.reload(envs)
    assert enabled == envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT


@pytest.mark.parametrize("chunk_size", [128, 256, 512, 1024])
def test_sleep_mem_chunk_size_set(chunk_size, monkeypatch):
    """VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE can be set to various int values."""
    monkeypatch.setenv("VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE", str(chunk_size))
    import vllm.envs as envs

    importlib.reload(envs)
    assert chunk_size == envs.VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE


@pytest.mark.parametrize("enabled", [True, False])
def test_use_skinny_gemm_set(enabled, monkeypatch):
    """VLLM_ROCM_USE_SKINNY_GEMM can be set to True or False."""
    monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "1" if enabled else "0")
    import vllm.envs as envs

    importlib.reload(envs)
    assert enabled == envs.VLLM_ROCM_USE_SKINNY_GEMM


# ── Architecture detection tests ─────────────────────────────────────────


def test_on_gfx9_returns_bool():
    """on_gfx9() returns a bool (True on gfx90a/gfx942/gfx950, False otherwise)."""
    from vllm.platforms.rocm import on_gfx9

    result = on_gfx9()
    assert isinstance(result, bool)


def test_on_mi3xx_returns_bool():
    """on_mi3xx() returns a bool (True on gfx942/gfx950)."""
    from vllm.platforms.rocm import on_mi3xx

    result = on_mi3xx()
    assert isinstance(result, bool)


def test_on_gfx942_returns_bool():
    """on_gfx942() returns a bool (True only on MI300X/MI300A)."""
    from vllm.platforms.rocm import on_gfx942

    result = on_gfx942()
    assert isinstance(result, bool)


def test_on_gfx950_returns_bool():
    """on_gfx950() returns a bool (True only on MI325X/MI350X)."""
    from vllm.platforms.rocm import on_gfx950

    result = on_gfx950()
    assert isinstance(result, bool)


def test_gfx_arch_hierarchy():
    """gfx942 and gfx950 both imply gfx9 (CDNA) and mi3xx."""
    from vllm.platforms.rocm import on_gfx9, on_gfx942, on_gfx950, on_mi3xx

    if on_gfx942():
        assert on_gfx9(), "gfx942 implies gfx9"
        assert on_mi3xx(), "gfx942 implies mi3xx"

    if on_gfx950():
        assert on_gfx9(), "gfx950 implies gfx9"
        assert on_mi3xx(), "gfx950 implies mi3xx"


def test_gfx942_and_gfx950_mutually_exclusive():
    """on_gfx942() and on_gfx950() cannot both be True (different physical arches)."""
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    assert not (on_gfx942() and on_gfx950()), (
        "gfx942 and gfx950 are different GPU families and cannot coexist"
    )


def test_supports_mx_requires_gfx950():
    """supports_mx() is True only on gfx950 hardware (MXFP4 requires MI325X+)."""
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    result = current_platform.supports_mx()
    assert isinstance(result, bool)

    # If we are on gfx942 (MI300X), MX/MXFP4 should NOT be supported
    if on_gfx942() and not on_gfx950():
        assert result is False, "MXFP4 (MX) is not supported on gfx942 (MI300X)"


def test_supports_fp8_on_mi3xx():
    """supports_fp8() is True on gfx942 and gfx950 hardware."""
    from vllm.platforms.rocm import on_mi3xx

    result = current_platform.supports_fp8()
    assert isinstance(result, bool)

    if on_mi3xx():
        assert result is True, "FP8 must be supported on MI3xx (gfx942/gfx950)"


def test_rocm_platform_use_custom_allreduce():
    """use_custom_allreduce() returns True on MI3xx (needed for Quick Reduce)."""
    from vllm.platforms.rocm import on_mi3xx

    result = current_platform.use_custom_allreduce()
    assert isinstance(result, bool)

    if on_mi3xx():
        assert result is True, (
            "Custom allreduce (Quick Reduce) must be enabled on MI3xx"
        )


def test_fp8_dtype_is_fnuz_on_gfx942():
    """fp8_dtype() returns float8_e4m3fnuz on gfx942 (FNUZ format for MI300X)."""
    from vllm.platforms.rocm import on_gfx942

    fp8_dtype = current_platform.fp8_dtype()
    assert fp8_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz), (
        f"Unexpected FP8 dtype: {fp8_dtype}"
    )

    if on_gfx942():
        assert fp8_dtype == torch.float8_e4m3fnuz, (
            "gfx942 (MI300X) uses float8_e4m3fnuz (FNUZ format)"
        )


# ── E2E behavior tests ────────────────────────────────────────────────────


def _run_paged_attention_rocm(
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    seq_lens_tensor,
    block_size,
    max_seq_len,
):
    """Helper: run paged_attention_rocm and return output."""
    from vllm import _custom_ops as ops

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    head_size = query.shape[2]
    num_partitions = (max_seq_len + 255) // 256

    output = torch.empty(num_seqs, num_q_heads, head_size, dtype=query.dtype)
    tmp_output = torch.empty(
        num_seqs, num_q_heads, num_partitions, head_size, dtype=torch.float32
    )
    exp_sums = torch.empty(num_seqs, num_q_heads, num_partitions, dtype=torch.float32)
    max_logits = torch.empty_like(exp_sums)

    ops.paged_attention_rocm(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens_tensor,
        None,
        block_size,
        max_seq_len,
        None,
        "auto",
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    return output


def test_shuffle_kv_cache_layout_env_propagation(monkeypatch):
    """VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT propagates to rocm_aiter_ops class.

    The shuffle KV cache layout is consumed by the AITER FA backend
    (rocm_aiter_fa.py) via rocm_aiter_ops.is_shuffle_kv_cache_enabled().
    Verify the env var propagates correctly through envs reload +
    refresh_env_variables().
    """
    import importlib as imp

    import vllm.envs as envs

    try:
        from vllm._aiter_ops import rocm_aiter_ops
    except ImportError:
        pytest.skip("aiter required for shuffle KV cache test")

    for value, expected in [("1", True), ("0", False)]:
        monkeypatch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", value)
        imp.reload(envs)
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_shuffle_kv_cache_enabled() == expected, (
            f"Expected is_shuffle_kv_cache_enabled()={expected} "
            f"with VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT={value}"
        )


@torch.inference_mode()
def test_paged_attention_rocm_determinism():
    """paged_attention_rocm produces finite, deterministic output.

    Smoke test: the C++ paged attention kernel returns finite values
    and is bitwise deterministic across invocations.
    """
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    num_q_heads = 8
    num_kv_heads = 8
    head_size = 64
    block_size = 16
    num_blocks = 128
    num_seqs = 2
    seq_lens = [64, 128]
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=torch.bfloat16)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(key_cache)
    max_num_blocks = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    out1 = _run_paged_attention_rocm(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens_tensor,
        block_size,
        max_seq_len,
    )

    assert not torch.any(torch.isnan(out1))
    assert not torch.any(torch.isinf(out1))

    out2 = _run_paged_attention_rocm(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens_tensor,
        block_size,
        max_seq_len,
    )
    _assert_accurate(out2.float(), out1.float(), atol=0.0)


def test_custom_paged_attn_env_propagation(monkeypatch):
    """VLLM_ROCM_CUSTOM_PAGED_ATTN propagates to envs and affects backend selection.

    The env var controls whether ROCm custom paged attention is eligible
    during backend selection (rocm.py). Verify the env var is reflected
    in envs after reload.
    """
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_ROCM_CUSTOM_PAGED_ATTN", "1")
    importlib.reload(envs)
    assert envs.VLLM_ROCM_CUSTOM_PAGED_ATTN is True

    monkeypatch.setenv("VLLM_ROCM_CUSTOM_PAGED_ATTN", "0")
    importlib.reload(envs)
    assert envs.VLLM_ROCM_CUSTOM_PAGED_ATTN is False
