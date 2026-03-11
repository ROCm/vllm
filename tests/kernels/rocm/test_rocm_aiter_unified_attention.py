# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AITER Unified Attention backend on ROCm.

Covers:
- ROCM_AITER_UNIFIED_ATTN backend name/registration
- VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION env var selection
- Backend properties: block size, head size, prefix/sink support
- Verify backend selection: USE_AITER=1 + USE_AITER_UNIFIED_ATTENTION=1
  → ROCM_AITER_UNIFIED_ATTN is chosen
- Verify decode output matches naive einsum reference
  (aiter Triton unified_attention kernel)
- Verify prefill output matches scaled_dot_product_attention (AITER FA kernel)
- Verify combined prefill+decode correctness: decode vs naive reference
- FP8 KV cache decode correctness
- Determinism
"""

import importlib
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.rocm.utils import _assert_accurate, _assert_deterministic
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

aiter_available = importlib.util.find_spec("aiter") is not None

BLOCK_SIZE = 16
NUM_BLOCKS = 1024
# Covers: 64 (Falcon/GPT-NeoX), 80 (Gemma 2/Code Llama), 96 (Phi-3),
#         128 (Llama/Mistral/Mixtral), 256 (Gemma/Yi)
HEAD_SIZES = [64, 80, 96, 128, 256]
BLOCK_SIZES = [16, 32, 64]
DTYPES = [torch.bfloat16, torch.float16]

FP8_DTYPE = (
    current_platform.fp8_dtype() if current_platform.is_rocm() else torch.float8_e4m3fn
)


def require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter package required for this test")


def require_fp8():
    if not current_platform.supports_fp8():
        pytest.skip("FP8 not supported on this hardware")


# ── Reference implementation ──────────────────────────────────────────────


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list,
    kv_lens: list,
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Naive reference paged attention using einsum.

    Args:
        query: [total_query_tokens, num_q_heads, head_size]
        key_cache: [num_blocks, block_size, num_kv_heads, head_size]
        value_cache: same shape as key_cache
        query_lens: list of per-sequence query lengths
        kv_lens: list of per-sequence KV lengths
        block_tables: [num_seqs, max_blocks_per_seq] int32
        scale: softmax scale (1/sqrt(head_size))
    Returns:
        [total_query_tokens, num_q_heads, head_size]
    """
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        # GQA: expand kv heads to match query heads
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1
        ).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


# ── Backend property tests ─────────────────────────────────────────────────


def test_unified_attn_backend_name():
    """The unified attention backend has the correct name."""
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )

    assert RocmAiterUnifiedAttentionBackend.get_name() == "ROCM_AITER_UNIFIED_ATTN"


def test_unified_attn_backend_block_size_support():
    """Unified attention supports block sizes that are multiples of 16."""
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )

    assert RocmAiterUnifiedAttentionBackend.supports_block_size(16)
    assert RocmAiterUnifiedAttentionBackend.supports_block_size(32)
    assert RocmAiterUnifiedAttentionBackend.supports_block_size(64)
    # Non-multiples of 16 should NOT be supported
    assert not RocmAiterUnifiedAttentionBackend.supports_block_size(1)
    assert not RocmAiterUnifiedAttentionBackend.supports_block_size(7)


def test_unified_attn_backend_head_size_support():
    """Unified attention supports head sizes >= 32."""
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )

    assert RocmAiterUnifiedAttentionBackend.supports_head_size(32)
    assert RocmAiterUnifiedAttentionBackend.supports_head_size(64)
    assert RocmAiterUnifiedAttentionBackend.supports_head_size(128)
    assert RocmAiterUnifiedAttentionBackend.supports_head_size(256)
    # Too small should not be supported
    assert not RocmAiterUnifiedAttentionBackend.supports_head_size(16)


def test_unified_attn_backend_supports_prefix():
    """Unified attention supports prefix (chunked prefill / mm prefix)."""
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )

    assert RocmAiterUnifiedAttentionBackend.supports_mm_prefix()


def test_unified_attn_backend_supports_sink():
    """Unified attention supports sink attention."""
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )

    assert RocmAiterUnifiedAttentionBackend.supports_sink()


# ── Backend selection tests ───────────────────────────────────────────────


def test_unified_attn_env_var_selection(monkeypatch):
    """VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
    → backend selection returns ROCM_AITER_UNIFIED_ATTN path."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "1")

    importlib.reload(importlib.import_module("vllm.envs"))

    from vllm.platforms.rocm import RocmPlatform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

    attn_selector_config = AttentionSelectorConfig(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=BLOCK_SIZE,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
    )

    backend_path = RocmPlatform.get_attn_backend_cls(
        selected_backend=None, attn_selector_config=attn_selector_config
    )

    assert backend_path == AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN.get_path()


def test_unified_attn_explicit_selection(monkeypatch):
    """Explicit ROCM_AITER_UNIFIED_ATTN backend enum selection works."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    importlib.reload(importlib.import_module("vllm.envs"))

    from vllm.platforms.rocm import RocmPlatform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

    attn_selector_config = AttentionSelectorConfig(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=BLOCK_SIZE,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
    )

    backend_path = RocmPlatform.get_attn_backend_cls(
        selected_backend=AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
        attn_selector_config=attn_selector_config,
    )

    assert backend_path == AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN.get_path()


# ── Numerical correctness tests ────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    "AITER unified_attention Triton decode bf16 precision gap: "
                    "requires atol=1e-2, NVIDIA paged_attention C++ achieves "
                    "atol=1e-3 (test_attention.py). Fix upstream."
                ),
            ),
        ),
        torch.float16,
    ],
)
@torch.inference_mode()
def test_unified_attn_decode_correctness(head_size, dtype, block_size):
    """Unified attention decode output matches naive einsum reference.

    For decode tokens (query_len=1), aiter.unified_attention is compared
    against a naive einsum reference to verify correctness across head sizes
    and block sizes. This tests the actual Triton kernel used by the backend.
    """
    require_aiter()
    from aiter.ops.triton.unified_attention import unified_attention

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    num_q_heads = 8
    num_kv_heads = 8
    num_seqs = 4
    seq_lens = [128, 256, 384, 512]
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5
    num_blocks = 1024

    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    max_num_blocks = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    # Reference: naive einsum paged attention
    output_ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=seq_lens,
        block_tables=block_tables,
        scale=scale,
    )

    # Kernel under test: aiter Triton unified_attention (actual backend kernel)
    # Decode: 1 query token per sequence → cu_seqlens_q = [0, 1, 2, ..., num_seqs]
    output_unified = torch.zeros(num_seqs, num_q_heads, head_size, dtype=dtype)
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
    k_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")
    v_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output_unified,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seq_lens_tensor,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=None,
        output_scale=None,
    )

    _assert_accurate(output_unified, output_ref, atol=1e-3, rtol=1e-3)


def test_unified_attn_rocm_aiter_ops_enabled():
    """Test rocm_aiter_ops.is_triton_unified_attn_enabled() API exists."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    result = rocm_aiter_ops.is_triton_unified_attn_enabled()
    # Returns None (aiter disabled) or bool
    assert result is None or isinstance(result, bool)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_unified_attn_prefill_correctness(head_size, dtype):
    """Unified attention prefill (AITER FA) matches scaled_dot_product_attention.

    Calls aiter.flash_attn_varlen_func on the same Q/K/V as SDPA and verifies
    the outputs agree within BF16/FP16 flash attention tolerance.
    """
    require_aiter()
    import aiter

    torch.set_default_device("cuda")
    torch.manual_seed(1)

    batch_size = 2
    seq_len = 32
    num_heads = 8
    scale = head_size**-0.5

    # [batch, heads, seq_len, head_size]
    q = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Reference: PyTorch scaled dot-product attention (causal)
    ref_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
    )
    assert ref_out.shape == (batch_size, num_heads, seq_len, head_size)
    assert not torch.any(torch.isnan(ref_out))
    assert not torch.any(torch.isinf(ref_out))

    # Kernel under test: AITER flash_attn_varlen_func
    # Reshape from [batch, heads, seq_len, head_size]
    # -> [batch*seq_len, heads, head_size]
    q_flat = (
        q.permute(0, 2, 1, 3)
        .reshape(batch_size * seq_len, num_heads, head_size)
        .contiguous()
    )
    k_flat = (
        k.permute(0, 2, 1, 3)
        .reshape(batch_size * seq_len, num_heads, head_size)
        .contiguous()
    )
    v_flat = (
        v.permute(0, 2, 1, 3)
        .reshape(batch_size * seq_len, num_heads, head_size)
        .contiguous()
    )
    # cu_seqlens: [0, seq_len, 2*seq_len] for two sequences of equal length
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda"
    )

    aiter_out_flat = torch.empty_like(q_flat)
    aiter.flash_attn_varlen_func(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=aiter_out_flat,
    )
    # Reshape back: [batch*seq_len, heads, head_size]
    # -> [batch, heads, seq_len, head_size]
    aiter_out = aiter_out_flat.reshape(batch_size, seq_len, num_heads, head_size)
    aiter_out = aiter_out.permute(0, 2, 1, 3)

    torch.testing.assert_close(aiter_out, ref_out, atol=1.5e-2, rtol=1e-2)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "AITER unified_attention combined prefill+decode "
        "precision gap: requires atol=1e-2, "
        "NVIDIA paged_attention C++ achieves "
        "atol=1e-3 (test_attention.py). "
        "Fix in upstream aiter unified_attention "
        "kernel."
    ),
)
@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_unified_attn_combined_prefill_decode_correctness(head_size, dtype):
    """Unified attention decode reads paged KV cache correctly vs naive reference.

    Uses a random KV cache (simulating a pre-filled cache) and verifies that
    aiter.unified_attention output matches the naive einsum reference.
    """
    require_aiter()
    from aiter.ops.triton.unified_attention import unified_attention

    torch.set_default_device("cuda")
    torch.manual_seed(2)

    num_q_heads = 4
    num_kv_heads = 4
    num_seqs = 2
    seq_lens = [65, 65]  # prefill_len + decode_len
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5

    num_blocks = 256
    key_cache = torch.randn(
        num_blocks, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    max_num_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    # Decode query: [num_seqs, num_q_heads, head_size]
    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype)

    # Reference: naive einsum paged attention
    output_ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=seq_lens,
        block_tables=block_tables,
        scale=scale,
    )

    # Kernel under test: aiter Triton unified_attention
    output = torch.zeros(num_seqs, num_q_heads, head_size, dtype=dtype)
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
    k_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")
    v_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seq_lens_tensor,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=None,
        output_scale=None,
    )

    assert output.shape == (num_seqs, num_q_heads, head_size)
    assert not torch.any(torch.isnan(output))
    assert not torch.any(torch.isinf(output))

    _assert_accurate(output, output_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_unified_attn_decode_determinism(dtype):
    """aiter.unified_attention (decode path) produces bitwise-identical results."""
    require_aiter()
    from aiter.ops.triton.unified_attention import unified_attention

    torch.set_default_device("cuda")
    torch.manual_seed(3)

    head_size = 128
    num_q_heads = 8
    num_kv_heads = 8
    num_seqs = 2
    seq_lens = [64, 128]
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    max_num_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
    k_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")
    v_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")

    def run_decode():
        out = torch.zeros(num_seqs, num_q_heads, head_size, dtype=dtype)
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seq_lens_tensor,
            max_seqlen_k=max_seq_len,
            softmax_scale=scale,
            causal=True,
            alibi_slopes=None,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
            sinks=None,
            output_scale=None,
        )
        return out

    _assert_deterministic(run_decode, n_runs=4)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@torch.inference_mode()
def test_unified_attn_prefill_determinism():
    """AITER flash_attn_varlen_func (prefill path) is deterministic on ROCm."""
    require_aiter()
    import aiter

    torch.set_default_device("cuda")
    torch.manual_seed(4)

    batch_size, num_heads, seq_len, head_size = 2, 8, 64, 128
    scale = head_size**-0.5
    q = torch.randn(batch_size * seq_len, num_heads, head_size, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32)

    def run_prefill():
        out = torch.empty_like(q)
        aiter.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            min_seqlen_q=1,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            return_lse=False,
            out=out,
        )
        return out

    _assert_deterministic(run_prefill, n_runs=4)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_unified_attn_decode_fp8_kv_cache(head_size, dtype):
    """FP8 KV cache decode: aiter.unified_attention with FP8 key/value cache.

    Compares FP8 KV cache output to naive reference on dtype-cast KV cache.
    Tolerance accounts for FP8 quantization error (unit scale, atol=0.5).
    """
    require_aiter()
    require_fp8()
    from aiter.ops.triton.unified_attention import unified_attention

    torch.set_default_device("cuda")
    torch.manual_seed(5)

    num_q_heads = 8
    num_kv_heads = 8
    num_seqs = 2
    seq_lens = [128, 256]
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5
    num_blocks = 512

    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype)
    # FP8 KV cache — clamp to avoid extreme values
    key_cache_fp8 = torch.clamp(
        torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
    ).to(FP8_DTYPE)
    value_cache_fp8 = torch.clamp(
        torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
    ).to(FP8_DTYPE)

    max_num_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    # Kernel under test: aiter Triton unified_attention with FP8 KV cache
    # k_descale/v_descale = 1.0 (unit scale — FP8 values represent actual values)
    output_fp8 = torch.zeros(num_seqs, num_q_heads, head_size, dtype=dtype)
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
    k_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")
    v_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")

    unified_attention(
        q=query,
        k=key_cache_fp8,
        v=value_cache_fp8,
        out=output_fp8,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seq_lens_tensor,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=None,
        output_scale=None,
    )

    assert output_fp8.shape == (num_seqs, num_q_heads, head_size)
    assert not torch.any(torch.isnan(output_fp8))
    assert not torch.any(torch.isinf(output_fp8))

    # Reference: naive reference on dtype-cast KV cache (unit-scale dequant)
    key_cache_ref = key_cache_fp8.to(dtype)
    value_cache_ref = value_cache_fp8.to(dtype)
    output_ref = ref_paged_attn(
        query=query,
        key_cache=key_cache_ref,
        value_cache=value_cache_ref,
        query_lens=[1] * num_seqs,
        kv_lens=seq_lens,
        block_tables=block_tables,
        scale=scale,
    )

    # FP8 quantization error introduces noise; use loose tolerance
    _assert_accurate(
        output_fp8,
        output_ref,
        atol=0.5,
        rtol=0.1,
        pass_rate=0.95,
        max_violation_factor=5.0,
    )
