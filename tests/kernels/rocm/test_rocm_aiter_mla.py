# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AITER MLA (Multi-head Latent Attention) on ROCm.

Covers:
- AITER MLA decode forward via rocm_aiter_ops custom op
- FP8 MLA support detection and caching (_check_aiter_mla_fp8_support)
- VLLM_ROCM_USE_AITER_MLA env var gating
- BF16 and FP8 decode paths
- op registration and torch.compile compatibility
"""

import importlib.util

import pytest
import torch

from tests.kernels.rocm.utils import _assert_accurate, _assert_deterministic
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

aiter_available = importlib.util.find_spec("aiter") is not None


# ── Helpers ───────────────────────────────────────────────────────────────


def require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter package required for this test")


# ── FP8 support detection tests ───────────────────────────────────────────


class TestAiterMlaFp8Support:
    """Tests for the _check_aiter_mla_fp8_support() function."""

    def setup_method(self):
        import vllm._aiter_ops as aiter_ops

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

    def test_fp8_support_returns_bool(self):
        """Result is always a bool (True or False, never None after caching)."""
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        result = _check_aiter_mla_fp8_support()
        # Returns bool once checked (even if aiter not available)
        assert isinstance(result, bool)

    def test_fp8_support_result_cached(self):
        """Check that the result is cached after first call."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = True
        result = _check_aiter_mla_fp8_support()
        assert result is True  # Returns cached value

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = False
        result = _check_aiter_mla_fp8_support()
        assert result is False  # Returns newly cached value

    def test_fp8_support_false_when_no_aiter(self):
        """Without aiter, support check returns False gracefully."""
        from unittest.mock import patch

        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with (
            patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True),
            patch(
                "inspect.signature",
                side_effect=ImportError("no aiter"),
            ),
        ):
            result = _check_aiter_mla_fp8_support()
            assert result is False

    def test_fp8_support_handles_attribute_error(self):
        """AttributeError during signature check returns False."""
        from unittest.mock import patch

        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with (
            patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True),
            patch(
                "inspect.signature",
                side_effect=AttributeError("no attribute"),
            ),
        ):
            result = _check_aiter_mla_fp8_support()
            assert result is False


# ── MLA op registration tests ─────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_mla_custom_op_registered():
    """Test that rocm_aiter_mla_decode_fwd custom op is registered."""
    require_aiter()
    # Import to trigger op registration
    import vllm._aiter_ops as aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_mla_decode_fwd")
    assert callable(torch.ops.vllm.rocm_aiter_mla_decode_fwd)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_mla_fake_tensor_support():
    """Test that the fake tensor implementation works for torch.compile."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    # Create representative tensors for opcheck.
    # nhead=128, q_head_dim=576, v_head_dim=512: DeepSeek MLA dimensions.
    # The kernel uses q_head_dim for attention scores and writes v_head_dim output.
    batch_size = 4
    nhead = 128
    q_head_dim = 576  # kv_lora_rank + qk_rope_head_dim
    v_head_dim = 512  # kv_lora_rank (output dimension)
    num_kv_heads = 1
    kv_seq_len = 128

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16, device="cuda")
    kv_buffer = torch.randn(
        kv_seq_len, num_kv_heads, q_head_dim, dtype=torch.bfloat16, device="cuda"
    )
    o = torch.zeros(batch_size, nhead, v_head_dim, dtype=torch.bfloat16, device="cuda")
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * (
        kv_seq_len // batch_size
    )
    kv_indices = torch.arange(0, kv_seq_len, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    # max_seqlen_qo=1: decode mode has exactly 1 query token per sequence
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_mla_decode_fwd,
        (q, kv_buffer, o, qo_indptr, 1),
        kwargs={
            "kv_indptr": kv_indptr,
            "kv_indices": kv_indices,
            "kv_last_page_lens": kv_last_page_lens,
            "sm_scale": q_head_dim**-0.5,
        },
        test_utils=("test_faketensor",),
    )


# ── MLA env var tests ─────────────────────────────────────────────────────


def test_aiter_mla_is_mla_enabled_reflects_env():
    """Test that rocm_aiter_ops.is_mla_enabled() works correctly."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    result = rocm_aiter_ops.is_mla_enabled()
    # Returns None (aiter disabled) or bool (enabled/disabled by env)
    assert result is None or isinstance(result, bool)


def test_aiter_mla_backend_name():
    """Test that the AITER MLA backend has the correct name."""
    require_aiter()
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    assert AiterMLABackend.get_name() == "ROCM_AITER_MLA"


def test_aiter_mla_supported_dtypes():
    """Test that AITER MLA backend supports FP16 and BF16."""
    require_aiter()
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    supported = AiterMLABackend.supported_dtypes
    assert torch.float16 in supported
    assert torch.bfloat16 in supported


def test_aiter_mla_supported_kv_dtypes():
    """Test that AITER MLA backend supports FP8 KV cache."""
    require_aiter()
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    kv_dtypes = AiterMLABackend.supported_kv_cache_dtypes
    assert "fp8" in kv_dtypes or "fp8_e4m3" in kv_dtypes


def test_aiter_mla_block_size_support():
    """Test that AITER MLA backend supports block_size=1 for decode."""
    require_aiter()
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    kernel_block_sizes = AiterMLABackend.get_supported_kernel_block_sizes()
    assert 1 in kernel_block_sizes


# ── MLA decode forward tests ──────────────────────────────────────────────
#
# Kernel constraints on gfx942 (precompiled ASM):
#   - num_heads (gqa_ratio) supported: 16 and 128.
#     Kernel files: mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co (nhead=16)
#                   mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co (nhead=128)
#     Other values raise: "get_heuristic_kernel_mla: cannot get heuristic kernel!"
#   - max_seqlen_qo must be 1 for decode (backend passes qo_len.max()=1 in decode).
#     Passing batch_size raises: "get_heuristic_kernel_mla: causal:0 qseqlen:N".
#   - block_size=1 always (each page holds exactly 1 KV token).
#   - No FP4 MLA kernel exists; FP4 in aiter is GEMM-only (gfx950).
#   Source: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (max_qo_len computation)
#           aiter/mla.py (nhead dispatch logic)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.xfail(
    reason=(
        "nhead=1 is not supported by gfx942 precompiled ASM MLA kernels; "
        "only nhead∈{16,128} are available. Kernel raises: "
        "'get_heuristic_kernel_mla: cannot get heuristic kernel!'"
    ),
    raises=RuntimeError,
    strict=True,
)
@torch.inference_mode()
def test_aiter_mla_decode_unsupported_nhead_raises():
    """Unsupported nhead values raise RuntimeError from the C++ kernel selector.

    gfx942 has precompiled ASM kernels only for nhead=16 and nhead=128.
    nhead=1 (or other unsupported values) fail at the C++ heuristic kernel
    selection step with: "get_heuristic_kernel_mla: cannot get heuristic kernel!"
    """
    require_aiter()

    torch.set_default_device("cuda")
    batch_size = 2
    nhead = 1  # unsupported: only 16 and 128 are precompiled
    q_head_dim = 576
    v_head_dim = 512
    kv_seq_len = 16
    num_kv_heads = 1

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16)
    kv_buffer = torch.randn(kv_seq_len, num_kv_heads, q_head_dim, dtype=torch.bfloat16)
    o = torch.zeros(batch_size, nhead, v_head_dim, dtype=torch.bfloat16)
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32)
    tokens_per_seq = kv_seq_len // batch_size
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * tokens_per_seq
    kv_indices = torch.arange(0, kv_seq_len, dtype=torch.int32)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32)

    # This call should raise RuntimeError from the C++ kernel heuristic selector
    torch.ops.vllm.rocm_aiter_mla_decode_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        1,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        sm_scale=q_head_dim**-0.5,
    )


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@torch.inference_mode()
def test_aiter_mla_decode_bf16_basic():
    """Test AITER MLA decode in BF16: output shape and dtype are correct."""
    require_aiter()

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    batch_size = 4
    # DeepSeek-style MLA: kv_lora_rank=512, qk_rope_head_dim=64 → q_head_dim=576
    # nhead=128: matches gfx942 precompiled ASM kernel
    # (mla_dec_stage1_bf16_a16w16_subQ128_mqa128)
    # v_head_dim=512: output uses only the kv_lora_rank portion of kv_buffer
    nhead = 128
    q_head_dim = 576  # kv_lora_rank + qk_rope_head_dim (used for attention scores)
    v_head_dim = 512  # kv_lora_rank (used for output weighted sum)
    kv_seq_len = 256
    num_kv_heads = 1

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16)
    kv_buffer = torch.randn(kv_seq_len, num_kv_heads, q_head_dim, dtype=torch.bfloat16)
    # o has v_head_dim (not q_head_dim) — kernel writes only the kv_lora_rank portion
    o = torch.zeros(batch_size, nhead, v_head_dim, dtype=torch.bfloat16)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32)
    tokens_per_seq = kv_seq_len // batch_size
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * tokens_per_seq
    kv_indices = torch.arange(0, kv_seq_len, dtype=torch.int32)
    # page_size=1: each page holds exactly 1 token, so last_page_len is always 1
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32)

    # max_seqlen_qo=1: decode has exactly 1 query token per sequence
    torch.ops.vllm.rocm_aiter_mla_decode_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        1,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        sm_scale=q_head_dim**-0.5,
    )

    assert o.shape == (batch_size, nhead, v_head_dim)
    assert o.dtype == torch.bfloat16
    # Output should be non-trivial (not all zeros)
    assert not torch.all(o == 0)


# ── Reference MLA decode implementation ───────────────────────────────────


def _ref_mla_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    """Pure PyTorch reference for MLA decode (absorbed formulation).

    In absorbed MLA, attention scores use the full kv_buffer dimension (K),
    but the output weighted sum uses only the first v_head_dim dims (V = kv_lora_rank).

    Args:
        q: Query tensor [batch_size, num_heads, q_head_dim] in BF16.
            q_head_dim = kv_lora_rank + qk_rope_head_dim.
        kv_buffer: KV buffer [total_tokens, num_kv_heads, q_head_dim] in BF16.
        kv_indptr: KV sequence start/end indices [batch_size + 1] int32.
        kv_indices: Token indices into kv_buffer [total_tokens] int32.
        sm_scale: Attention scale factor (typically 1/sqrt(q_head_dim)).
        v_head_dim: Output dimension = kv_lora_rank
            (first v_head_dim dims of kv_buffer).

    Returns:
        Output tensor [batch_size, num_heads, v_head_dim] in BF16.
    """
    batch_size, num_heads, q_head_dim = q.shape
    output = torch.zeros(
        batch_size, num_heads, v_head_dim, dtype=q.dtype, device=q.device
    )

    for b in range(batch_size):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        token_indices = kv_indices[start:end]  # [seq_len]

        # K uses full q_head_dim (for attention scores)
        # V uses first v_head_dim dims (kv_lora_rank, for output)
        kv_seq = kv_buffer[token_indices]  # [seq_len, num_kv_heads, q_head_dim]
        k = kv_seq[:, 0, :].float()  # [seq_len, q_head_dim]
        v = kv_seq[:, 0, :v_head_dim].float()  # [seq_len, v_head_dim]

        for h in range(num_heads):
            q_h = q[b, h, :].float()  # [q_head_dim]
            scores = torch.mv(k, q_h) * sm_scale  # [seq_len]
            attn_weights = torch.softmax(scores, dim=0)  # [seq_len]
            output[b, h, :] = torch.mv(v.t(), attn_weights).to(q.dtype)

    return output


# ── Accuracy and determinism tests ────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@torch.inference_mode()
def test_aiter_mla_decode_bf16_accuracy():
    """AITER MLA decode BF16 output matches PyTorch reference.

    Compares the AITER custom op against _ref_mla_decode with
    allow_close tolerance for BF16 attention operations.
    """
    require_aiter()

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    batch_size = 4
    # DeepSeek-style dimensions
    # (see test_aiter_mla_decode_bf16_basic for full explanation)
    nhead = 128
    q_head_dim = 576  # kv_lora_rank + qk_rope_head_dim (used for attention scores)
    v_head_dim = 512  # kv_lora_rank (output dim — kernel writes only this many dims)
    kv_seq_len = 64  # 16 tokens per sequence
    num_kv_heads = 1
    sm_scale = q_head_dim**-0.5

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16)
    kv_buffer = torch.randn(kv_seq_len, num_kv_heads, q_head_dim, dtype=torch.bfloat16)
    o = torch.zeros(batch_size, nhead, v_head_dim, dtype=torch.bfloat16)

    tokens_per_seq = kv_seq_len // batch_size
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32)
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * tokens_per_seq
    kv_indices = torch.arange(0, kv_seq_len, dtype=torch.int32)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32)  # page_size=1

    # max_seqlen_qo=1: decode has exactly 1 query token per sequence
    torch.ops.vllm.rocm_aiter_mla_decode_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        1,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        sm_scale=sm_scale,
    )

    # Reference: K=kv_buffer[:,0,:] (full q_head_dim), V=kv_buffer[:,0,:v_head_dim]
    ref = _ref_mla_decode(q, kv_buffer, kv_indptr, kv_indices, sm_scale, v_head_dim)

    assert o.shape == ref.shape
    _assert_accurate(o.float(), ref.float(), atol=0.01, rtol=0.0)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@torch.inference_mode()
def test_aiter_mla_decode_fp8_accuracy():
    """AITER MLA decode with BF16 KV on FP8-capable hardware: output close to reference.

    This test exercises the FP8-capable code path with BF16 inputs to verify
    the kernel runs without error on FP8-supporting hardware. The op's FP8 KV
    path (passing uint8 kv_buffer with q_scale/kv_scale) is gated separately by
    _check_aiter_mla_fp8_support(). See test_fp8_support_returns_bool for that gate.
    """
    require_aiter()

    torch.set_default_device("cuda")
    torch.manual_seed(1)

    batch_size = 2
    nhead = 128
    q_head_dim = 576
    v_head_dim = 512
    kv_seq_len = 32
    num_kv_heads = 1
    sm_scale = q_head_dim**-0.5

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16)
    kv_buffer = torch.randn(kv_seq_len, num_kv_heads, q_head_dim, dtype=torch.bfloat16)
    o = torch.zeros(batch_size, nhead, v_head_dim, dtype=torch.bfloat16)

    tokens_per_seq = kv_seq_len // batch_size
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32)
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * tokens_per_seq
    kv_indices = torch.arange(0, kv_seq_len, dtype=torch.int32)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32)  # page_size=1

    # max_seqlen_qo=1: decode has exactly 1 query token per sequence
    torch.ops.vllm.rocm_aiter_mla_decode_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        1,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        sm_scale=sm_scale,
    )

    assert not torch.any(torch.isnan(o))
    assert not torch.any(torch.isinf(o))

    ref = _ref_mla_decode(q, kv_buffer, kv_indptr, kv_indices, sm_scale, v_head_dim)
    _assert_accurate(o.float(), ref.float(), atol=0.01, rtol=0.0)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@torch.inference_mode()
def test_aiter_mla_decode_determinism():
    """AITER MLA decode produces bitwise-identical results across N runs."""
    require_aiter()

    torch.set_default_device("cuda")
    torch.manual_seed(2)

    batch_size = 4
    nhead = 128
    q_head_dim = 576
    v_head_dim = 512
    kv_seq_len = 64
    num_kv_heads = 1
    sm_scale = q_head_dim**-0.5

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16)
    kv_buffer = torch.randn(kv_seq_len, num_kv_heads, q_head_dim, dtype=torch.bfloat16)
    tokens_per_seq = kv_seq_len // batch_size
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32)
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * tokens_per_seq
    kv_indices = torch.arange(0, kv_seq_len, dtype=torch.int32)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32)  # page_size=1

    def run_mla():
        o = torch.zeros(batch_size, nhead, v_head_dim, dtype=torch.bfloat16)
        # max_seqlen_qo=1: decode has exactly 1 query token per sequence
        torch.ops.vllm.rocm_aiter_mla_decode_fwd(
            q,
            kv_buffer,
            o,
            qo_indptr,
            1,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=kv_last_page_lens,
            sm_scale=sm_scale,
        )
        return o

    _assert_deterministic(run_mla, n_runs=4)


# ── Parametrized parity tests (parity with NVIDIA FlashMLA / CutlassMLA) ──
#
# NVIDIA tests parametrize over:
#   h_q=[16,32,64,128], batch=[1,16,128], seq_len=[4096,8192,16384], FP8+BF16
#
# ROCm parity: we test both supported nhead values (16 and 128) across a range
# of batch sizes and sequence lengths. FP8 KV path is a separate capability check.
#
# Note: nhead=16 uses mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co
#       nhead=128 uses mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize("nhead", [16, 128])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("kv_seq_len_per_seq", [16, 256])
@torch.inference_mode()
def test_aiter_mla_decode_parametrized_accuracy(nhead, batch_size, kv_seq_len_per_seq):
    """AITER MLA decode accuracy across supported nhead values,
    batch sizes and seq lens.

    Tests both gfx942 precompiled ASM kernels:
    - nhead=16:  mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co
    - nhead=128: mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co

    Parity reference: NVIDIA FlashMLA tests use h_q=[16,32,64,128] across
    batch=[1..128] and mean_sk=[4096,8192,16384].
    """
    require_aiter()

    torch.set_default_device("cuda")
    torch.manual_seed(nhead + batch_size * 100 + kv_seq_len_per_seq)
    torch.cuda.manual_seed_all(nhead + batch_size * 100 + kv_seq_len_per_seq)

    # DeepSeek MLA dimensions (fixed per architecture)
    q_head_dim = 576  # kv_lora_rank(512) + qk_rope_head_dim(64)
    v_head_dim = 512  # kv_lora_rank
    num_kv_heads = 1
    kv_seq_len = batch_size * kv_seq_len_per_seq
    sm_scale = q_head_dim**-0.5

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16)
    kv_buffer = torch.randn(kv_seq_len, num_kv_heads, q_head_dim, dtype=torch.bfloat16)
    o = torch.zeros(batch_size, nhead, v_head_dim, dtype=torch.bfloat16)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32)
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * kv_seq_len_per_seq
    kv_indices = torch.arange(0, kv_seq_len, dtype=torch.int32)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32)  # page_size=1

    torch.ops.vllm.rocm_aiter_mla_decode_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        1,  # max_seqlen_qo=1 always for decode
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        sm_scale=sm_scale,
    )

    ref = _ref_mla_decode(q, kv_buffer, kv_indptr, kv_indices, sm_scale, v_head_dim)

    assert o.shape == (batch_size, nhead, v_head_dim)
    _assert_accurate(o.float(), ref.float(), atol=0.01, rtol=0.0)
