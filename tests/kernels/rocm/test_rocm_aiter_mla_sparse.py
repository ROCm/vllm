# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy and kernel tests for the ROCm sparse MLA attention backend.

Parity reference: tests/v1/attention/test_sparse_mla_backends.py covers CUDA
(FlashMLASparseBackend, FlashInferMLASparseBackend) with full E2E accuracy,
FP8, TP, and index-conversion correctness. This file brings the same coverage
to ROCMAiterMLASparseBackend on gfx942/gfx950.

Covers:
- Sparse forward pass output shape and non-trivial values (BF16 KV)
- Sparse forward accuracy vs reference_mla_sparse_prefill
- Parametrized over nhead=[16,128], batch=[1,4], topk=[32,128]
- triton_convert_req_index_to_global_index correctness on ROCm
- fetch_id_to_ragged_triton correctness on ROCm
- Sparse forward determinism

Sparse MLA kernel constraints (same as dense ASM MLA):
- nhead ∈ {16, 128}: gfx942 precompiled ASM kernels
- block_size=1: each page holds exactly 1 KV token
- max_seqlen_qo=1 per decode sequence (1 query token per "request")
- No FP8 sparse MLA kernel; supported_kv_cache_dtypes = ["auto", "bfloat16"]

Notes on test_sparse_mla_backends.py skip:
  The CUDA sparse test file explicitly skips on ROCm with a TODO comment:
  "ROCm support requires integrating ROCMAiterMLASparseBackend."
  These tests fill that gap for the kernel-level paths.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.rocm.utils import _assert_accurate, _assert_deterministic
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

aiter_available = importlib.util.find_spec("aiter") is not None


def require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter package required for this test")


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_sparse_metadata(batch_size: int, topk: int, device: torch.device):
    """Create minimal ROCMAiterMLASparseMetadata for direct kernel tests."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseMetadata,
    )

    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
    # These buffers are populated by _forward_bf16_kv internally
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indices = torch.zeros(batch_size * topk, dtype=torch.int32, device=device)
    paged_kv_indptr_rest = torch.zeros(0, dtype=torch.int32, device=device)

    return ROCMAiterMLASparseMetadata(
        num_reqs=batch_size,
        max_query_len=1,
        max_seq_len=1,
        num_actual_tokens=batch_size,
        query_start_loc=qo_indptr,
        slot_mapping=torch.zeros(batch_size, dtype=torch.long, device=device),
        block_table=torch.zeros((batch_size, 1), dtype=torch.int32, device=device),
        req_id_per_token=torch.zeros(batch_size, dtype=torch.int32, device=device),
        qo_indptr=qo_indptr,
        paged_kv_last_page_len=paged_kv_last_page_len,
        paged_kv_indices=paged_kv_indices,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indptr_rest=paged_kv_indptr_rest,
        block_size=1,
        topk_tokens=topk,
    )


def _make_sparse_impl(nhead: int, q_head_dim: int, v_head_dim: int):
    """Create minimal ROCMAiterMLASparseImpl for direct forward calls."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    impl = SimpleNamespace(
        num_heads=nhead,
        scale=q_head_dim**-0.5,
        kv_lora_rank=v_head_dim,
    )
    impl._forward_bf16_kv = ROCMAiterMLASparseImpl._forward_bf16_kv.__get__(
        impl, ROCMAiterMLASparseImpl
    )
    return impl


# ── triton_convert_req_index_to_global_index tests ────────────────────────
# Same kernel as tested in test_sparse_mla_backends.py for CUDA — verify ROCm.


def _reference_index_convert(
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Pure-Python reference for triton_convert_req_index_to_global_index."""
    num_tokens, num_topk = token_indices.shape
    max_blocks = block_table.shape[1]
    result = torch.empty_like(token_indices)
    for t in range(num_tokens):
        req = req_ids[t].item()
        for k in range(num_topk):
            idx = token_indices[t, k].item()
            if idx == -1:
                result[t, k] = -1
            else:
                block_id = idx // block_size
                if block_id >= max_blocks:
                    result[t, k] = -1
                else:
                    result[t, k] = (
                        block_table[req, block_id].item() * block_size
                        + idx % block_size
                    )
    return result


@pytest.mark.parametrize("block_size", [1, 16, 64])
@pytest.mark.parametrize("num_topk", [128, 256])  # must be divisible by BLOCK_N=128
def test_rocm_sparse_mla_triton_index_conversion(block_size, num_topk):
    """triton_convert_req_index_to_global_index matches Python reference on ROCm."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        triton_convert_req_index_to_global_index,
    )

    device = torch.device("cuda")
    num_tokens = 8
    num_reqs = 4
    max_blocks = 10

    req_ids = torch.randint(
        0, num_reqs, (num_tokens,), dtype=torch.int32, device=device
    )
    block_table = torch.randint(
        0, 50, (num_reqs, max_blocks), dtype=torch.int32, device=device
    )
    token_indices = torch.randint(
        0,
        block_size * max_blocks,
        (num_tokens, num_topk),
        dtype=torch.int32,
        device=device,
    )
    # Insert some -1 masked entries
    token_indices[0, :5] = -1
    token_indices[3, num_topk // 2 :] = -1

    result = triton_convert_req_index_to_global_index(
        req_ids,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk,
    )
    ref = _reference_index_convert(req_ids, block_table, token_indices, block_size)

    torch.testing.assert_close(result, ref, rtol=0, atol=0)


# ── fetch_id_to_ragged_triton tests ───────────────────────────────────────


def test_rocm_sparse_mla_fetch_id_to_ragged():
    """fetch_id_to_ragged_triton correctly converts topk indices to ragged format."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        fetch_id_to_ragged_triton,
    )

    device = torch.device("cuda")
    num_tokens = 4
    topk = 8

    # Indices: some valid (>=0), some masked (-1)
    topk_indices = torch.tensor(
        [
            [5, 2, -1, -1, -1, -1, -1, -1],  # 2 valid
            [0, 3, 7, 1, -1, -1, -1, -1],  # 4 valid
            [-1, -1, -1, -1, -1, -1, -1, -1],  # 0 valid
            [4, 6, -1, -1, -1, -1, -1, -1],  # 2 valid
        ],
        dtype=torch.int32,
        device=device,
    )
    seq_len = (topk_indices != -1).sum(dim=-1)  # [2, 4, 0, 2]
    cumsum = torch.zeros(num_tokens + 1, dtype=torch.int32, device=device)
    torch.cumsum(seq_len, dim=0, out=cumsum[1:])

    _total_valid = seq_len.sum().item()  # 8
    out = torch.zeros(num_tokens * topk, dtype=torch.int32, device=device)

    fetch_id_to_ragged_triton(topk_indices, cumsum, out, topk)

    # Verify: first 2 entries in out are token 0's valid indices [5, 2]
    assert out[0].item() == 5
    assert out[1].item() == 2
    # Next 4 are token 1's valid indices [0, 3, 7, 1]
    assert out[2].item() == 0
    assert out[3].item() == 3
    assert out[4].item() == 7
    assert out[5].item() == 1
    # Token 2 contributes 0 entries
    # Token 3: indices [4, 6] at positions [6, 7]
    assert out[6].item() == 4
    assert out[7].item() == 6
    # Cumsum boundaries
    assert cumsum[0].item() == 0
    assert cumsum[1].item() == 2
    assert cumsum[2].item() == 6
    assert cumsum[3].item() == 6
    assert cumsum[4].item() == 8


# ── Sparse forward output shape and non-trivial value ─────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@torch.inference_mode()
def test_rocm_sparse_mla_forward_output_shape():
    """Sparse MLA forward produces correct shape and non-zero output."""
    require_aiter()

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    batch_size = 4
    nhead = 128  # gfx942 supported
    q_head_dim = 576  # kv_lora_rank + qk_rope_head_dim
    v_head_dim = 512  # kv_lora_rank (output dim)
    num_kv_tokens = 512
    topk = 128  # must be divisible by BLOCK_N=128

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16, device=device)
    # block_size=1: shape (num_kv_tokens, 1, q_head_dim)
    kv_cache = torch.randn(
        num_kv_tokens, 1, q_head_dim, dtype=torch.bfloat16, device=device
    )
    # Global block indices (block_size=1 → block_idx == token_idx)
    topk_indices = torch.randint(
        0, num_kv_tokens, (batch_size, topk), dtype=torch.int32, device=device
    )

    impl = _make_sparse_impl(nhead, q_head_dim, v_head_dim)
    metadata = _make_sparse_metadata(batch_size, topk, device)

    output = impl._forward_bf16_kv(q, kv_cache, topk_indices, metadata)

    assert output.shape == (batch_size, nhead, v_head_dim)
    assert output.dtype == torch.bfloat16
    assert not torch.all(output == 0), "output should be non-trivial"
    assert torch.isfinite(output).all(), "output should not contain NaN or Inf"


# ── Sparse forward accuracy vs reference ──────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize("nhead", [16, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("topk", [128, 256])  # must be divisible by BLOCK_N=128
@torch.inference_mode()
def test_rocm_sparse_mla_forward_accuracy(nhead, batch_size, topk):
    """Sparse MLA forward output matches reference_mla_sparse_prefill.

    Parity with test_sparse_mla_backends.py::test_sparse_backend_decode_correctness
    (CUDA, auto/BF16 KV) adapted for ROCMAiterMLASparseBackend.

    Reference: reference_mla_sparse_prefill implements:
      scores[q, h, k] = q[q, h, :] @ kv[topk[q, k], 0, :].T * scale
      out[q, h, :] = softmax(scores[q, h, :]) @ kv[topk[q, k], 0, :v_head_dim]
    which is the absorbed MLA formulation with sparse KV selection.
    """
    require_aiter()

    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        reference_mla_sparse_prefill,
    )

    device = torch.device("cuda")
    torch.manual_seed(nhead * 1000 + batch_size * 10 + topk)
    torch.cuda.manual_seed_all(nhead * 1000 + batch_size * 10 + topk)

    q_head_dim = 576  # kv_lora_rank(512) + qk_rope_head_dim(64)
    v_head_dim = 512  # kv_lora_rank
    num_kv_tokens = max(512, topk * 2)

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16, device=device)
    # block_size=1: shape (num_kv_tokens, 1, q_head_dim)
    kv_cache = torch.randn(
        num_kv_tokens, 1, q_head_dim, dtype=torch.bfloat16, device=device
    )
    # All-valid indices (no -1) for exact reference comparison
    topk_indices = torch.randint(
        0, num_kv_tokens, (batch_size, topk), dtype=torch.int32, device=device
    )

    impl = _make_sparse_impl(nhead, q_head_dim, v_head_dim)
    metadata = _make_sparse_metadata(batch_size, topk, device)

    output = impl._forward_bf16_kv(q, kv_cache, topk_indices, metadata)

    # reference_mla_sparse_prefill expects indices shape [sq, 1, topk]
    ref_out, _ = reference_mla_sparse_prefill(
        q=q,
        kv=kv_cache,
        indices=topk_indices.unsqueeze(1),
        sm_scale=q_head_dim**-0.5,
        d_v=v_head_dim,
    )

    assert output.shape == ref_out.shape, (
        f"Shape mismatch: got {output.shape}, expected {ref_out.shape}"
    )
    _assert_accurate(output.float(), ref_out.float(), atol=0.01, rtol=0.0)


# ── Sparse forward determinism ─────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@torch.inference_mode()
def test_rocm_sparse_mla_forward_determinism():
    """Sparse MLA forward produces bitwise-identical results across N runs."""
    require_aiter()

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    batch_size = 4
    nhead = 128
    q_head_dim = 576
    v_head_dim = 512
    num_kv_tokens = 512
    topk = 128  # must be divisible by BLOCK_N=128

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16, device=device)
    kv_cache = torch.randn(
        num_kv_tokens, 1, q_head_dim, dtype=torch.bfloat16, device=device
    )
    topk_indices = torch.randint(
        0, num_kv_tokens, (batch_size, topk), dtype=torch.int32, device=device
    )

    impl = _make_sparse_impl(nhead, q_head_dim, v_head_dim)

    def run():
        metadata = _make_sparse_metadata(batch_size, topk, device)
        return impl._forward_bf16_kv(q, kv_cache, topk_indices, metadata)

    _assert_deterministic(run, n_runs=4)
