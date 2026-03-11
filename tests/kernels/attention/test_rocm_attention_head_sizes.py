# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime accuracy tests for ROCm attention backends across head sizes.

Tests each ROCm backend with various head sizes by running the kernel
and comparing against a naive reference implementation.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

DTYPES = [torch.bfloat16, torch.float16]
NUM_BLOCKS = 2048
BLOCK_SIZE = 16
SEQ_LENS = [(1, 128), (1, 512), (8, 256), (32, 1024)]


_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"


def _assert_close_or_xfail(
    test_name: str,
    output: torch.Tensor,
    ref_output: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    """Try assert_close; print colored accuracy report, xfail on failure."""
    diff = (output.float() - ref_output.float()).abs()
    ref_abs = ref_output.float().abs()
    tol = atol + rtol * ref_abs
    in_tol = (diff <= tol).float().mean().item() * 100

    max_ad = diff.max().item()
    mean_ad = diff.mean().item()
    p50 = diff.median().item()
    p99 = diff.quantile(0.99).item()

    nonzero = ref_abs > 1e-8
    max_rd = (
        (diff[nonzero] / ref_abs[nonzero]).max().item()
        if nonzero.any()
        else float("nan")
    )

    needed_atol = (diff - rtol * ref_abs).clamp(min=0).max().item()
    overshoot = needed_atol / atol if atol > 0 else float("inf")

    passed = True
    try:
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
    except (AssertionError, RuntimeError):
        passed = False

    tag = f"{_GREEN}PASS{_RESET}" if passed else f"{_RED}FAIL{_RESET}"
    print(
        f"\n  [{tag}] {test_name}\n"
        f"    parity: {in_tol:.1f}% in tol | need atol={needed_atol:.4e}"
        f" ({overshoot:.1f}x target)\n"
        f"    target: atol={atol:.0e} rtol={rtol:.0e}\n"
        f"    abs diff: max={max_ad:.4e} mean={mean_ad:.4e}"
        f" p50={p50:.4e} p99={p99:.4e}\n"
        f"    rel diff: max={max_rd:.4e}"
    )

    if not passed:
        pytest.xfail(f"{test_name} | {in_tol:.1f}% in tol, {overshoot:.1f}x target")


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference paged attention using naive einsum implementation."""
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

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


# ---------------------------------------------------------------------------
# ROCM AITER Flash Attention (head sizes: 64, 128, 256)
# ---------------------------------------------------------------------------

AITER_FA_HEAD_SIZES = [64, 128, 256]


@pytest.mark.parametrize("head_size", AITER_FA_HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_heads", [(16, 16), (16, 4)])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@torch.inference_mode()
def test_aiter_fa_head_sizes(head_size, dtype, num_heads, seq_lens):
    """Test AITER Flash Attention accuracy across supported head sizes."""
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter package required")

    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    torch.set_default_device("cuda")
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    query_len, kv_len = seq_lens
    scale = head_size**-0.5

    query = torch.randn(query_len, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seq_lens = torch.tensor([0, kv_len], dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    max_num_blocks_per_seq = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, NUM_BLOCKS, (1, max_num_blocks_per_seq), dtype=torch.int32
    )

    # Gather paged KV into contiguous tensors
    token_to_batch = torch.zeros(kv_len, dtype=torch.int32)
    seq_starts = torch.zeros(1, dtype=torch.int32)
    gathered_key = torch.empty(kv_len, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)

    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=kv_len,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )

    _assert_close_or_xfail(
        f"aiter_fa[hs={head_size},dt={dtype},heads={num_heads},seq={seq_lens}]",
        output,
        ref_output,
        atol=1.5e-2,
        rtol=1e-2,
    )


# ---------------------------------------------------------------------------
# ROCm paged attention with ALiBi slopes
# ---------------------------------------------------------------------------


def _make_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    """Generate standard ALiBi slopes for num_heads attention heads.

    Slopes follow the original ALiBi paper: 2^(-8/n) for n in [1..num_heads].
    """
    closest_power_of_2 = 2 ** (num_heads - 1).bit_length()
    base = torch.tensor(
        [
            2 ** (-(2 ** -(closest_power_of_2 - 8 + i).bit_length()))
            for i in range(1, closest_power_of_2 + 1)
        ],
        dtype=torch.float32,
        device=device,
    )
    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            [
                2 ** (-(2 ** -(closest_power_of_2 - 7 + i).bit_length()))
                for i in range(1, 2 * (num_heads - closest_power_of_2) + 1, 2)
            ],
            dtype=torch.float32,
            device=device,
        )
        base = torch.cat([base, extra_base])
    return base[:num_heads]


def ref_paged_attn_alibi(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list,
    kv_lens: list,
    block_tables: torch.Tensor,
    scale: float,
    alibi_slopes: torch.Tensor,
) -> torch.Tensor:
    """Naive reference paged attention with ALiBi position bias.

    The ALiBi bias is: logit[h, q_idx, k_idx] -= slopes[h] * (q_pos - k_pos)
    where q_pos and k_pos are absolute token positions within the sequence.
    """
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    device = query.device
    slopes = alibi_slopes.to(device)

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

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()

        # Causal mask
        mask = torch.triu(
            torch.ones(query_len, kv_len, device=device),
            diagonal=kv_len - query_len + 1,
        ).bool()
        attn.masked_fill_(mask, float("-inf"))

        # ALiBi bias: bias[h, q_i, k_j] = -slopes[h] * (q_pos_i - k_pos_j)
        # k_pos_j = j (0-indexed), q_pos_i = kv_len - query_len + q_i
        q_positions = torch.arange(
            kv_len - query_len, kv_len, dtype=torch.float32, device=device
        )
        k_positions = torch.arange(kv_len, dtype=torch.float32, device=device)
        pos_diff = q_positions.unsqueeze(1) - k_positions.unsqueeze(0)
        alibi_bias = -slopes.float().unsqueeze(1).unsqueeze(2) * pos_diff.unsqueeze(0)
        attn += alibi_bias

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_heads", [(8, 8), (16, 4)])
@torch.inference_mode()
def test_rocm_paged_attn_alibi(head_size, dtype, num_heads):
    """ROCm paged attention with ALiBi position bias matches naive reference.

    ALiBi modifies the attention logits with a head-specific position penalty,
    allowing models to extrapolate beyond training context lengths.
    """
    from vllm import _custom_ops as ops

    torch.set_default_device("cuda")
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    num_seqs = 4
    seq_lens = [64, 128, 192, 256]
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    max_num_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    alibi_slopes = _make_alibi_slopes(num_query_heads, device=torch.device("cuda"))

    output = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    num_partitions = (max_seq_len + 255) // 256
    tmp_output = torch.empty(
        num_seqs, num_query_heads, num_partitions, head_size, dtype=torch.float32
    )
    exp_sums = torch.empty(
        num_seqs, num_query_heads, num_partitions, dtype=torch.float32
    )
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
        None,  # cu_seq_lens
        BLOCK_SIZE,
        max_seq_len,
        alibi_slopes,
        "auto",
        torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        torch.tensor(1.0, dtype=torch.float32, device="cuda"),
    )

    ref_output = ref_paged_attn_alibi(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=seq_lens,
        block_tables=block_tables,
        scale=scale,
        alibi_slopes=alibi_slopes,
    )

    _assert_close_or_xfail(
        f"paged_attn_alibi[hs={head_size},dt={dtype},heads={num_heads}]",
        output,
        ref_output,
        atol=1e-3,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# ROCM paged attention (head sizes: 32, 64, 80, 96, 128, 160, 192, 224, 256)
# ---------------------------------------------------------------------------

_ROCM_PAGED_ATTN_SUPPORTED = [64, 128]
_ROCM_PAGED_ATTN_UNSUPPORTED = [32, 80, 96, 160, 192, 224, 256]


def _run_paged_attention_rocm(head_size, dtype, num_heads):
    """Set up and run ROCm paged attention, return (output, ref_output)."""
    from vllm import _custom_ops as ops

    num_query_heads, num_kv_heads = num_heads
    num_seqs = 4
    max_seq_len = 512
    scale = head_size**-0.5

    seq_lens = [128, 256, 384, 512]
    max_kv_len = max(seq_lens)

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    max_num_blocks_per_seq = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    output = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    num_partitions = (max_kv_len + 255) // 256
    tmp_output = torch.empty(
        num_seqs,
        num_query_heads,
        num_partitions,
        head_size,
        dtype=torch.float32,
    )
    exp_sums = torch.empty(
        num_seqs, num_query_heads, num_partitions, dtype=torch.float32
    )
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
        None,  # cu_seq_lens
        BLOCK_SIZE,
        max_seq_len,
        None,  # alibi_slopes
        "auto",  # kv_cache_dtype
        torch.tensor(1.0, dtype=torch.float32, device="cuda"),  # k_scale
        torch.tensor(1.0, dtype=torch.float32, device="cuda"),  # v_scale
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=seq_lens,
        block_tables=block_tables,
        scale=scale,
    )
    return output, ref_output


@pytest.mark.parametrize("head_size", _ROCM_PAGED_ATTN_SUPPORTED)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_heads", [(16, 16), (16, 4)])
@torch.inference_mode()
def test_rocm_paged_attn_head_sizes(head_size, dtype, num_heads):
    """Test ROCm paged attention accuracy across supported head sizes."""
    torch.set_default_device("cuda")
    set_random_seed(0)

    output, ref_output = _run_paged_attention_rocm(head_size, dtype, num_heads)

    _assert_close_or_xfail(
        f"paged_attn[hs={head_size},dt={dtype},heads={num_heads}]",
        output,
        ref_output,
        atol=1e-3,
        rtol=1e-5,
    )


@pytest.mark.parametrize("head_size", _ROCM_PAGED_ATTN_UNSUPPORTED)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_heads", [(16, 16)])
@torch.inference_mode()
def test_rocm_paged_attn_unsupported_head_sizes(head_size, dtype, num_heads):
    """Verify ROCm paged attention rejects unsupported head sizes."""
    torch.set_default_device("cuda")
    set_random_seed(0)

    with pytest.raises(RuntimeError, match="Unsupported head size"):
        _run_paged_attention_rocm(head_size, dtype, num_heads)


# ---------------------------------------------------------------------------
# Triton Attention decode (head sizes: any, test common ones)
# ---------------------------------------------------------------------------

TRITON_ATTN_HEAD_SIZES = [32, 64, 80, 96, 128, 192, 256]


@pytest.mark.parametrize("head_size", TRITON_ATTN_HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads", [(16, 16), (16, 4)])
@torch.inference_mode()
def test_triton_attn_decode_head_sizes(head_size, dtype, num_heads):
    """Test Triton decode attention accuracy across head sizes."""
    from vllm.v1.attention.ops.triton_decode_attention import (
        decode_attention_fwd,
    )

    torch.set_default_device("cuda")
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    batch_size = 4
    seq_lens = [128, 256, 384, 512]

    query = torch.randn(batch_size, num_query_heads, head_size, dtype=dtype)
    # Triton decode expects [num_blocks, page_size, num_kv_heads, head_size]
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    scale = head_size**-0.5
    max_kv_len = max(seq_lens)
    max_num_blocks_per_seq = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, NUM_BLOCKS, (batch_size, max_num_blocks_per_seq), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    output = torch.zeros(batch_size, num_query_heads, head_size, dtype=dtype)
    lse = torch.zeros(batch_size, num_query_heads, dtype=dtype)

    num_kv_splits = 4
    attn_logits = torch.empty(
        batch_size,
        num_query_heads,
        num_kv_splits,
        head_size + 1,
        dtype=torch.float32,
    )

    # Triton decode attention uses combined KV cache [blocks, page, heads, d]
    # but key and value as separate caches for the reference
    # The kernel expects req_to_token = block_table
    decode_attention_fwd(
        query,
        key_cache,
        value_cache,
        output,
        lse,
        block_tables,
        seq_lens_tensor,
        attn_logits,
        num_kv_splits,
        scale,
        BLOCK_SIZE,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * batch_size,
        kv_lens=seq_lens,
        block_tables=block_tables,
        scale=scale,
    )

    _assert_close_or_xfail(
        f"triton_decode[hs={head_size},dt={dtype},heads={num_heads}]",
        output,
        ref_output,
        atol=1e-5,
        rtol=1e-5,
    )
