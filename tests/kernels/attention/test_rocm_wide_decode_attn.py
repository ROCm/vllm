# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the gfx1151 wide MHA decode-attention kernel.

Validates ``torch.ops._rocm_C.wide_decode_attn`` against an explicit fp32
attention over the paged KV cache, plus the host-side predicate and the
gather/scatter that packs a varlen decode batch into the kernel's fixed-M block.

The reference is deliberately not fused and not fast: it exists to catch a
kernel that is quick and wrong, which is the live risk here. Two cases get
specific attention because both were silently broken before and neither is
reachable from the standalone benchmark harness:

* **num_seqs > 1** -- Q and the output used to be indexed without the sequence
  index, so a batch read one sequence's Q and wrote one sequence's rows. The
  sequences here are given DIFFERENT lengths on purpose: equal ones would still
  pass if the batch collapsed onto one sequence.
* **head counts whose tuned KV segment count is not a power of two** (40, 48,
  56 give 6, 5, 4). The reduce used to be templated on the count and only
  instantiated {1,2,4,8,16}, returning -1 for the rest -- after the split kernel
  had launched, leaving the output stale. Hq=40 at head_size=128 is
  Llama-2-13B, Qwen-14B, OPT-13B and Baichuan-13B.

Run ``pytest tests/kernels/attention/test_rocm_wide_decode_attn.py``.
"""

import pytest
import torch

# The op is compiled into _rocm_C on all ROCm builds but has a real body only on
# gfx11 (stub elsewhere). Probe op presence before importing rocm.py so non-ROCm
# platforms skip without touching ROCm-only platform code.
try:
    import vllm._rocm_C  # noqa: F401

    _have_op = hasattr(torch.ops._rocm_C, "wide_decode_attn")
except Exception:
    _have_op = False
if not _have_op:
    pytest.skip("_rocm_C.wide_decode_attn not available", allow_module_level=True)

from vllm.platforms.rocm import on_gfx1151

if not on_gfx1151():
    pytest.skip("requires gfx1151 (RDNA3.5)", allow_module_level=True)

from vllm.v1.attention.ops import rocm_wide_decode_attn  # noqa: E402

BLOCK_SIZE = 16

# fp16 keeps 11 mantissa bits, bf16 only 8, so the same algorithm lands about an
# order of magnitude further out in bf16. Both accumulate in fp32; the gap is
# input quantization, not a different computation.
TOLERANCE = {torch.float16: 5e-3, torch.bfloat16: 4e-2}


def _tuned_nseg(head_size: int, num_kv_heads: int) -> int:
    return rocm_wide_decode_attn.tuned_num_kv_segments(1, head_size, num_kv_heads)


def _reference(q, k_cache, v_cache, block_table, seq_lens, scale, softcap=0.0):
    """Explicit fp32 attention over the paged KV, one sequence at a time.

    ``seq_lens[i]`` is the TOTAL length including the M new query tokens, so row
    ``m`` of a block attends to ``[0, seq_len - M + m]``.

    A row can be entirely masked when ``seq_len < M`` -- with seq_len=1 and M=4
    the first three rows see nothing. ``torch.softmax`` over an all -inf row is
    NaN, while the kernel defines that case as zero (an all -inf running max is
    clamped before the rescale, so no NaN is produced). Zero is the useful
    answer: those rows are query-padding and get discarded. So the reference
    zeroes them rather than propagating NaN, otherwise this asserts against the
    reference's own degenerate case instead of against the kernel.
    """
    num_seqs, m, num_heads, head_size = q.shape
    out = torch.empty_like(q, dtype=torch.float32)
    for s in range(num_seqs):
        ctx = int(seq_lens[s])
        nb = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        blocks = block_table[s, :nb].tolist()
        kf = torch.cat([k_cache[b] for b in blocks]).reshape(-1, num_heads, head_size)
        vf = torch.cat([v_cache[b] for b in blocks]).reshape(-1, num_heads, head_size)
        kf, vf = kf[:ctx].float(), vf[:ctx].float()
        pos = torch.arange(ctx, device=q.device)
        qpos = (ctx - m) + torch.arange(m, device=q.device)
        for h in range(num_heads):
            sc = (q[s, :, h, :].float() @ kf[:, h, :].T) * scale
            if softcap > 0:
                sc = softcap * torch.tanh(sc / softcap)
            sc = sc.masked_fill(pos[None, :] > qpos[:, None], float("-inf"))
            probs = torch.softmax(sc, dim=-1)
            # Fully masked rows: match the kernel's zero instead of NaN.
            probs = torch.where(qpos[:, None] >= 0, probs, probs.new_zeros(()))
            out[s, :, h, :] = probs @ vf[:, h, :]
    return out


def _run(num_heads, head_size, m, ctxs, dtype, softcap=0.0, seed=0):
    """One launch over len(ctxs) sequences; returns (kernel_out, reference)."""
    torch.manual_seed(seed)
    dev = "cuda"
    num_seqs = len(ctxs)
    scale = head_size**-0.5
    nb_per = [(c + BLOCK_SIZE - 1) // BLOCK_SIZE for c in ctxs]
    max_nb = max(nb_per)

    # vLLM's layout: one [num_blocks, 2, block_size, num_kv_heads, head_size]
    # allocation, split into k/v views, so block_stride comes from stride(0).
    kv = torch.randn(
        sum(nb_per), 2, BLOCK_SIZE, num_heads, head_size, device=dev, dtype=dtype
    )
    k_cache, v_cache = kv.unbind(1)

    block_table = torch.zeros(num_seqs, max_nb, device=dev, dtype=torch.int32)
    off = 0
    for s, n in enumerate(nb_per):
        block_table[s, :n] = torch.arange(off, off + n, device=dev, dtype=torch.int32)
        off += n
    seq_lens = torch.tensor(ctxs, device=dev, dtype=torch.int32)

    q = torch.randn(num_seqs, m, num_heads, head_size, device=dev, dtype=dtype)
    out = torch.zeros_like(q)

    nseg = _tuned_nseg(head_size, num_heads)
    po = torch.empty(
        num_seqs, num_heads, nseg, m, head_size, device=dev, dtype=torch.float32
    )
    pm = torch.empty(num_seqs, num_heads, nseg, m, device=dev, dtype=torch.float32)
    ps = torch.empty(num_seqs, num_heads, nseg, m, device=dev, dtype=torch.float32)

    torch.ops._rocm_C.wide_decode_attn(
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        po,
        pm,
        ps,
        scale,
        softcap,
    )
    torch.accelerator.synchronize()
    return out, _reference(q, k_cache, v_cache, block_table, seq_lens, scale, softcap)


@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("head_size", [64, 128, 256, 512])
@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_shapes(num_heads, head_size, m, dtype):
    """The tuned grid: head counts whose segment count is already a power of 2."""
    out, ref = _run(num_heads, head_size, m, [1024], dtype)
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max().item() < TOLERANCE[dtype]


@pytest.mark.parametrize("num_heads", [40, 48, 56])
@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("m", [1, 4])
def test_non_power_of_two_segments(num_heads, head_size, m):
    """Head counts whose tuned segment count is not a power of two.

    These used to leave the output untouched: the reduce dispatch returned -1
    after the split kernel had already launched.
    """
    nseg = _tuned_nseg(head_size, num_heads)
    out, ref = _run(num_heads, head_size, m, [2048], torch.float16)
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max().item() < TOLERANCE[torch.float16], (
        f"num_heads={num_heads} head_size={head_size} nseg={nseg}"
    )


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("num_seqs", [2, 5])
def test_batched_unequal_lengths(m, num_seqs):
    """Several sequences of DIFFERENT lengths in one launch.

    Equal lengths would still pass if the kernel collapsed the batch onto one
    sequence's Q, which is exactly what it used to do.
    """
    ctxs = [512 + 137 * i for i in range(num_seqs)]
    out, ref = _run(32, 128, m, ctxs, torch.float16)
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max().item() < TOLERANCE[torch.float16]
    # Distinct sequences must produce distinct output; a collapsed batch would
    # make these equal.
    if num_seqs > 1:
        assert not torch.allclose(out[0], out[1])


@pytest.mark.parametrize("ctx", [1, 4, 15, 16, 17, 127, 4097])
def test_context_edges(ctx):
    """Contexts that are not multiples of the block or the burst, and ctx < M.

    At ctx < M the leading rows are entirely masked; the reduce must turn that
    into zeros rather than NaN.
    """
    out, ref = _run(16, 128, 4, [ctx], torch.float16)
    assert torch.isfinite(out).all(), f"non-finite output at ctx={ctx}"
    assert (out.float() - ref).abs().max().item() < TOLERANCE[torch.float16]


def test_softcap():
    out, ref = _run(16, 128, 1, [1024], torch.float16, softcap=30.0)
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max().item() < TOLERANCE[torch.float16]


def test_rejects_gqa():
    """MHA only: the kernel maps one query head per KV head by construction."""
    dev = "cuda"
    q = torch.randn(1, 1, 16, 128, device=dev, dtype=torch.float16)
    kv = torch.randn(4, 2, BLOCK_SIZE, 4, 128, device=dev, dtype=torch.float16)
    k, v = kv.unbind(1)
    bt = torch.zeros(1, 4, device=dev, dtype=torch.int32)
    sl = torch.tensor([64], device=dev, dtype=torch.int32)
    po = torch.empty(1, 16, 8, 1, 128, device=dev, dtype=torch.float32)
    pm = torch.empty(1, 16, 8, 1, device=dev, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="MHA-only"):
        torch.ops._rocm_C.wide_decode_attn(
            q.clone(), q, k, v, bt, sl, po, pm, pm.clone(), 0.088, 0.0
        )


def test_rejects_mismatched_segment_count():
    """Partial buffers sized for a different segment count must raise.

    This is the failure the C++ check exists for: the partials are indexed
    [num_seqs, num_heads, nseg, M, head_size], so a silent disagreement would be
    a memory fault rather than an exception.
    """
    dev = "cuda"
    num_heads, head_size = 32, 128
    wrong = _tuned_nseg(head_size, num_heads) + 1
    q = torch.randn(1, 1, num_heads, head_size, device=dev, dtype=torch.float16)
    kv = torch.randn(
        4, 2, BLOCK_SIZE, num_heads, head_size, device=dev, dtype=torch.float16
    )
    k, v = kv.unbind(1)
    bt = torch.zeros(1, 4, device=dev, dtype=torch.int32)
    sl = torch.tensor([64], device=dev, dtype=torch.int32)
    po = torch.empty(1, num_heads, wrong, 1, head_size, device=dev, dtype=torch.float32)
    pm = torch.empty(1, num_heads, wrong, 1, device=dev, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="KV segments"):
        torch.ops._rocm_C.wide_decode_attn(
            q.clone(), q, k, v, bt, sl, po, pm, pm.clone(), 0.088, 0.0
        )


def test_rejects_strided_partials():
    """A strided partial buffer must raise, not read the wrong elements.

    The workspace is allocated once at the worst case (M=4, max_num_seqs) and
    reused for smaller launches. Slicing the M dimension off it --
    ``buf[:n, :, :, :1]`` -- keeps the parent's strides, and since the kernels
    index the partials with raw pointer arithmetic, that reads the wrong
    elements silently. The backend carves a dense view instead; this pins the
    guard that catches it if that ever regresses.
    """
    dev = "cuda"
    num_heads, head_size = 32, 128
    nseg = _tuned_nseg(head_size, num_heads)
    q = torch.randn(1, 1, num_heads, head_size, device=dev, dtype=torch.float16)
    kv = torch.randn(
        4, 2, BLOCK_SIZE, num_heads, head_size, device=dev, dtype=torch.float16
    )
    k, v = kv.unbind(1)
    bt = torch.zeros(1, 4, device=dev, dtype=torch.int32)
    sl = torch.tensor([64], device=dev, dtype=torch.int32)

    # Allocated at M=4, then sliced down to M=1: the shape is right and the
    # strides are not.
    po_full = torch.empty(
        2, num_heads, nseg, 4, head_size, device=dev, dtype=torch.float32
    )
    pm_full = torch.empty(2, num_heads, nseg, 4, device=dev, dtype=torch.float32)
    po, pm, ps = po_full[:1, :, :, :1], pm_full[:1, :, :, :1], pm_full[:1, :, :, :1]
    assert not po.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops._rocm_C.wide_decode_attn(
            q.clone(), q, k, v, bt, sl, po, pm, ps, 0.088, 0.0
        )


def test_kv_cache_from_unbind_is_accepted():
    """k/v views from a [num_blocks, 2, ...] allocation are strided by design.

    They must NOT be rejected as non-contiguous: the gap between blocks is what
    block_stride carries. Only density *within* a block is required.
    """
    out, ref = _run(16, 128, 1, [512], torch.float16)
    assert (out.float() - ref).abs().max().item() < TOLERANCE[torch.float16]


@pytest.mark.parametrize("lens", [[1, 3, 2, 4], [1, 1, 1], [4], [2, 2], [1, 4, 1]])
def test_gather_scatter_roundtrip(lens):
    """The varlen -> fixed-M packing must be a bijection on the real tokens."""
    m = rocm_wide_decode_attn.KERNEL_M
    num_seqs, total = len(lens), sum(lens)
    qsl = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist())
    rows, scatter_rows = rocm_wide_decode_attn.build_gather_index(
        qsl, num_seqs, scratch_row=total
    )

    # Fixed shape, independent of the query lengths: a data-dependent shape
    # would sync and would not be cudagraph-capturable.
    assert rows.shape == (num_seqs * m,)
    assert scatter_rows.shape == (num_seqs * m,)

    q = torch.arange(total * 6, dtype=torch.float32).reshape(total, 2, 3)
    padded = q.index_select(0, rows).view(num_seqs, m, 2, 3)

    # Real tokens are the LAST len_i rows of each block, in order; padding rows
    # are aimed at the scratch slot.
    dest_rows = scatter_rows.view(num_seqs, m)
    for i, length in enumerate(lens):
        assert (dest_rows[i, : m - length] == total).all()
        assert (dest_rows[i, m - length :] != total).all()

    # Identity "kernel": scattering back must reproduce the input exactly, and
    # every real row must be written (no NaN survives).
    out = torch.full((total + 1, 2, 3), float("nan"))
    out.index_copy_(0, scatter_rows, padded.view(num_seqs * m, 2, 3))
    assert torch.equal(out[:total], q)


def test_predicate_rejects_unsupported():
    """The host predicate must screen everything the kernel cannot do."""
    base = dict(
        num_heads=32,
        num_kv_heads=32,
        head_size=128,
        max_query_len=1,
        dtype=torch.float16,
        kv_quant_mode=rocm_wide_decode_attn.KVQuantMode.NONE,
        alibi_slopes=None,
        sinks=None,
        sliding_window=None,
        output_scale=None,
        kv_cache_layout="NHD",
    )
    assert rocm_wide_decode_attn.can_run(**base)

    slopes = torch.zeros(32)
    for override in (
        {"num_kv_heads": 8},  # GQA
        {"head_size": 80},  # not instantiated
        {"head_size": 96},  # multiple of 32 but not of 64
        {"max_query_len": 5},  # beyond the padded M
        {"max_query_len": 0},
        {"dtype": torch.float32},
        {"alibi_slopes": slopes},
        {"sinks": slopes},
        {"sliding_window": 1024},
        {"output_scale": torch.ones(1)},
        {"kv_cache_layout": "HND"},
        {"causal": False},
    ):
        assert not rocm_wide_decode_attn.can_run(**{**base, **override}), override

    # Query lengths 2 and 3 are padded up to 4 rather than rejected.
    for qlen in (2, 3, 4):
        assert rocm_wide_decode_attn.can_run(**{**base, "max_query_len": qlen})

    # Every real window is rejected, including 1. The Impl stores a window W as
    # (W-1, 0), so W=1 becomes (0, 0) -- a `> 0` test on the stored extent would
    # let it through and the kernel would attend to the whole context.
    for window in (1, 2, 1024):
        assert not rocm_wide_decode_attn.can_run(
            **{**base, "sliding_window": window}
        ), window

    # Shapes the kernel is instantiated for but measurably loses on are declined
    # too, so Triton keeps the work. head_size 64 with 32 heads runs at
    # 0.81-0.86x of Triton on gfx1151 -- the DPP reduction dominates the inner
    # loop at that head size, and this is the head count where it bites.
    assert not rocm_wide_decode_attn.can_run(
        **{**base, "num_heads": 32, "num_kv_heads": 32, "head_size": 64}
    )
    # ...but only at M=1. Query lengths 2-4 run on the M=4 kernel, which wins
    # that shape (1.19-1.42x of Triton), so they stay enabled.
    for qlen in (2, 3, 4):
        assert rocm_wide_decode_attn.can_run(
            **{
                **base,
                "num_heads": 32,
                "num_kv_heads": 32,
                "head_size": 64,
                "max_query_len": qlen,
            }
        ), qlen
    # Neighbouring head counts at the same head size do win, and stay enabled.
    for heads in (8, 16, 64):
        assert rocm_wide_decode_attn.can_run(
            **{**base, "num_heads": heads, "num_kv_heads": heads, "head_size": 64}
        ), heads
