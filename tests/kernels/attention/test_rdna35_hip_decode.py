# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA3.5 HIP decode attention: agreement with a reference, and honest fallback.

The kernel walks the paged KV cache with its own address arithmetic rather than
reading the tensor's strides, so a layout it did not expect yields finite,
wrong numbers instead of an error. These tests therefore cover what a plain
tolerance check would miss:

- every variant built into _rocm_C at a short context, where a causal
  off-by-one is visible: the error from one masked-off-by-one key falls off as
  ~1/S while the tolerance is fixed, so long contexts hide the bug most likely
  to be present.
- several launches on one scratch, each on new inputs: a split-KV merge that
  reads stale partials, or skips part of the output, reproduces the previous
  launch's correct numbers.
- that an unsupported shape falls back to Triton rather than being served
  wrong.
"""

import csv
import itertools

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")

rdna35 = pytest.importorskip("vllm.v1.attention.ops.rdna35_hip_decode")

HQ, HKV, HEAD_DIM = 32, 16, 256
# Relative error bound per dtype.  The reference sees the same rounded inputs,
# so what differs is the kernel's arithmetic and its output rounding -- the
# latter alone up to 2^-8 relative in bf16.
TOL = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
DTYPES = list(TOL)


def _skip_unless_gfx1151():
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        pytest.skip("kernel is built for gfx1151")


def _built(dtype=torch.float16, batch=0, **want):
    """The first built-in variant with these properties, loaded.

    Args:
        dtype: Element type.
        batch: 1 for a batch build.
        **want: Predicates on KernelVariant fields, by name.

    Returns:
        (variant, module).

    """
    _skip_unless_gfx1151()
    for v in rdna35.variants():
        if (
            v.dtype == dtype
            and v.batched == batch
            and all(pred(getattr(v, k)) for k, pred in want.items())
        ):
            module = rdna35.load(v)
            assert module is not None, f"{v.name} is not in _rocm_C"
            return v, module
    raise AssertionError(f"no built-in variant with {sorted(want)}")


def _paged_inputs(seq_len, v, seed=0):
    """An HND KV cache of whole pages, the last one partly used."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    num_blocks = -(-seq_len // v.page_size)
    kv = torch.randn(
        num_blocks,
        v.num_kv_heads,
        v.page_size,
        2 * v.head_size,
        device=dev,
        dtype=v.dtype,
    )
    kv = kv * 0.5
    q = torch.randn(
        v.max_query_len, v.num_q_heads, v.head_size, device=dev, dtype=v.dtype
    )
    return q * 0.5, kv, torch.arange(num_blocks, device=dev, dtype=torch.int32)


def _reference(q, kv, seq_len, window=0):
    m, hq, hkv, hd = q.shape[0], q.shape[1], kv.shape[1], q.shape[2]
    flat = kv.transpose(1, 2).reshape(-1, hkv, 2 * hd)[:seq_len]
    k, v = flat[..., :hd], flat[..., hd:]
    gqa = hq // hkv
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    vf = v.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * (hd**-0.5)
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - m + torch.arange(m, device=q.device)).view(m, 1)
    masked = pos > lim
    if window:
        masked |= pos < lim - (window - 1)
    scores = scores.masked_fill(masked.view(1, m, seq_len), float("-inf"))
    return torch.bmm(torch.softmax(scores, -1), vf).permute(1, 0, 2)


def _max_rel(got, ref) -> float:
    """Relative error with a floor, which is what catches a causal off-by-one.

    max_abs alone does not: one key masked wrongly gives max_abs=1.2e-03 at S=2048
    and slips past a 2e-2 absolute threshold.
    """
    return ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()


def _launch(module, q, kv, block_table, out, scratch, seq_len):
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.decode_attn(q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5)


def _check_one(v, module, lens):
    """One sequence per launch, all on one scratch, each on new inputs and
    against its own reference: a split merge that leaves part of the output to
    stale data passes once."""
    scratch = rdna35.make_scratch(v, torch.device("cuda"))
    for seed, seq_len in enumerate(lens):
        q, kv, block_table = _paged_inputs(seq_len, v, seed)
        out = torch.full_like(q, float("nan"))
        _launch(module, q, kv, block_table, out, scratch, seq_len)
        torch.accelerator.synchronize()
        ref = _reference(q, kv, seq_len, v.window)
        got = out.float().nan_to_num(1e9)
        assert _max_rel(got, ref) <= TOL[v.dtype], f"{v.name}, S={seq_len}"


def _check_batch(v, module, lens):
    """Launch a batch build over sequences of these lengths and check each
    against its own reference, twice, a padded one (S=0) left unwritten."""
    width = max(lens) // v.page_size + 1
    parts = [_paged_inputs(max(s, 16), v, seed=i) for i, s in enumerate(lens)]
    kv = torch.cat([p[1] for p in parts])
    first = torch.tensor([0] + [p[1].shape[0] for p in parts]).cumsum(0)
    bt = torch.zeros(len(lens), width, device=kv.device, dtype=torch.int32)
    for i, p in enumerate(parts):
        bt[i, : p[1].shape[0]] = p[2] + int(first[i])
    q = torch.cat([p[0] for p in parts])
    scratch = rdna35.make_scratch(v, q.device, max_seqs=len(lens))
    seq_lens = torch.tensor(lens, device=q.device, dtype=torch.int32)
    out = torch.full_like(q, 7.0)
    for _ in range(2):
        module.decode_attn(q, kv, bt, out, *scratch, seq_lens, q.shape[2] ** -0.5)
    torch.accelerator.synchronize()
    for i, s in enumerate(lens):
        rows = out[i * v.max_query_len : (i + 1) * v.max_query_len].float()
        if s == 0:
            assert (rows == 7.0).all(), f"{v.name}: a padded sequence was written"
            continue
        ref = _reference(parts[i][0], parts[i][1], s, v.window)
        assert _max_rel(rows, ref) <= TOL[v.dtype], f"{v.name}: sequence {i}, S={s}"


def _unit(v):
    """The translation unit a variant is built in (generate_rdna35_attn.py)."""
    dt = "bf16" if v.dtype == torch.bfloat16 else "fp16"
    return f"q{v.num_q_heads}_kv{v.num_kv_heads}_d{v.head_size}_w{v.window}_{dt}"


def test_variants_csv_is_well_formed():
    """The CSV CMake builds from and the backend looks up in: its columns are
    the fields the loader maps, one row per key, and every configuration at
    every M the backend sends, in both dtypes, for one sequence and a batch."""
    with rdna35.VARIANTS_CSV.open(newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    assert set(header) == set(rdna35._FIELDS) | {"BF16"}
    assert all(len(r) == len(header) and all(x.isdigit() for x in r) for r in body)
    variants = rdna35.variants()
    assert len(variants) == len(body), "two rows share a key"
    configs = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.window, v.page_size)
        for v in variants
    }
    for hq, hkv, d, window, bs in configs:
        assert d in rdna35.SUPPORTED_HEAD_SIZES and hq % hkv == 0
        for m, dtype, batch in itertools.product(
            range(1, rdna35.MAX_M + 1), DTYPES, (False, True)
        ):
            v = rdna35.variant_for(hq, hkv, d, m, bs, 1, window, dtype, batch)
            assert v is not None, (hq, hkv, d, window, bs, m, dtype, batch)
            # A batch fills the machine, where the dot decomposition loses.
            assert not (batch and v.dot_product), v.name


def test_every_listed_variant_is_built_in():
    _skip_unless_gfx1151()
    missing = [v.name for v in rdna35.variants() if rdna35.load(v) is None]
    assert not missing, f"{len(missing)} variants not in _rocm_C, e.g. {missing[0]}"


@pytest.mark.parametrize("unit", sorted({_unit(v) for v in rdna35.variants()}))
def test_built_in_variants_match_reference(unit):
    """Every variant built into _rocm_C, launched through the registry: one
    sequence at a short context and twice past a partial tile, on one scratch,
    and a batch with a padded sequence.  Windows run past the window, so that
    it masks; large pages end partly filled."""
    _skip_unless_gfx1151()
    for v in rdna35.variants():
        if _unit(v) != unit:
            continue
        module = rdna35.load(v)
        assert module is not None, v.name
        s = max(1024, v.window + 64, v.page_size + 1000)
        if v.batched:
            _check_batch(v, module, [s, 48, 0])
        else:
            _check_one(v, module, [48, s, s + 1000])


@pytest.mark.parametrize(
    "want",
    [
        {
            "window": bool,
            "dot_product": lambda x: not x,
            "max_segments": lambda x: x > 1,
        },
        {
            "window": bool,
            "dot_product": lambda x: not x,
            "max_segments": lambda x: x == 1,
        },
        {"window": bool, "dot_product": bool},
    ],
    ids=["wmma segments", "wmma", "dot_product"],
)
def test_sliding_window_never_reads_freed_pages(want):
    """Pages wholly before the window may be freed; their contents must not
    reach the output even as 0 * NaN.

    S=2016 puts the window's first key late in its WMMA block, so that block's
    first tile is a whole freed page: with the V zeroing removed the WMMA
    cases fail.  (The dot path's tiles never leave the first key's page.)
    """
    v, module = _built(**want)
    seq_len = 2016
    q, kv, block_table = _paged_inputs(seq_len, v)
    ref = _reference(q, kv, seq_len, v.window)
    # vLLM frees the pages no query's window reaches and points their table
    # entries elsewhere: here, at a page of NaN the kernel must never let into
    # the output.
    first = max(0, seq_len - v.max_query_len - (v.window - 1)) // v.page_size
    kv = torch.cat([kv, torch.full_like(kv[:1], float("nan"))])
    block_table = block_table.clone()
    block_table[:first] = kv.shape[0] - 1
    out = torch.empty_like(q)
    _launch(module, q, kv, block_table, out, rdna35.make_scratch(v, q.device), seq_len)
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()
    assert _max_rel(out.float(), ref) <= TOL[v.dtype]


def test_graph_replay_follows_seq_lens():
    """S is read on the device, so a captured graph serves a longer sequence.

    A host int is frozen into the graph at capture: every replay would attend
    over the capture-time length, which is what full CUDA-graph decode does.
    """
    v, module = _built(
        window=lambda x: not x,
        dot_product=lambda x: not x,
        max_segments=lambda x: x > 1,
    )
    long_s = 1024
    q, kv, block_table = _paged_inputs(long_s, v)
    scratch = rdna35.make_scratch(v, q.device)
    out = torch.empty_like(q)
    seq_lens = torch.tensor([48], device=q.device, dtype=torch.int32)

    def launch():
        module.decode_attn(
            q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5
        )

    launch()  # warm-up outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for s in (48, 208, long_s):
        seq_lens.fill_(s)
        out.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        ref = _reference(q, kv, s)
        assert _max_rel(out.float(), ref) <= TOL[v.dtype], f"S={s}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_other_dtype_is_refused_not_miscomputed(dtype):
    """A build must refuse the other 16-bit type, not return nonsense.

    The loads reinterpret the tensors as the element type the variant was
    built for, so fp16 read as bf16 or the reverse stays finite and is wrong
    by orders of magnitude.
    """
    other = next(d for d in DTYPES if d != dtype)
    v, module = _built(dtype=other)
    q, kv, block_table = _paged_inputs(1024, v)
    q, kv = q.to(dtype), kv.to(dtype)
    with pytest.raises(RuntimeError, match="kernel built for"):
        _launch(
            module,
            q,
            kv,
            block_table,
            torch.empty_like(q),
            rdna35.make_scratch(v, q.device),
            1024,
        )


def test_unsupported_head_size_falls_back_to_triton():
    """An unbuilt shape must be served by Triton, not served wrong."""
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionRdna35Impl
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = RocmAttentionRdna35Impl.__new__(RocmAttentionRdna35Impl)
    impl._rejected = None
    # 96 is a real shipped head size (Phi-3.5-vision) and is deliberately not
    # built: 96/32 is 3 fp16 per lane, which is not a power of two and so has
    # no single load width. It must fall back, not be served wrong.
    impl.head_size, impl.num_heads, impl.num_kv_heads = 96, HQ, HKV

    fits = impl._prepare(
        kv_cache=torch.empty(0, dtype=torch.float16),
        q=torch.empty(0, dtype=torch.float16),
        alibi_slopes=None,
        sinks=None,
        softcap=0,
        causal=True,
        window_size=None,
        kv_quant_mode=KVQuantMode.NONE,
        seqused_k=torch.zeros(1),
        max_seqlen_q=0,
    )
    assert not fits
    assert "head_size 96" in impl._rejected


@pytest.mark.parametrize(
    "nseq, tokens, max_q, reason",
    [
        (3, 5, 4, "unequal query length"),  # a prefill mixed with decodes
        (1, 64, 64, "more than"),  # a prompt: one build per length otherwise
    ],
)
def test_non_decode_batches_fall_back_to_triton(nseq, tokens, max_q, reason):
    """Only batches of equal, decode-sized query lengths reach the kernel."""
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionRdna35Impl
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = RocmAttentionRdna35Impl.__new__(RocmAttentionRdna35Impl)
    impl._rejected = None
    impl.head_size, impl.num_heads, impl.num_kv_heads = HEAD_DIM, HQ, HKV
    fits = impl._prepare(
        kv_cache=torch.empty(0, dtype=torch.float16),
        q=torch.empty(tokens, HQ, HEAD_DIM, dtype=torch.float16),
        alibi_slopes=None,
        sinks=None,
        softcap=0,
        causal=True,
        window_size=None,
        kv_quant_mode=KVQuantMode.NONE,
        seqused_k=torch.zeros(nseq),
        max_seqlen_q=max_q,
    )
    assert not fits
    assert reason in impl._rejected


def test_rocm_attn_ahead_of_triton_on_gfx1151(monkeypatch):
    """On gfx1151 ROCM_ATTN, which is this kernel there, comes before Triton,
    which serves what it does not."""
    import vllm.platforms.rocm as rocm
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as B

    monkeypatch.setattr(rocm, "on_gfx1x", lambda: True)
    monkeypatch.setattr(rocm, "on_gfx1151", lambda: True)
    order = rocm._get_backend_priorities(use_mla=False, use_sparse=False)
    assert order.index(B.ROCM_ATTN) < order.index(B.TRITON_ATTN)


@pytest.mark.parametrize("gfx1151", [True, False])
def test_rocm_attn_layout_and_impl_only_on_gfx1151(gfx1151, monkeypatch):
    """On gfx1151 ROCM_ATTN runs this kernel on Triton's packed HND cache, K
    and V in the content of one head; elsewhere it keeps its K and V head
    groups and its impl.  ROCM_AITER_UNIFIED_ATTN, its subclass, keeps its own
    on both."""
    import vllm.platforms.rocm as rocm
    from vllm.v1.attention.backends import rocm_attn
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend as Aiter,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout

    monkeypatch.setattr(rocm, "on_gfx1151", lambda: gfx1151)
    backend = rocm_attn.RocmAttentionBackend
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=HKV, head_size=HEAD_DIM, dtype=torch.float16
    )
    if gfx1151:
        assert backend.get_impl_cls() is rocm_attn.RocmAttentionRdna35Impl
        assert backend.get_builder_cls() is rocm_attn.RocmAttentionRdna35MetadataBuilder
        assert backend.customize_spec(spec) == TritonAttentionBackend.customize_spec(
            spec
        )
        assert backend.supported_kv_cache_layouts() == (KVCacheLayout.LBHNC,)
        assert 512 in backend.get_supported_head_sizes()
    else:
        assert backend.get_impl_cls() is rocm_attn.RocmAttentionImpl
        assert backend.customize_spec(spec).num_head_slots == 2
        assert KVCacheLayout.LHBNC in backend.supported_kv_cache_layouts()
        assert 512 not in backend.get_supported_head_sizes()
    assert 512 not in Aiter.get_supported_head_sizes()
