# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA3.5 decode and prefill attention: accuracy against a reference, and
fallback to Triton for unsupported calls."""

import itertools

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")

hip = pytest.importorskip("vllm.v1.attention.ops.rocm_rdna_singleseq_hip")

HQ, HKV, HEAD_DIM = 32, 16, 256
# Relative error bound per dtype.
TOL = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
DTYPES = list(TOL)


def _skip_unless_gfx1151():
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        pytest.skip("kernel is built for gfx1151")


def _built(dtype=torch.float16, **want):
    """The first built variant matching `want`, and its module."""
    _skip_unless_gfx1151()
    for v in hip.variants():
        if v.dtype == dtype and all(pred(getattr(v, k)) for k, pred in want.items()):
            module = hip.load(v)
            assert module is not None, f"{v.name} is not in _rocm_C"
            return v, module
    raise AssertionError(f"no built-in variant with {sorted(want)}")


def _paged_inputs(seq_len, v, seed=0):
    """An HND KV cache whose last page is partly used."""
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
    """Relative error with a floor."""
    return ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()


def _launch(module, q, kv, block_table, out, scratch, seq_len):
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.decode_attn(q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5)


def _check_one(v, module, lens):
    """Launches sharing one scratch, each on new inputs."""
    scratch = hip.make_scratch(v, torch.device("cuda"))
    for seed, seq_len in enumerate(lens):
        q, kv, block_table = _paged_inputs(seq_len, v, seed)
        out = torch.full_like(q, float("nan"))
        _launch(module, q, kv, block_table, out, scratch, seq_len)
        torch.accelerator.synchronize()
        ref = _reference(q, kv, seq_len, v.window)
        got = out.float().nan_to_num(1e9)
        assert _max_rel(got, ref) <= TOL[v.dtype], f"{v.name}, S={seq_len}"


def _unit(v):
    """A test case: the variants of one shape and dtype."""
    dt = "bf16" if v.dtype == torch.bfloat16 else "fp16"
    return f"q{v.num_q_heads}_kv{v.num_kv_heads}_d{v.head_size}_w{v.window}_{dt}"


def test_decode_configs_are_well_formed():
    """No two decode builds share a shape, and every built shape covers every M
    in both dtypes."""
    variants = hip.variants()
    if not variants:
        pytest.skip("_rocm_C carries no RDNA3.5 variants")
    assert len({hip._lookup_key(v) for v in variants}) == len(variants), (
        "two builds share a shape"
    )
    configs = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.window, v.page_size)
        for v in variants
    }
    for hq, hkv, d, window, bs in configs:
        assert d in hip.SUPPORTED_HEAD_SIZES and hq % hkv == 0
        for m, dtype in itertools.product(range(1, hip.MAX_M + 1), DTYPES):
            v = hip.variant_for(hq, hkv, d, m, bs, 1, window, dtype)
            assert v is not None, (hq, hkv, d, window, bs, m, dtype)


def test_every_listed_variant_is_built_in():
    _skip_unless_gfx1151()
    # Raises if a build has no instantiation unit.
    assert torch.ops._rocm_C.rdna_singleseq_decode_configs()
    missing = [v.name for v in hip.variants() if hip.load(v) is None]
    assert not missing, f"{len(missing)} variants not in _rocm_C, e.g. {missing[0]}"


@pytest.mark.parametrize("unit", sorted({_unit(v) for v in hip.variants()}))
def test_built_in_variants_match_reference(unit):
    """Every built decode variant matches the reference."""
    _skip_unless_gfx1151()
    for v in hip.variants():
        if _unit(v) != unit:
            continue
        module = hip.load(v)
        assert module is not None, v.name
        s = max(1024, v.window + 64, v.page_size + 1000)
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
    """Freed pages before the window do not reach the output."""
    v, module = _built(**want)
    seq_len = 2016
    q, kv, block_table = _paged_inputs(seq_len, v)
    ref = _reference(q, kv, seq_len, v.window)
    first = max(0, seq_len - v.max_query_len - (v.window - 1)) // v.page_size
    kv = torch.cat([kv, torch.full_like(kv[:1], float("nan"))])
    block_table = block_table.clone()
    block_table[:first] = kv.shape[0] - 1
    out = torch.empty_like(q)
    _launch(module, q, kv, block_table, out, hip.make_scratch(v, q.device), seq_len)
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()
    assert _max_rel(out.float(), ref) <= TOL[v.dtype]


def test_graph_replay_follows_seq_lens():
    """A captured graph follows the sequence length on the device."""
    v, module = _built(
        window=lambda x: not x,
        dot_product=lambda x: not x,
        max_segments=lambda x: x > 1,
    )
    long_s = 1024
    q, kv, block_table = _paged_inputs(long_s, v)
    scratch = hip.make_scratch(v, q.device)
    out = torch.empty_like(q)
    seq_lens = torch.tensor([48], device=q.device, dtype=torch.int32)

    def launch():
        module.decode_attn(
            q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5
        )

    launch()
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
    """A build refuses the other 16-bit dtype."""
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
            hip.make_scratch(v, q.device),
            1024,
        )


def _impl():
    from vllm.v1.attention.backends.rocm_rdna_singleseq_attn import (
        RocmRdnaSingleSeqAttentionImpl as Impl,
    )

    return Impl.__new__(Impl)


def test_unsupported_head_size_falls_back_to_triton():
    """An unbuilt head size falls back to Triton."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = _impl()
    impl._rejected = None
    impl.head_size, impl.num_heads, impl.num_kv_heads = 80, HQ, HKV

    fits = impl._prepare(
        kv_cache=torch.empty(0, dtype=torch.float16),
        q=torch.empty(0, dtype=torch.float16),
        out=torch.empty(0, dtype=torch.float16),
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
    assert "head_size 80" in impl._rejected


def test_strided_query_falls_back_to_triton():
    """A strided Q falls back to Triton."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = _impl()
    impl._rejected = None
    impl.head_size, impl.num_heads, impl.num_kv_heads = HEAD_DIM, HQ, HKV
    qkv = torch.empty(2, (HQ + 2 * HKV) * HEAD_DIM, dtype=torch.float16)
    q = qkv[:, : HQ * HEAD_DIM].view(-1, HQ, HEAD_DIM)
    fits = impl._prepare(
        kv_cache=torch.empty(0, dtype=torch.float16),
        q=q,
        out=torch.empty(2, HQ, HEAD_DIM, dtype=torch.float16),
        alibi_slopes=None,
        sinks=None,
        softcap=0,
        causal=True,
        window_size=None,
        kv_quant_mode=KVQuantMode.NONE,
        seqused_k=torch.zeros(1),
        max_seqlen_q=2,
    )
    assert not fits
    assert "not contiguous" in impl._rejected


@pytest.mark.parametrize(
    "nseq, tokens, max_q, reason",
    [
        (3, 3, 1, "3 sequences"),
        (1, 64, 64, "more than"),
    ],
)
def test_non_decode_batches_fall_back_to_triton(nseq, tokens, max_q, reason):
    """Only one sequence of at most MAX_M tokens reaches the decode kernel."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = _impl()
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


@pytest.mark.parametrize(
    "gfx1151, max_num_seqs, dtype, cache_dtype, head_sizes, expected",
    [
        (True, 1, torch.bfloat16, "auto", {256, 512}, True),
        (False, 1, torch.bfloat16, "auto", {256, 512}, False),
        (True, 2, torch.bfloat16, "auto", {256, 512}, False),
        (True, 1, torch.float32, "auto", {256, 512}, False),
        (True, 1, torch.bfloat16, "fp8", {256, 512}, False),
        (True, 1, torch.bfloat16, "auto", {256, 576}, False),
    ],
    ids=["serves", "not_gfx1151", "two_seqs", "fp32", "fp8_kv", "head_size"],
)
def test_gemma4_forces_singleseq_where_it_serves(
    gfx1151, max_num_seqs, dtype, cache_dtype, head_sizes, expected, monkeypatch
):
    """Gemma4's heterogeneous head sizes force one backend for every layer:
    ours where it serves them all, else Triton."""
    import types

    import vllm.platforms.rocm as rocm
    from vllm.model_executor.models.config import rdna_singleseq_serves

    monkeypatch.setattr(rocm, "on_gfx1151", lambda: gfx1151)
    config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=types.SimpleNamespace(dtype=dtype),
        cache_config=types.SimpleNamespace(cache_dtype=cache_dtype),
    )
    assert rdna_singleseq_serves(config, head_sizes) == expected


@pytest.mark.parametrize("gfx1151", [True, False])
def test_singleseq_is_first_on_gfx1151(gfx1151, monkeypatch):
    """gfx1151 tries the backend first; other GPUs never list it."""
    import vllm.platforms.rocm as rocm
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as B

    monkeypatch.setattr(rocm, "on_gfx1151", lambda: gfx1151)
    order = rocm._get_backend_priorities(use_mla=False, use_sparse=False)
    if gfx1151:
        assert order[0] == B.ROCM_RDNA_SINGLESEQ_ATTN
    else:
        assert B.ROCM_RDNA_SINGLESEQ_ATTN not in order


@pytest.mark.parametrize("max_num_seqs", [1, 2])
def test_singleseq_requires_one_sequence(max_num_seqs, monkeypatch):
    """The backend requires max_num_seqs=1 and uses Triton's KV cache."""
    import types

    from vllm.v1.attention.backends import rocm_rdna_singleseq_attn as rdna
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout

    config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs)
    )
    monkeypatch.setattr(rdna, "get_current_vllm_config_or_none", lambda: config)
    backend = rdna.RocmRdnaSingleSeqAttentionBackend
    reason = backend.supports_combination(
        HEAD_DIM, torch.float16, "auto", 16, False, False, False, False, None
    )
    assert reason == (None if max_num_seqs == 1 else "max_num_seqs must be 1")
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=HKV, head_size=HEAD_DIM, dtype=torch.float16
    )
    assert backend.customize_spec(spec) == TritonAttentionBackend.customize_spec(spec)
    assert backend.supported_kv_cache_layouts() == (KVCacheLayout.LBHNC,)
    assert 512 in backend.get_supported_head_sizes()


# ---------------------------------------------------------------- prefill


def _prefill_reference(q, kv, block_table, seq_len, window=0, mm_ranges=()):
    """Float reference of a prefill; a token in a multimodal range [start, end]
    also sees the range's later keys."""
    m, hq, hkv, hd = q.shape[0], q.shape[1], kv.shape[1], q.shape[2]
    pages = kv[block_table.long()]
    flat = pages.transpose(1, 2).reshape(-1, hkv, 2 * hd)[:seq_len].float()
    gqa = hq // hkv
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - m + torch.arange(m, device=q.device)).view(m, 1)
    masked = pos > lim
    for start, end in mm_ranges:
        q_in = (lim >= start) & (lim <= end)
        k_in = (pos >= start) & (pos <= end)
        masked &= ~(q_in & k_in)
    if window:
        masked |= pos < lim - (window - 1)
    out = torch.empty(m, hq, hd, device=q.device)
    for h0 in range(0, hq, 4):
        heads = range(h0, min(hq, h0 + 4))
        kvh = [h // gqa for h in heads]
        qf = q[:, h0 : h0 + len(heads)].float().permute(1, 0, 2)
        k = flat[:, kvh, :hd].permute(1, 0, 2)
        v = flat[:, kvh, hd:].permute(1, 0, 2)
        scores = torch.bmm(qf, k.transpose(1, 2)) * (hd**-0.5)
        scores = scores.masked_fill(masked.view(1, m, seq_len), float("-inf"))
        out[:, h0 : h0 + len(heads)] = torch.bmm(torch.softmax(scores, -1), v).permute(
            1, 0, 2
        )
    return out


def _mm_ranges_tensor(mm_ranges, device):
    """One sequence's ranges as TRITON_ATTN's (1, ranges, 2), padded with an
    empty range."""
    pairs = [*mm_ranges, (0, 0)]
    return torch.tensor([pairs], device=device, dtype=torch.int32)


def _triton_prefill(q, kv, block_table, seq_len, window=0, mm_ranges=()):
    """Triton's unified attention on the same cache."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    m, hd, hkv = q.shape[0], q.shape[2], kv.shape[1]
    k, v = kv.transpose(1, 2).split(hd, dim=-1)
    out = torch.empty_like(q)
    one = torch.ones(1, hkv, device=q.device)
    unified_attention(
        q,
        k,
        v,
        out,
        torch.tensor([0, m], device=q.device, dtype=torch.int32),
        m,
        torch.tensor([seq_len], device=q.device, dtype=torch.int32),
        seq_len,
        hd**-0.5,
        True,
        (window - 1, 0) if window else (-1, -1),
        block_table.view(1, -1),
        0,
        None,
        one,
        one,
        mm_prefix_range=(_mm_ranges_tensor(mm_ranges, q.device) if mm_ranges else None),
        mm_prefix_clamp_sliding_window=True,
    )
    return out


def _assert_prefill_close(got, q, kv, block_table, seq_len, window, what, mm_ranges=()):
    """Within twice Triton's error against the float reference."""
    ref = _prefill_reference(q, kv, block_table, seq_len, window, mm_ranges)
    err = (got.float().nan_to_num(1e9) - ref).abs().max().item()
    triton = _triton_prefill(q, kv, block_table, seq_len, window, mm_ranges)
    bound = 2 * (triton.float() - ref).abs().max().item() + 1e-4
    assert err <= bound, f"{what}: max error {err:.2e}, Triton's x2 {bound:.2e}"


def _prefill_inputs(v, m, seq_len, seed=0):
    """Q for m tokens and a shuffled HND cache of seq_len keys."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    num_blocks = -(-seq_len // v.page_size)
    kv = torch.randn(
        num_blocks + 2, v.num_kv_heads, v.page_size, 2 * v.head_size, device=dev
    )
    q = torch.randn(m, v.num_q_heads, v.head_size, device=dev)
    block_table = torch.randperm(num_blocks + 2, device=dev)[:num_blocks]
    return (
        (q * 0.5).to(v.dtype),
        (kv * 0.5).to(v.dtype),
        block_table.to(torch.int32).contiguous(),
    )


def _prefill_unit(v):
    dt = "bf16" if v.dtype == torch.bfloat16 else "fp16"
    return (
        f"prefill_q{v.num_q_heads}_kv{v.num_kv_heads}_d{v.head_size}_w{v.window}_{dt}"
    )


def _prefill_rows():
    return [v for rows in hip._prefill_table().values() for v in rows]


def test_prefill_configs_are_well_formed():
    """No two prefill builds share a shape and step, every decode configuration
    has prefill builds, and the build of the largest step reached runs."""
    if not _prefill_rows():
        pytest.skip("_rocm_C carries no RDNA3.5 variants")
    keys = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
        + (v.dtype, v.min_query_len)
        for v in _prefill_rows()
    }
    assert len(keys) == len(_prefill_rows()), "two builds share a step"
    configs = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.window)
        for v in hip.variants()
        if v.page_size == 16
    }
    for (hq, hkv, d, window), dtype in itertools.product(configs, DTYPES):
        steps = hip.prefill_variants(hq, hkv, d, 16, window, dtype)
        assert steps, (hq, hkv, d, window, dtype)
        starts = [v.min_query_len for v in steps]
        assert starts == sorted(set(starts))
        for v in steps:
            for m in (v.min_query_len, v.min_query_len + 1):
                assert hip.prefill_variant_for(hq, hkv, d, 16, 1, window, dtype, m) == v
    v = steps[0]
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, 16, 1, v.window, v.dtype)
    assert hip.prefill_variant_for(*args, hip.PREFILL_MIN_M - 1) is None
    assert hip.prefill_variant_for(*args, hip.PREFILL_MIN_M) == steps[0]
    assert hip.prefill_variant_for(*args, 1 << 20) == steps[-1]


def test_every_listed_prefill_variant_is_built_in():
    _skip_unless_gfx1151()
    # Raises if a build has no instantiation unit.
    assert torch.ops._rocm_C.rdna_singleseq_prefill_configs()
    missing = [v for v in _prefill_rows() if hip.load_prefill(v) is None]
    assert not missing, f"{len(missing)} prefill variants not in _rocm_C"


@pytest.mark.parametrize("unit", sorted({_prefill_unit(v) for v in _prefill_rows()}))
def test_built_in_prefill_variants_match_reference(unit):
    """Every built prefill variant matches the reference, with and without a
    prefix."""
    _skip_unless_gfx1151()
    for v in _prefill_rows():
        if _prefill_unit(v) != unit:
            continue
        module = hip.load_prefill(v)
        assert module is not None, v
        args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
        scratch = hip.make_prefill_scratch(
            hip.prefill_variants(*args, v.dtype), torch.device("cuda")
        )
        prefix = max(4096, 2 * v.window)
        cases = [(v.min_query_len, 0), (v.min_query_len + 37, prefix)]
        if v == hip.prefill_variants(*args, v.dtype)[0]:
            cases += [(hip.PREFILL_MIN_M, prefix), (hip.PREFILL_MIN_M + 50, 0)]
        for seed, (m, p) in enumerate(cases):
            seq_len = m + p
            q, kv, block_table = _prefill_inputs(v, m, seq_len, seed)
            out = torch.full_like(q, float("nan"))
            seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
            module.prefill_attn(
                q,
                kv,
                block_table,
                out,
                *scratch,
                seq_lens,
                seq_len,
                v.head_size**-0.5,
            )
            torch.accelerator.synchronize()
            _assert_prefill_close(
                out, q, kv, block_table, seq_len, v.window, f"{v}, M={m}, prefix={p}"
            )


def test_prefill_past_the_last_step():
    """M past the last step runs the last row."""
    _skip_unless_gfx1151()
    v = next(
        rows[-1]
        for rows in hip._prefill_table().values()
        if rows[-1].dtype == torch.float16 and rows[-1].head_size == 128
    )
    module = hip.load_prefill(v)
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
    scratch = hip.make_prefill_scratch(
        hip.prefill_variants(*args, v.dtype), torch.device("cuda")
    )
    for seed, (m, p) in enumerate([(8193, 0), (12000, 1000)]):
        seq_len = m + p
        q, kv, block_table = _prefill_inputs(v, m, seq_len, seed)
        out = torch.full_like(q, float("nan"))
        seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
        module.prefill_attn(
            q, kv, block_table, out, *scratch, seq_lens, seq_len, v.head_size**-0.5
        )
        torch.accelerator.synchronize()
        _assert_prefill_close(
            out, q, kv, block_table, seq_len, v.window, f"{v}, M={m}, prefix={p}"
        )


def test_prefill_window_never_reads_freed_pages():
    """Freed pages before the window do not reach the output."""
    _skip_unless_gfx1151()
    v = next(v for v in _prefill_rows() if v.window and v.dtype == torch.float16)
    module = hip.load_prefill(v)
    m, seq_len = v.min_query_len, v.min_query_len + 3 * v.window
    q, kv, block_table = _prefill_inputs(v, m, seq_len)
    clean = (kv.clone(), block_table.clone())
    first = (seq_len - m - (v.window - 1)) // v.page_size
    kv = torch.cat([kv, torch.full_like(kv[:1], float("nan"))])
    block_table = block_table.clone()
    block_table[:first] = kv.shape[0] - 1
    out = torch.empty_like(q)
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
    scratch = hip.make_prefill_scratch(hip.prefill_variants(*args, v.dtype), q.device)
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.prefill_attn(
        q, kv, block_table, out, *scratch, seq_lens, seq_len, v.head_size**-0.5
    )
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()
    _assert_prefill_close(out, q, *clean, seq_len, v.window, "freed pages")


@pytest.mark.parametrize(
    "shape",
    [(32, 16, 256, 1024, torch.float16), (32, 4, 512, 0, torch.float16)]
    + [(32, 16, 256, 1024, torch.bfloat16)],
    ids=["swa_fp16", "global_fp16", "swa_bf16"],
)
@pytest.mark.parametrize(
    "m, prefix, mm_ranges",
    [
        (600, 0, [(40, 295), (400, 469)]),
        (700, 3000, [(3100, 3355)]),
        (300, 500, [(450, 560)]),
        # Longer than the 1024-key window: the window bounds earlier keys.
        (2048, 512, [(600, 2099)]),
    ],
    ids=["two_images", "split_kv", "from_prefix", "longer_than_window"],
)
def test_prefill_mm_prefix_matches_reference(shape, m, prefix, mm_ranges):
    """Tokens of a multimodal range see the whole range (Gemma4's shapes)."""
    _skip_unless_gfx1151()
    hq, hkv, d, window, dtype = shape
    v = hip.prefill_variant_for(hq, hkv, d, 16, 1, window, dtype, m)
    assert v is not None, shape
    module = hip.load_prefill(v)
    assert module is not None, v
    scratch = hip.make_prefill_scratch(
        hip.prefill_variants(hq, hkv, d, 16, window, dtype), torch.device("cuda")
    )
    seq_len = m + prefix
    q, kv, block_table = _prefill_inputs(v, m, seq_len)
    out = torch.full_like(q, float("nan"))
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.prefill_attn(
        q,
        kv,
        block_table,
        out,
        *scratch,
        seq_lens,
        seq_len,
        d**-0.5,
        _mm_ranges_tensor(mm_ranges, q.device)[0],
    )
    torch.accelerator.synchronize()
    _assert_prefill_close(
        out, q, kv, block_table, seq_len, window, f"{v}, M={m}", mm_ranges
    )


@pytest.mark.parametrize("bad", ["batch", "triples", "cpu"])
def test_prefill_refuses_malformed_mm_ranges(bad):
    """Only one sequence's (ranges, 2) on the device reaches the kernel: the
    batch's (sequences, ranges, 2) would unmask other sequences' keys."""
    _skip_unless_gfx1151()
    v = hip.prefill_variant_for(32, 16, 256, 16, 1, 1024, torch.float16, 64)
    module = hip.load_prefill(v)
    scratch = hip.make_prefill_scratch([v], torch.device("cuda"))
    q, kv, block_table = _prefill_inputs(v, 64, 64)
    good = _mm_ranges_tensor([(10, 40)], q.device)
    ranges = {
        "batch": good,
        "triples": torch.zeros(2, 3, dtype=torch.int32, device=q.device),
        "cpu": good[0].cpu(),
    }[bad]
    seq_lens = torch.tensor([64], device=q.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="mm_ranges must be"):
        module.prefill_attn(
            q, kv, block_table, torch.empty_like(q), *scratch, seq_lens, 64,
            256**-0.5, ranges,
        )  # fmt: skip


@pytest.mark.parametrize(
    "ranges, seq_len, query_len, expected",
    [
        # Decode after the image: no query token in a range.
        ([(100, 355)], 500, 1, (False, -1)),
        # The image's last token recomputed alone: no later keys.
        ([(100, 355)], 356, 1, (False, 255)),
        # A short call inside a long image (e.g. after a prefix-cache hit).
        ([(100, 355)], 300, 5, (True, 199)),
        # A short image long past.
        ([(10, 14)], 500, 1, (False, -1)),
        # An empty range over the query tokens.
        ([(495, 495)], 500, 8, (False, -1)),
        # Text after a 2000-token image.
        ([(0, 1999)], 2200, 100, (False, -1)),
        # A prefill chunk ending inside the image.
        ([(100, 355)], 200, 150, (True, 99)),
    ],
    ids=[
        "after",
        "last_token",
        "short_call_inside",
        "short_past",
        "empty",
        "text_after_long",
        "prefill_into",
    ],  # fmt: skip
)
def test_mm_query_overlap(ranges, seq_len, query_len, expected):
    """Only ranges holding the step's query tokens count."""
    import types

    from vllm.v1.attention.backends.rocm_rdna_singleseq_attn import mm_query_overlap

    md = types.SimpleNamespace(
        mm_req_doc_ranges={0: ranges},
        seq_lens_cpu_upper_bound=torch.tensor([seq_len]),
        query_start_loc_cpu=torch.tensor([0, query_len]),
    )
    assert mm_query_overlap(md) == expected


@pytest.mark.parametrize(
    "future_keys, reach, clamp, reason",
    [
        (True, 4, True, "later keys"),
        (False, 1500, False, "past the window"),
        (False, 1500, True, None),
        (False, 1000, False, None),
    ],
    ids=["later_keys", "unclamped_reach", "clamped_reach", "within_window"],
)
def test_mm_ranges_decode_decision(future_keys, reach, clamp, reason):
    """Decode runs Triton when a query token's range has later keys, or when an
    unclamped range reaches past the window; otherwise the ranges do not
    stop it."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = _impl()
    impl._rejected = None
    impl.head_size, impl.num_heads, impl.num_kv_heads = HEAD_DIM, HQ, HKV
    fits = impl._prepare(
        kv_cache=torch.empty(0, dtype=torch.float16),
        q=torch.empty(1, HQ, HEAD_DIM, dtype=torch.float16),
        out=torch.empty(1, HQ, HEAD_DIM, dtype=torch.float16),
        alibi_slopes=None,
        sinks=None,
        softcap=0,
        causal=True,
        window_size=(1023, 0),
        kv_quant_mode=KVQuantMode.NONE,
        seqused_k=torch.zeros(1),
        max_seqlen_q=2,
        mm_future_keys=future_keys,
        mm_reach=reach,
        mm_prefix_clamp=clamp,
    )
    assert not fits
    if reason is None:
        # Refused later, on the tensors, not for the ranges.
        assert "multimodal" not in impl._rejected
    else:
        assert reason in impl._rejected
