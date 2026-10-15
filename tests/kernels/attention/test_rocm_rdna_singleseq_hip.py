# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA3.5 decode and prefill attention: accuracy against a reference, and
fallback to Triton for unsupported calls."""

import functools
import itertools

import numpy as np
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


def _built(dtype=torch.float16, batch=0, **want):
    """The first built-in variant with these properties, loaded.

    Args:
        dtype: Element type.
        batch: 1 for a batched build.
        **want: Predicates on KernelVariant fields, by name.

    Returns:
        (variant, module).

    """
    _skip_unless_gfx1151()
    for v in hip.variants():
        if (
            v.dtype == dtype
            and v.batched == batch
            and all(pred(getattr(v, k)) for k, pred in want.items())
        ):
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


def _check_batch(v, module, lens):
    """Check a batched build over these lengths, twice on one scratch; a
    padded sequence (S=0) stays unwritten."""
    width = max(lens) // v.page_size + 1
    parts = [_paged_inputs(max(s, 16), v, seed=i) for i, s in enumerate(lens)]
    kv = torch.cat([p[1] for p in parts])
    first = torch.tensor([0] + [p[1].shape[0] for p in parts]).cumsum(0)
    block_table = torch.zeros(len(lens), width, device=kv.device, dtype=torch.int32)
    for i, p in enumerate(parts):
        block_table[i, : p[1].shape[0]] = p[2] + int(first[i])
    q = torch.cat([p[0] for p in parts])
    scratch = hip.make_scratch(v, q.device, max_seqs=len(lens))
    seq_lens = torch.tensor(lens, device=q.device, dtype=torch.int32)
    out = torch.full_like(q, 7.0)
    for _ in range(2):
        module.decode_attn(
            q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5
        )
    torch.accelerator.synchronize()
    for i, s in enumerate(lens):
        rows = out[i * v.max_query_len : (i + 1) * v.max_query_len].float()
        if s == 0:
            assert (rows == 7.0).all(), f"{v.name}: a padded sequence was written"
            continue
        ref = _reference(parts[i][0], parts[i][1], s, v.window)
        assert _max_rel(rows, ref) <= TOL[v.dtype], f"{v.name}: sequence {i}, S={s}"


def _unit(v):
    """A test case: the variants of one shape and dtype."""
    dt = "bf16" if v.dtype == torch.bfloat16 else "fp16"
    return f"q{v.num_q_heads}_kv{v.num_kv_heads}_d{v.head_size}_w{v.window}_{dt}"


def test_decode_configs_are_well_formed():
    """No two decode builds share a shape, and every built shape covers every
    M in both dtypes, for one sequence and a batch."""
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
        for m, dtype, batch in itertools.product(
            range(1, hip.MAX_M + 1), DTYPES, (False, True)
        ):
            v = hip.variant_for(hq, hkv, d, m, bs, 1, window, dtype, batch)
            assert v is not None, (hq, hkv, d, window, bs, m, dtype, batch)
            assert not (batch and v.dot_product), v.name


def test_every_listed_variant_is_built_in():
    _skip_unless_gfx1151()
    # Raises if a build has no instantiation unit.
    assert torch.ops._rocm_C.rdna_singleseq_decode_configs()
    assert torch.ops._rocm_C.rdna_singleseq_decode_batch_configs()
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
        (3, 4, 1, "unequal"),
        (1, 64, 64, "more than"),
    ],
)
def test_non_decode_batches_fall_back_to_triton(nseq, tokens, max_q, reason):
    """Only sequences of one query length up to MAX_M reach the decode
    kernel."""
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


@pytest.mark.parametrize("gfx1151", [True, False])
@pytest.mark.parametrize("max_num_seqs", [1, 2])
def test_singleseq_first_only_on_gfx1151(gfx1151, max_num_seqs, monkeypatch):
    """First on gfx1151 whatever max_num_seqs; elsewhere not a candidate."""
    import types

    import vllm.config
    import vllm.platforms.rocm as rocm
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as B

    config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs)
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(rocm, "on_gfx1151", lambda: gfx1151)
    order = rocm._get_backend_priorities(use_mla=False, use_sparse=False)
    if gfx1151:
        assert order[0] == B.ROCM_RDNA_SINGLESEQ_ATTN
    else:
        assert B.ROCM_RDNA_SINGLESEQ_ATTN not in order


def test_singleseq_valid_for_any_batch():
    """Valid whatever max_num_seqs, on Triton's packed HND cache."""
    from vllm.v1.attention.backends import rocm_rdna_singleseq_attn as rdna
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout

    backend = rdna.RocmRdnaSingleSeqAttentionBackend
    reason = backend.supports_combination(
        HEAD_DIM, torch.float16, "auto", 16, False, False, False, False, None
    )
    assert reason is None
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=HKV, head_size=HEAD_DIM, dtype=torch.float16
    )
    assert backend.customize_spec(spec) == TritonAttentionBackend.customize_spec(spec)
    assert backend.supported_kv_cache_layouts() == (KVCacheLayout.LBHNC,)
    assert 512 in backend.get_supported_head_sizes()


# ---------------------------------------------------------------- prefill


def _prefill_reference(q, kv, block_table, seq_len, window=0):
    """Float reference of a prefill."""
    m, hq, hkv, hd = q.shape[0], q.shape[1], kv.shape[1], q.shape[2]
    pages = kv[block_table.long()]
    flat = pages.transpose(1, 2).reshape(-1, hkv, 2 * hd)[:seq_len].float()
    gqa = hq // hkv
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - m + torch.arange(m, device=q.device)).view(m, 1)
    masked = pos > lim
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


def _triton_prefill(q, kv, block_table, seq_len, window=0):
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
    )
    return out


def _assert_prefill_close(got, q, kv, block_table, seq_len, window, what):
    """Within twice Triton's error against the float reference."""
    ref = _prefill_reference(q, kv, block_table, seq_len, window)
    err = (got.float().nan_to_num(1e9) - ref).abs().max().item()
    triton = _triton_prefill(q, kv, block_table, seq_len, window)
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


# ---------------------------------------------------------------- prefill batches

# (M, S) per sequence.
_BATCHES = {
    "decode+chunk+prompt": [(1, 6000), (300, 2348), (40, 40)],
    "8 prompts": [(256, 256)] * 8,
    "unequal decodes": [(1, 37), (3, 1000)],
    "padded": [(64, 1064), (0, 0), (17, 17)],
}


def _steps(v, m, s, window):
    """Key steps of each tile of a sequence, as the kernel computes them."""
    g = v.num_q_heads // v.num_kv_heads
    context, out = s - m, []
    for t in range(-(-m * g // v.block_rows)):
        first, last = (
            t * v.block_rows // g,
            min(m - 1, ((t + 1) * v.block_rows - 1) // g),
        )
        begin = 0
        if window:
            begin = max(0, context + first - window + 1) // v.key_tile * v.key_tile
        out.append(-(-(context + last + 1 - begin) // v.key_tile))
    return out


@pytest.mark.parametrize("window", [0, 512])
@pytest.mark.parametrize("batch", [*_BATCHES, "at the bound"])
def test_prefill_plan_covers_every_tile_once(batch, window):
    """The work list covers every tile once, heaviest first, within the
    scratch's bounds."""
    target = 160
    v = hip.prefill_variants(32, 8, 128, 16, 0, torch.float16)[2]
    # The most counters and partials the bounds allow.
    lens = (
        [(1, 32768)] * 19 + [(16, 16)] if batch == "at the bound" else _BATCHES[batch]
    )
    starts = np.cumsum([0] + [m for m, _ in lens])
    plan, head_group = hip.plan_prefill(
        starts, np.array([s for _, s in lens]), v, window, target
    )
    seq, tile = plan[:, 0], plan[:, 1] & 0xFFFF
    seg, nseg, slot = plan[:, 2] & 0xFFFF, plan[:, 2] >> 16, plan[:, 3]
    steps = {
        (i, t): n
        for i, (m, s) in enumerate(lens)
        for t, n in enumerate(_steps(v, m, s, window))
    }
    entries = sorted(zip(seq.tolist(), tile.tolist(), seg.tolist(), nseg.tolist()))
    want = sorted(
        (i, t, k, n)
        for (i, t) in steps
        for n in {int(nseg[(seq == i) & (tile == t)][0])}
        for k in range(n)
    )
    assert entries == want
    assert set(steps) == set(zip(seq.tolist(), tile.tolist()))
    split = nseg > 1
    assert all(
        steps[(i, t)] // n >= 16
        for i, t, n in zip(seq[split], tile[split], nseg[split])
    )
    weight = np.array([steps[(i, t)] / n for i, t, n in zip(seq, tile, nseg)])
    assert (np.diff(weight) <= 0).all()
    split_tiles = {(i, t) for i, t in zip(seq[split], tile[split])}
    assert len(split_tiles) * v.num_kv_heads < target
    assert int(nseg[split & (seg == 0)].sum()) * v.num_kv_heads < 2 * target
    assert ((slot >= 0) == split).all()
    assert head_group == (v.num_kv_heads if split_tiles else 0)
    if batch == "at the bound" and not window:
        assert len(split_tiles) == 19


def _prefill_batch(v, lens, seed=0):
    """One cache, each sequence its own shuffled pages; q packed."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    pages = [-(-s // v.page_size) for _, s in lens]
    total = sum(pages) + 2
    kv = torch.randn(total, v.num_kv_heads, v.page_size, 2 * v.head_size, device=dev)
    order = torch.randperm(total, device=dev).to(torch.int32)
    block_table = torch.zeros(len(lens), max(pages), dtype=torch.int32, device=dev)
    for i, (first, n) in enumerate(zip(np.cumsum([0] + pages), pages)):
        block_table[i, :n] = order[first : first + n]
    starts = torch.tensor(np.cumsum([0] + [m for m, _ in lens]), dtype=torch.int32)
    q = torch.randn(int(starts[-1]), v.num_q_heads, v.head_size, device=dev)
    seq_lens = torch.tensor([s for _, s in lens], dtype=torch.int32)
    return (q * 0.5).to(v.dtype), (kv * 0.5).to(v.dtype), block_table, starts, seq_lens


def _serve_batch(v, q, kv, block_table, starts, seq_lens, out, decodes=0):
    """The backend's prefill of a batch; leading decodes on the decode
    kernel."""
    import types

    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = _impl()
    impl._rejected = None
    impl.num_heads, impl.num_kv_heads = v.num_q_heads, v.num_kv_heads
    impl.head_size = v.head_size
    impl._variant, impl._built, impl._cap, impl._max_seqs = None, None, 0, 64
    md = types.SimpleNamespace(
        query_start_loc_cpu=starts,
        seq_lens_cpu=seq_lens,
        query_start_loc=starts.to(q.device),
        max_seq_len=int(seq_lens.max()),
        num_decodes=decodes,
        num_decode_tokens=decodes,
        plans={},
    )
    serve = impl._serve_mixed if decodes else impl._serve_prefill
    served = serve(
        kv,
        md,
        {
            "q": q,
            "out": out,
            "seqused_k": seq_lens.to(q.device),
            "block_table": block_table,
            "softmax_scale": v.head_size**-0.5,
            "causal": True,
            "window_size": (v.window - 1, 0) if v.window else None,
            "alibi_slopes": None,
            "sinks": None,
            "softcap": 0,
            "kv_quant_mode": KVQuantMode.NONE,
            "max_seqlen_q": int((starts[1:] - starts[:-1]).max()),
        },
    )
    assert served, impl._rejected
    return md


@pytest.mark.parametrize(
    "unit", sorted({_prefill_unit(v) for v in _prefill_rows() if v.page_size == 16})
)
def test_prefill_batch_matches_reference(unit):
    """A batch in one launch, each sequence against its reference."""
    _skip_unless_gfx1151()
    v = next(v for v in _prefill_rows() if _prefill_unit(v) == unit)
    for seed, (name, lens) in enumerate(_BATCHES.items()):
        lens = [(m, s + v.window if m != s else s) for m, s in lens]
        q, kv, block_table, starts, seq_lens = _prefill_batch(v, lens, seed)
        out = torch.full_like(q, float("nan"))
        _serve_batch(v, q, kv, block_table, starts, seq_lens, out)
        torch.accelerator.synchronize()
        for i, (m, s) in enumerate(lens):
            if not m:
                continue
            rows = slice(int(starts[i]), int(starts[i + 1]))
            _assert_prefill_close(
                out[rows],
                q[rows],
                kv,
                block_table[i, : -(-s // v.page_size)],
                s,
                v.window,
                f"{unit} {name}: sequence {i} (M={m}, S={s})",
            )


@pytest.mark.parametrize(
    "unit",
    sorted(
        {
            _prefill_unit(v)
            for v in _prefill_rows()
            if v.page_size == 16 and v.head_size in hip.SUPPORTED_HEAD_SIZES
        }
    ),
)
def test_mixed_batch_matches_reference(unit):
    """Decodes then prefills, including a short one, each against its
    reference."""
    _skip_unless_gfx1151()
    v = next(v for v in _prefill_rows() if _prefill_unit(v) == unit)
    batches = [
        [(1, 3000 + v.window), (1, 77), (1, 1024), (130, 2000 + v.window), (40, 40)],
        [(1, 3000 + v.window), (1, 77), (1, 1024), (5, 2000 + v.window)],
    ]
    for lens in batches:
        q, kv, block_table, starts, seq_lens = _prefill_batch(v, lens)
        out = torch.full_like(q, float("nan"))
        _serve_batch(v, q, kv, block_table, starts, seq_lens, out, decodes=3)
        torch.accelerator.synchronize()
        for i, (m, s) in enumerate(lens):
            rows = slice(int(starts[i]), int(starts[i + 1]))
            _assert_prefill_close(
                out[rows],
                q[rows],
                kv,
                block_table[i, : -(-s // v.page_size)],
                s,
                v.window,
                f"{unit}: sequence {i} (M={m}, S={s})",
            )


def test_prefill_batch_configs_are_well_formed():
    """Every cell names a prefill build of its shape and a head group that
    divides its kv heads; each shape has the first cell."""
    table = hip._prefill_batch_table()
    if not table:
        pytest.skip("_rocm_C carries no RDNA3.5 variants")
    seen = set()
    for (hq, hkv, d, page, window), cells in table.items():
        for mean_m, keys, min_query_len, head_group in cells:
            key = (hq, hkv, d, page, window, mean_m, keys)
            assert key not in seen, key
            seen.add(key)
            assert mean_m in hip.PREFILL_BATCH_MEAN_M
            assert keys in hip.PREFILL_BATCH_KEYS
            assert head_group == 0 or hkv % head_group == 0
            for dtype in DTYPES:
                steps = hip.prefill_variants(hq, hkv, d, page, window, dtype)
                assert min_query_len in {v.min_query_len for v in steps}, key
    for config in {k[:5] for k in seen}:
        assert (*config, hip.PREFILL_BATCH_MEAN_M[0], 0) in seen, config


def test_prefill_batch_cell_lookup(monkeypatch):
    """A batch takes the cell of the largest steps at or below its features."""
    table = {
        (32, 8, 128, 16, 0): [(1, 0, 16, 8), (1, 2048, 128, 1), (256, 512, 512, 4)]
    }
    monkeypatch.setattr(hip, "_prefill_batch_table", lambda: table)
    cell = functools.partial(hip.prefill_batch_cell, 32, 8, 128, 16, 0)
    assert cell(4, 100) == (16, 8)
    assert cell(4, 5000) == (128, 1)
    assert cell(300, 5000) == (512, 4)
    assert cell(300, 100) == (512, 4)  # no keys step at or below: the smallest
    assert hip.prefill_batch_cell(16, 2, 64, 16, 0, 4, 100) is None


def test_prefill_batch_features():
    """Padding left out; keys capped at the window."""
    starts = np.array([0, 4, 4, 12])
    seq_lens = np.array([100, 0, 8])
    mean_m, keys = hip.prefill_batch_features(starts, seq_lens, 0)
    assert mean_m == 6
    # Sequence 0: context 96, keys 96 + 2.5; sequence 1: context 0, keys 4.5.
    assert keys == pytest.approx((4 * 98.5 + 8 * 4.5) / 12)
    assert hip.prefill_batch_features(starts, seq_lens, 10)[1] == pytest.approx(
        (4 * 10 + 8 * 4.5) / 12
    )


def test_prefill_batch_needs_this_steps_host_copy():
    """A batch without this step's host query starts goes to Triton."""
    import types

    from vllm.v1.kv_cache_interface import KVQuantMode

    v = hip.prefill_variants(32, 8, 128, 16, 0, torch.float16)[0]
    impl = _impl()
    impl._rejected = None
    impl.num_heads, impl.num_kv_heads = v.num_q_heads, v.num_kv_heads
    impl.head_size = v.head_size
    kv = torch.empty(4, v.num_kv_heads, 16, 2 * v.head_size, dtype=v.dtype)
    for starts in (None, torch.tensor([0, 5, 9], dtype=torch.int32)):
        md = types.SimpleNamespace(query_start_loc_cpu=starts)
        served = impl._serve_prefill(
            kv,
            md,
            {
                "q": torch.empty(12, v.num_q_heads, v.head_size, dtype=v.dtype),
                "out": torch.empty(12, v.num_q_heads, v.head_size, dtype=v.dtype),
                "seqused_k": torch.zeros(2),
                "causal": True,
                "window_size": None,
                "alibi_slopes": None,
                "sinks": None,
                "softcap": 0,
                "kv_quant_mode": KVQuantMode.NONE,
            },
        )
        assert not served
        assert "host's query starts" in impl._rejected
