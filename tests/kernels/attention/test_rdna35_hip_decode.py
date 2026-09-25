# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA3.5 HIP decode attention: agreement with a reference, and honest fallback.

The kernel walks the paged KV cache with its own address arithmetic rather than
reading the tensor's strides, so a layout it did not expect yields finite,
wrong numbers instead of an error. These tests therefore cover three things a
plain tolerance check would miss:

- a short context, where a causal off-by-one is visible. The error from one
  masked-off-by-one key falls off as ~1/S while the tolerance is fixed, so long
  contexts hide the bug most likely to be present.
- a deliberately mutated kernel, which the comparison *must* fail. A test that
  passes a known-broken kernel proves nothing about the working one. It also
  exercises two variants in one process, which only works because each build
  gets its own device-function symbols.
- that an unsupported shape falls back to Triton rather than being served
  wrong.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")

rdna35 = pytest.importorskip("vllm.v1.attention.ops.rdna35_hip_decode")

HQ, HKV, HEAD_DIM, M, BLOCK_SIZE = 32, 16, 256, 4, 16
# Partials big enough (256 KiB per group) for the segments to share the merge,
# on a grid small enough (32 workgroups) to be resident at once, which the
# host requires before it lets them wait for each other.
SHARED = dict(hq=16, hkv=2, hd=512, rg=2, nseg=8)
# Four row tiles per kv head, split over the waves that share each key tile
# (RSPL).  With nw=4 there is one tile per workgroup, so no merge round orders
# the waves' rows before they are published; nseg=16 makes the merge shared.
RSPL = {
    "one tile, shared merge": dict(hq=32, hkv=2, hd=128, rspl=4, nw=4, nseg=16),
    "two tiles": dict(hq=32, hkv=2, hd=128, rspl=4, nw=8, nseg=4),
    "with dspl": dict(hq=32, hkv=2, hd=128, rspl=2, nw=8, dspl=2, rg=2, nseg=4),
}
# Two decompositions in one build, switched on the device at S >= 512: row
# groups across workgroups below, rows split inside a four-wave workgroup and
# sixteen segments above, out of a block launched with eight waves.
DUAL = dict(
    hq=32,
    hkv=2,
    hd=128,
    rg=4,
    nw=8,
    nseg=2,
    sw=512,
    nseg2=16,
    rg2=1,
    minb2=1,
    rspl2=4,
    nw2=4,
)
# Relative error bound per dtype.  The reference sees the same rounded inputs,
# so what differs is the kernel's arithmetic and its output rounding -- the
# latter alone up to 2^-8 relative in bf16.
TOL = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
DTYPES = list(TOL)


def _skip_unless_gfx1151():
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        pytest.skip("kernel is built for gfx1151")


@pytest.fixture(scope="session", autouse=True)
def _build_variants():
    """Build the variants this module needs before any test runs.

    They are independent builds of ~19.6 s each, almost all of it torch's
    headers rather than the kernel, so serially they were the whole runtime of
    the suite on a cold cache. Built together it is one build's worth.

    Not a correctness concern: a build that fails here fails again in the test
    that needs it, where it is reported against that test rather than as a
    collection error.
    """
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        return
    rdna35.precompile(
        v
        for dtype in DTYPES
        for v in (
            _variant(layout=0, dtype=dtype),
            _variant(layout=1, dtype=dtype),
            _variant(layout=0, nseg=4, dtype=dtype),
            _variant(layout=1, nseg=4, dtype=dtype),
            _variant(layout=1, mutate=1, dtype=dtype),
            _variant(**SHARED, dtype=dtype),
            *(_variant(**shape, dtype=dtype) for shape in RSPL.values()),
            _variant(**DUAL, dtype=dtype),
        )
    )


def _variant(
    layout: int = 1,
    mutate: int = 0,
    nseg: int = 1,
    hq: int = HQ,
    hkv: int = HKV,
    hd: int = HEAD_DIM,
    rg: int = 1,
    dtype: torch.dtype = torch.float16,
    rspl: int = 1,
    nw: int = 8,
    dspl: int = 0,
    **mode_b,
):
    # nseg > 1 with minb = 1 splits even a short context over several
    # workgroups, which is the only way to reach the cross-workgroup merge.
    return rdna35.KernelVariant(
        head_size=hd,
        num_q_heads=hq,
        num_kv_heads=hkv,
        max_m=M,
        block_size=BLOCK_SIZE,
        layout=layout,
        nseg=nseg,
        rg=rg,
        minb=1,
        mutate=mutate,
        dtype=dtype,
        rspl=rspl,
        nw=nw,
        dspl=dspl,
        **mode_b,
    )


def _paged_inputs(
    seq_len: int,
    layout: int,
    dtype: torch.dtype,
    seed: int = 0,
    hq: int = HQ,
    hkv: int = HKV,
    hd: int = HEAD_DIM,
):
    """Build a KV cache whose physical order matches the layout, then present
    it in the logical (num_blocks, num_kv_heads, block_size, 2*hs) order the
    backend passes down."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    num_blocks = seq_len // BLOCK_SIZE
    if layout == 0:  # NHD
        kv = torch.randn(num_blocks, BLOCK_SIZE, hkv, 2 * hd, device=dev, dtype=dtype)
        kv = kv.transpose(1, 2)
    else:  # HND
        kv = torch.randn(num_blocks, hkv, BLOCK_SIZE, 2 * hd, device=dev, dtype=dtype)
    kv = kv * 0.5
    q = torch.randn(M, hq, hd, device=dev, dtype=dtype) * 0.5
    return q, kv, torch.arange(num_blocks, device=dev, dtype=torch.int32)


def _reference(q, kv, seq_len):
    hq, hkv, hd = q.shape[1], kv.shape[1], q.shape[2]
    flat = kv.transpose(1, 2).reshape(seq_len, hkv, 2 * hd)
    k, v = flat[..., :hd], flat[..., hd:]
    gqa = hq // hkv
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    vf = v.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * (hd**-0.5)
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - M + torch.arange(M, device=q.device)).view(M, 1)
    scores = scores.masked_fill((pos > lim).view(1, M, seq_len), float("-inf"))
    return torch.bmm(torch.softmax(scores, -1), vf).permute(1, 0, 2)


def _run(
    seq_len,
    layout=1,
    dtype=torch.float16,
    mutate=0,
    nseg=1,
    repeat=1,
    build_dtype=None,
    **shape,
):
    _skip_unless_gfx1151()
    dims = {k: v for k, v in shape.items() if k in ("hq", "hkv", "hd")}
    q, kv, block_table = _paged_inputs(seq_len, layout, dtype, **dims)
    variant = _variant(layout, mutate, nseg, dtype=build_dtype or dtype, **shape)
    module = rdna35.load(variant)
    acc, m, ln, arrivals = rdna35.make_scratch(variant, q.device)
    out = torch.empty_like(q)
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    for _ in range(repeat):
        out.zero_()
        module.decode_attn(
            q, kv, block_table, out, acc, m, ln, arrivals, seq_lens, q.shape[2] ** -0.5
        )
    torch.accelerator.synchronize()
    return out.float(), _reference(q, kv, seq_len)


def _max_rel(got, ref) -> float:
    """Relative error with a floor, which is what catches a causal off-by-one.

    max_abs alone does not: one key masked wrongly gives max_abs=1.2e-03 at S=2048
    and slips past a 2e-2 absolute threshold.
    """
    return ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_len", [48, 1024])
@pytest.mark.parametrize("layout", [0, 1])
@pytest.mark.parametrize("nseg", [1, 4])
def test_matches_reference(seq_len, layout, nseg, dtype):
    got, ref = _run(seq_len, layout=layout, nseg=nseg, dtype=dtype)
    assert torch.isfinite(got).all()
    assert _max_rel(got, ref) <= TOL[dtype]


@pytest.mark.parametrize("dtype", DTYPES)
def test_split_merge_survives_relaunch(dtype):
    """The arrival counters must be back at zero after every launch.

    The last workgroup to arrive resets them; if it did not, the second launch
    would elect no merger and leave the output unwritten.
    """
    got, ref = _run(1024, nseg=4, repeat=3, dtype=dtype)
    assert _max_rel(got, ref) <= TOL[dtype]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_len", [48, 1024])
def test_shared_split_merge(seq_len, dtype):
    """Segments that wait for each other and merge a slice each.

    Relaunched, because the wait is on a generation that must keep advancing:
    one stuck at the value a later launch reads first would release nobody.
    """
    shape = {k: v for k, v in SHARED.items() if k != "nseg"}
    got, ref = _run(seq_len, nseg=SHARED["nseg"], repeat=3, dtype=dtype, **shape)
    assert torch.isfinite(got).all()
    assert _max_rel(got, ref) <= TOL[dtype]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_len", [48, 1024])
@pytest.mark.parametrize("name", list(RSPL))
def test_row_tiles_split_over_waves(name, seq_len, dtype):
    """Waves that share a key tile through LDS and split its rows."""
    shape = dict(RSPL[name])
    nseg = shape.pop("nseg")
    got, ref = _run(seq_len, nseg=nseg, repeat=2, dtype=dtype, **shape)
    assert torch.isfinite(got).all()
    assert _max_rel(got, ref) <= TOL[dtype]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_len", [48, 1024])
def test_two_modes(seq_len, dtype):
    """Both decompositions of one build: S=48 runs the first, 1024 the second."""
    shape = dict(DUAL)
    nseg = shape.pop("nseg")
    got, ref = _run(seq_len, nseg=nseg, repeat=2, dtype=dtype, **shape)
    assert torch.isfinite(got).all()
    assert _max_rel(got, ref) <= TOL[dtype]


def test_graph_replay_follows_seq_lens():
    """S is read on the device, so a captured graph serves a longer sequence.

    A host int is frozen into the graph at capture: every replay would attend
    over the capture-time length, which is what full CUDA-graph decode does.
    """
    _skip_unless_gfx1151()
    long_s = 1024
    dims = {k: v for k, v in DUAL.items() if k in ("hq", "hkv", "hd")}
    q, kv, block_table = _paged_inputs(long_s, 1, torch.float16, **dims)
    # The two-mode build, so the replays also cross its switch at S=512.
    variant = _variant(layout=1, **DUAL)
    module = rdna35.load(variant)
    scratch = rdna35.make_scratch(variant, q.device)
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
        n = s // BLOCK_SIZE
        ref = _reference(q, kv[:n], s)
        assert _max_rel(out.float(), ref) <= TOL[torch.float16], f"S={s}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_other_dtype_is_refused_not_miscomputed(dtype):
    """A build must refuse the other 16-bit type, not return nonsense.

    The loads reinterpret the tensors as the element type the variant was
    built for, so fp16 read as bf16 or the reverse stays finite and is wrong
    by orders of magnitude.
    """
    other = next(d for d in DTYPES if d != dtype)
    with pytest.raises(RuntimeError, match="kernel built for"):
        _run(1024, dtype=dtype, build_dtype=other)


@pytest.mark.parametrize("dtype", DTYPES)
def test_negative_control_is_detected(dtype):
    """A kernel mutated to admit one key too many must fail the comparison."""
    got, ref = _run(48, mutate=1, dtype=dtype)
    assert _max_rel(got, ref) > TOL[dtype], (
        "the mutated kernel passed, so this comparison cannot detect a causal "
        "off-by-one and proves nothing about the real kernel"
    )


def test_layouts_disagree_when_data_differs():
    """NHD and HND must be distinct code paths, not the same one twice."""
    assert rdna35.expected_kv_cache_strides(
        _variant(layout=0)
    ) != rdna35.expected_kv_cache_strides(_variant(layout=1))


def test_unsupported_head_size_falls_back_to_triton():
    """An unbuilt shape must be served by Triton, not served wrong."""
    from vllm.v1.attention.backends.rdna35_hip_attn import Rdna35HipAttentionImpl
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = Rdna35HipAttentionImpl.__new__(Rdna35HipAttentionImpl)
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
    )
    assert not fits
    assert "head_size 96" in impl._rejected
