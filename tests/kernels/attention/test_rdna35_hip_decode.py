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


def _skip_unless_gfx1151():
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        pytest.skip("kernel is built for gfx1151")


def _variant(layout: int = 1, mutate: int = 0):
    return rdna35.KernelVariant(
        head_size=HEAD_DIM,
        num_q_heads=HQ,
        num_kv_heads=HKV,
        max_m=M,
        block_size=BLOCK_SIZE,
        layout=layout,
        mutate=mutate,
    )


def _paged_inputs(seq_len: int, layout: int, dtype: torch.dtype, seed: int = 0):
    """Build a KV cache whose physical order matches the layout, then present
    it in the logical (num_blocks, num_kv_heads, block_size, 2*hs) order the
    backend passes down."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    num_blocks = seq_len // BLOCK_SIZE
    if layout == 0:  # NHD
        kv = torch.randn(
            num_blocks, BLOCK_SIZE, HKV, 2 * HEAD_DIM, device=dev, dtype=dtype
        )
        kv = kv.transpose(1, 2)
    else:  # HND
        kv = torch.randn(
            num_blocks, HKV, BLOCK_SIZE, 2 * HEAD_DIM, device=dev, dtype=dtype
        )
    kv = kv * 0.5
    q = torch.randn(M, HQ, HEAD_DIM, device=dev, dtype=dtype) * 0.5
    return q, kv, torch.arange(num_blocks, device=dev, dtype=torch.int32)


def _reference(q, kv, seq_len):
    flat = kv.transpose(1, 2).reshape(seq_len, HKV, 2 * HEAD_DIM)
    k, v = flat[..., :HEAD_DIM], flat[..., HEAD_DIM:]
    gqa = HQ // HKV
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    vf = v.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * (HEAD_DIM**-0.5)
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - M + torch.arange(M, device=q.device)).view(M, 1)
    scores = scores.masked_fill((pos > lim).view(1, M, seq_len), float("-inf"))
    return torch.bmm(torch.softmax(scores, -1), vf).permute(1, 0, 2)


def _run(seq_len, layout=1, dtype=torch.float16, mutate=0):
    _skip_unless_gfx1151()
    q, kv, block_table = _paged_inputs(seq_len, layout, dtype)
    variant = _variant(layout, mutate)
    module = rdna35.load(variant)
    acc, m, ln, arrivals = rdna35.make_scratch(variant, q.device)
    out = torch.empty_like(q)
    module.decode_attn(
        q, kv, block_table, out, acc, m, ln, arrivals, seq_len, HEAD_DIM**-0.5
    )
    torch.accelerator.synchronize()
    return out.float(), _reference(q, kv, seq_len)


def _max_rel(got, ref) -> float:
    """Relative error with a floor, which is what catches a causal off-by-one.

    max_abs alone does not: one key masked wrongly gives max_abs=1.2e-03 at S=2048
    and slips past a 2e-2 absolute threshold.
    """
    return ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()


@pytest.mark.parametrize("seq_len", [48, 1024])
@pytest.mark.parametrize("layout", [0, 1])
def test_matches_reference(seq_len, layout):
    got, ref = _run(seq_len, layout=layout)
    assert torch.isfinite(got).all()
    assert _max_rel(got, ref) <= 1e-3


def test_bfloat16_is_refused_not_miscomputed():
    """bf16 must raise, not return nonsense.

    The inner product is __builtin_amdgcn_fdot2 and the loads reinterpret the
    cache as _Float16, so bf16 input reads the same bits as fp16: the output
    stays finite and is wrong by three orders of magnitude.
    """
    with pytest.raises(RuntimeError, match="fp16 only"):
        _run(1024, dtype=torch.bfloat16)


def test_negative_control_is_detected():
    """A kernel mutated to admit one key too many must fail the comparison."""
    got, ref = _run(48, mutate=1)
    assert _max_rel(got, ref) > 1e-3, (
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
    impl.head_size, impl.num_heads, impl.num_kv_heads = 128, HQ, HKV

    fits = impl._prepare(
        kv_cache=torch.empty(0),
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
    assert "head_size 128" in impl._rejected
