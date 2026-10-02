# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused decode Q/K norm + RoPE + K/V cache write of ROCM_ATTN on gfx1151.

Each case is a model family's attention as vLLM runs it: the norm modules and
rotary class the model builds, and the Triton writer ROCM_ATTN uses on gfx1151
today, on the packed HND cache.  RoPE is referenced in fp32 from the class's
own cos/sin cache with one rounding, as the kernel computes it; a separate
test ties that reference to each class's forward.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")

rc = pytest.importorskip("vllm.v1.attention.ops.rdna35_rope_cache")

from tests.kernels.utils import opcheck  # noqa: E402
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm  # noqa: E402
from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: E402
from vllm.model_executor.models.mistral import MistralAttention  # noqa: E402
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    triton_reshape_and_cache_flash,
)

BLOCK_SIZE = 16
EPS = 1e-6
# The reference rounds to the activation dtype after each RoPE multiply and
# add (forward_native), the kernel once at the end: up to 2 ulps apart on
# a few heads (at T = 32, in 5 of 40 seeds for Gemma 4).
MAX_ULPS = 2


@pytest.fixture(autouse=True)
def _skip_unless_built(default_vllm_config):
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151() or not rc.rdna35_rope_cache_available():
        pytest.skip("op is built for gfx1151")


@dataclass(frozen=True)
class Case:
    """One attention layer's pre-attention work, by model family."""

    hq: int
    hkv: int
    d: int
    rope: dict[str, Any] | None = field(default_factory=lambda: {"rope_theta": 1e6})
    norm: str = "none"  # none | rms | gemma
    v_norm: bool = False
    gate: bool = False  # Qwen3.5: [q | gate] per head
    mrope: bool = False
    long_rope: bool | None = None  # Phi longrope, use_long_rope
    q_scale: tuple[float, int] | None = None  # Ministral llama_4_scaling
    kv_shared: bool = False  # Gemma 4: Q only, no cache write
    max_pos: int = 4096


_LONGROPE = {
    "rope_type": "longrope",
    "rope_theta": 1e4,
    "original_max_position_embeddings": 4096,
}

CASES = {
    "llama31": Case(32, 8, 128, {"rope_theta": 5e5}),
    "qwen25_0.5b": Case(14, 2, 64),
    "llama2_mha": Case(32, 32, 128, {"rope_theta": 1e4}),
    "llama32_llama3": Case(
        24,
        8,
        128,
        {
            "rope_type": "llama3",
            "rope_theta": 5e5,
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    ),
    "ministral3_yarn_qscale": Case(
        32,
        8,
        128,
        {
            "rope_type": "yarn",
            "rope_theta": 1e6,
            "factor": 16.0,
            "original_max_position_embeddings": 16384,
        },
        q_scale=(0.1, 16384),
        max_pos=262144,
    ),
    "granite_nope": Case(32, 8, 128, None),
    "phi35v_longrope_short": Case(
        32, 32, 96, {**_LONGROPE, "partial_rotary_factor": 1.0}, long_rope=False
    ),
    "phi4_longrope_long": Case(
        24,
        8,
        128,
        {**_LONGROPE, "partial_rotary_factor": 0.75},
        long_rope=True,
        max_pos=8192,
    ),
    "nemotron_rot78": Case(
        40, 8, 128, {"rope_theta": 1e4, "partial_rotary_factor": 78 / 128}
    ),
    "qwen3": Case(32, 8, 128, norm="rms"),
    "qwen3vl_imrope": Case(
        32,
        8,
        128,
        {"rope_theta": 5e6, "mrope_section": [24, 20, 20], "mrope_interleaved": True},
        norm="rms",
        mrope=True,
    ),
    "qwen35_gate_imrope": Case(
        8,
        2,
        256,
        {
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        norm="gemma",
        gate=True,
        mrope=True,
    ),
    "gemma3_linear": Case(
        16,
        8,
        256,
        {"rope_type": "linear", "rope_theta": 1e6, "factor": 8.0},
        norm="gemma",
    ),  # fmt: skip
    "paligemma2": Case(8, 4, 256, {"rope_theta": 1e4}),
    "gemma4_sliding": Case(8, 1, 256, {"rope_theta": 1e4}, norm="rms", v_norm=True),
    "gemma4_full_proportional": Case(
        16,
        2,
        512,
        {"rope_type": "proportional", "rope_theta": 1e6, "partial_rotary_factor": 0.25},
        norm="rms",
        v_norm=True,
    ),
    "gemma4_kv_shared": Case(
        8, 1, 512, {"rope_theta": 1e6}, norm="rms", kv_shared=True
    ),
}


def _rope(case: Case, dtype: torch.dtype, monkeypatch):
    if case.rope is None:
        return None
    if case.long_rope is not None:
        from vllm.model_executor.layers.rotary_embedding import (
            phi3_long_rope_scaled_rope as phi3,
        )

        max_len = case.max_pos if case.long_rope else 4096
        monkeypatch.setattr(
            phi3,
            "get_current_vllm_config",
            lambda: SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=max_len)
            ),
        )
        params = dict(case.rope)
        half = int(case.d * params["partial_rotary_factor"]) // 2
        params["short_factor"] = [1.0 + 0.01 * i for i in range(half)]
        params["long_factor"] = [2.0 + 0.05 * i for i in range(half)]
        return get_rope(case.d, case.max_pos, True, params, dtype).cuda()
    return get_rope(case.d, case.max_pos, True, case.rope, dtype).cuda()


def _cache_and_offset(rope, case: Case):
    """The cos/sin rows the kernel indexes and the offset into them."""
    if rope is None:
        return None, 0
    if case.long_rope is not None:
        offset = rope.original_max_position_embeddings if rope.use_long_rope else 0
        return rope.long_short_cos_sin_cache, offset
    return rope.cos_sin_cache, 0


def _rope_ref(x, cs, pos):
    """NeoX RoPE in fp32 from the rounded input, rounded once."""
    if cs is None:
        return x
    rh = cs.shape[1] // 2
    c, s = cs[pos].float().chunk(2, dim=-1)
    c, s = c[:, None], s[:, None]
    xf = x.float()
    x1, x2, rest = xf[..., :rh], xf[..., rh : 2 * rh], xf[..., 2 * rh :]
    out = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, rest), dim=-1)
    return out.to(x.dtype)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest difference in units in the last place of the head's largest
    value.  Per head rather than per element: RoPE subtracts products, so one
    ulp of difference in a normalised input (rsqrt, summation order) is many
    ulps of an output that cancels."""
    bits = {torch.float16: 10, torch.bfloat16: 7}[a.dtype]
    af, bf = a.float(), b.float()
    big = torch.maximum(af.abs(), bf.abs()).amax(dim=-1, keepdim=True)
    big = big.clamp_min(2.0**-14)
    _, exp = torch.frexp(big)
    ulp = torch.ldexp(torch.ones_like(big), exp - 1 - bits)
    return ((af - bf).abs() / ulp).max().item() if a.numel() else 0.0


def _inputs(case, dtype, nseq, m, block_size=BLOCK_SIZE, pad_stride=0):
    t = nseq * m
    qd = 2 * case.d if case.gate else case.d
    qkv = torch.randn(
        t, case.hq * qd + 2 * case.hkv * case.d, dtype=dtype, device="cuda"
    )
    q = qkv[:, : case.hq * qd].view(t, case.hq, qd)[..., : case.d]
    k, v = (
        x.view(t, case.hkv, case.d)
        for x in qkv[:, case.hq * qd :].split(case.hkv * case.d, dim=-1)
    )
    # Each sequence decodes its last M tokens of a context somewhere below
    # max_pos; long contexts reach Ministral's scaling and Phi's long cache.
    hi = min(case.max_pos, 3 * 16384) - m
    ctx = torch.randint(m, hi, (nseq,))
    pos = (ctx[:, None] - m + torch.arange(m)).flatten().cuda()
    page = case.hkv * block_size * 2 * case.d
    blocks = max(4 * t, 64)
    flat = torch.zeros(blocks * (page + pad_stride), dtype=dtype, device="cuda")
    kv_cache = flat.as_strided(
        (blocks, case.hkv, block_size, 2 * case.d),
        (page + pad_stride, block_size * 2 * case.d, 2 * case.d, 1),
    )
    slots = torch.randperm(blocks * block_size, device="cuda")[:t]
    return qkv, q, k, v, pos, kv_cache, slots


def _norms(case, dtype):
    if case.norm == "none":
        return None, None, None, None
    if case.norm == "rms":
        qn, kn = (RMSNorm(case.d, EPS, dtype=dtype).cuda() for _ in range(2))
    else:
        qn, kn = (GemmaRMSNorm(case.d, EPS).cuda() for _ in range(2))
    for n in (qn, kn):
        n.weight.data.uniform_(-0.5, 0.5)
        if case.norm == "rms":
            n.weight.data += 1.0
    if case.norm == "rms":
        return qn, kn, qn.weight.data, kn.weight.data
    return qn, kn, qn.weight.data.float() + 1.0, kn.weight.data.float() + 1.0


def _reference(case, rope, q, k, v, pos, kv_cache, slots, qn, kn):
    cs, off = _cache_and_offset(rope, case)
    idx = pos + off
    q_ref = qn.forward_native(q.contiguous()) if qn is not None else q.clone()
    q_ref = _rope_ref(q_ref, cs, idx)
    if case.q_scale is not None:
        beta, orig = case.q_scale
        attn = SimpleNamespace(
            llama_4_scaling_beta=beta,
            llama_4_scaling_original_max_position_embeddings=orig,
        )
        scale = MistralAttention._get_llama_4_attn_scale(attn, pos)
        q_ref = (q_ref * scale.unsqueeze(-1)).to(q.dtype)
    if case.kv_shared:
        return q_ref, None, kv_cache.clone()
    k_ref = kn.forward_native(k.contiguous()) if kn is not None else k.clone()
    k_ref = _rope_ref(k_ref, cs, idx)
    v_ref = v.contiguous()
    if case.v_norm:
        v_ref = RMSNorm(case.d, EPS, has_weight=False).forward_native(v_ref)
    cache_ref = kv_cache.clone()
    key_cache, value_cache = cache_ref.transpose(1, 2).split(case.d, dim=-1)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    triton_reshape_and_cache_flash(
        k_ref, v_ref, key_cache, value_cache, slots, "auto", one, one
    )
    return q_ref, k_ref, cache_ref


def _run(case, rope, q, k, v, pos, kv_cache, slots, qw, kw, *, pos_shift=0,
         q_out=None, k_out=None, v_out=None):  # fmt: skip
    cs, off = _cache_and_offset(rope, case)
    positions = pos + pos_shift
    if case.mrope:
        positions = positions.expand(3, -1).contiguous()
    beta, orig = case.q_scale or (0.0, 0)
    rc.rdna35_rope_cache(
        positions,
        q,
        None if case.kv_shared else k,
        None if case.kv_shared else v,
        None if case.kv_shared else kv_cache,
        None if case.kv_shared else slots,
        cs,
        q_weight=qw,
        k_weight=None if case.kv_shared else kw,
        v_norm=case.v_norm,
        eps=EPS,
        pos_offset=off,
        q_scale_beta=beta,
        q_scale_orig_max=orig,
        q_out=q_out,
        k_out=k_out,
        v_out=v_out,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", list(CASES))
# 32 x 8 = 256 tokens: the fusion passes' range (rope_kvcache_fusion_max_token_num)
# also sends short prefills and mixed batches here.
@pytest.mark.parametrize("nseq,m", [(1, 1), (4, 8), (32, 8)])
def test_matches_reference(name, dtype, nseq, m, monkeypatch):
    case = CASES[name]
    torch.manual_seed(0)
    rope = _rope(case, dtype, monkeypatch)
    qn, kn, qw, kw = _norms(case, dtype)
    qkv, q, k, v, pos, kv_cache, slots = _inputs(case, dtype, nseq, m)
    q_ref, k_ref, cache_ref = _reference(
        case, rope, q, k, v, pos, kv_cache, slots, qn, kn
    )

    def gate():
        heads = qkv[:, : case.hq * 2 * case.d].view(-1, case.hq, 2 * case.d)
        return heads[..., case.d :]

    gate_before = gate().clone() if case.gate else None

    _run(case, rope, q, k, v, pos, kv_cache, slots, qw, kw)

    assert _ulps(q, q_ref) <= MAX_ULPS
    if case.gate:
        assert torch.equal(gate(), gate_before)
    if case.kv_shared:
        assert torch.equal(kv_cache, cache_ref)
        return
    assert _ulps(k, k_ref) <= MAX_ULPS
    d = case.d
    assert _ulps(kv_cache[..., :d], cache_ref[..., :d]) <= MAX_ULPS
    if case.v_norm:
        assert _ulps(kv_cache[..., d:], cache_ref[..., d:]) <= MAX_ULPS
    else:
        assert torch.equal(kv_cache[..., d:], cache_ref[..., d:])


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("name", ["qwen3", "gemma4_full_proportional"])
def test_out_tensors_leave_inputs(name, dtype, monkeypatch):
    """With q_out/k_out/v_out the result lands there and q, k stay
    untouched; v_out holds V as written to the cache."""
    case = CASES[name]
    torch.manual_seed(1)
    rope = _rope(case, dtype, monkeypatch)
    qn, kn, qw, kw = _norms(case, dtype)
    _, q, k, v, pos, kv_cache, slots = _inputs(case, dtype, 4, 2)
    q_ref, k_ref, cache_ref = _reference(
        case, rope, q, k, v, pos, kv_cache, slots, qn, kn
    )
    q0, k0 = q.clone(), k.clone()
    q_out, k_out = torch.empty_like(q0), torch.empty_like(k0)
    v_out = torch.empty_like(v, memory_format=torch.contiguous_format)
    _run(case, rope, q, k, v, pos, kv_cache, slots, qw, kw,
         q_out=q_out, k_out=k_out, v_out=v_out)  # fmt: skip
    assert torch.equal(q, q0) and torch.equal(k, k0)
    assert _ulps(q_out, q_ref) <= MAX_ULPS and _ulps(k_out, k_ref) <= MAX_ULPS
    assert _ulps(kv_cache, cache_ref) <= MAX_ULPS
    block, offset = slots // kv_cache.shape[2], slots % kv_cache.shape[2]
    v_ref = cache_ref[block, :, offset, case.d :]
    assert _ulps(v_out, v_ref) <= MAX_ULPS


@pytest.mark.parametrize("dtype", [torch.float16])
def test_padded_slots_and_hybrid_pages(dtype, monkeypatch):
    """Qwen3.5's 544-token pages with a padded block stride, and CUDA-graph
    padding: a negative slot writes nothing and the tokens past slot_mapping
    are still rotated."""
    case = CASES["qwen35_gate_imrope"]
    torch.manual_seed(2)
    rope = _rope(case, dtype, monkeypatch)
    qn, kn, qw, kw = _norms(case, dtype)
    _, q, k, v, pos, kv_cache, slots = _inputs(
        case, dtype, 4, 2, block_size=544, pad_stride=64
    )
    slots[1] = -1
    q_ref, k_ref, cache_ref = _reference(
        case, rope, q, k, v, pos, kv_cache, slots[:6], qn, kn
    )
    padding = kv_cache.as_strided(
        (64,), (1,), kv_cache.storage_offset() + 544 * 2 * 2 * 256
    )
    pad_before = padding.clone()
    _run(case, rope, q, k, v, pos, kv_cache, slots[:6], qw, kw)
    assert _ulps(q, q_ref) <= MAX_ULPS and _ulps(k, k_ref) <= MAX_ULPS
    assert _ulps(kv_cache, cache_ref) <= MAX_ULPS
    assert torch.equal(padding, pad_before)


@pytest.mark.parametrize("name", ["qwen3", "ministral3_yarn_qscale"])
def test_wrong_position_fails(name, monkeypatch):
    """Negative control: positions off by one must not pass the comparison,
    or the tests above are not reaching the rotation."""
    case = CASES[name]
    dtype = torch.bfloat16
    torch.manual_seed(3)
    rope = _rope(case, dtype, monkeypatch)
    qn, kn, qw, kw = _norms(case, dtype)
    _, q, k, v, pos, kv_cache, slots = _inputs(case, dtype, 4, 8)
    q_ref, k_ref, _ = _reference(case, rope, q, k, v, pos, kv_cache, slots, qn, kn)
    _run(case, rope, q, k, v, pos, kv_cache, slots, qw, kw, pos_shift=1)
    assert _ulps(q, q_ref) > MAX_ULPS and _ulps(k, k_ref) > MAX_ULPS


@pytest.mark.parametrize("name", [n for n in CASES if CASES[n].rope is not None])
def test_rope_reference_matches_class(name, monkeypatch):
    """The fp32 reference above against each rotary class's own forward."""
    case = CASES[name]
    dtype = torch.float16
    torch.manual_seed(4)
    rope = _rope(case, dtype, monkeypatch)
    t = 16
    q = torch.randn(t, case.hq * case.d, dtype=dtype, device="cuda")
    k = torch.randn(t, case.hkv * case.d, dtype=dtype, device="cuda")
    pos = torch.randint(0, min(case.max_pos, 3 * 16384), (t,), device="cuda")
    positions = pos.expand(3, -1).contiguous() if case.mrope else pos
    forward = rope.forward if case.long_rope is not None else rope.forward_native
    q_cls, k_cls = forward(positions, q.clone(), k.clone())
    cs, off = _cache_and_offset(rope, case)
    q_ref = _rope_ref(q.view(t, case.hq, case.d), cs, pos + off)
    k_ref = _rope_ref(k.view(t, case.hkv, case.d), cs, pos + off)
    torch.testing.assert_close(q_cls.view_as(q_ref), q_ref, atol=4e-3, rtol=4e-3)
    torch.testing.assert_close(k_cls.view_as(k_ref), k_ref, atol=4e-3, rtol=4e-3)


def test_opcheck():
    case = CASES["gemma4_sliding"]
    dtype = torch.bfloat16
    rope = get_rope(case.d, case.max_pos, True, case.rope, dtype).cuda()
    _, kn, _, kw = _norms(case, dtype)
    _, q, k, v, pos, kv_cache, slots = _inputs(case, dtype, 2, 4)
    opcheck(
        torch.ops._rocm_C.rdna35_rope_cache,
        (
            pos,
            q,
            k,
            v,
            rope.cos_sin_cache,
            kw,
            kw,
            True,
            EPS,
            0,
            0.0,
            0,
            kv_cache,
            slots,
            None,
            None,
        ),  # fmt: skip
    )
