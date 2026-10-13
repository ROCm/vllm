# SPDX-License-Identifier: Apache-2.0
"""Two KV pools merged into one index stream must attend identically.

DSv4 sparse-MLA prefill reads two sources: the compressed CSA pool and the
sliding-window pool. The aiter kernel can take them as two streams, but that
path costs ~1.8x -- a second stream forces a per-tile column mask the single
stream does not need (measured 1162 -> 648 us at T=16384 H=32 on the prefill
ubench; the decode path saw 449 -> 59 us at T=512 H=128).

It does not have to be paid. vLLM allocates ALL KV from one int8 buffer and
hands each layer a view into it (``allocate_kv_cache``), so the two pools
already share an index space; ``_dsv4_merge_to_unified_kv_buffer`` rebases
them into one stream over one flat 640-byte record grid without copying any
KV. Decode relies on this today.

This test pins the equivalence that makes it legal: merging and then
attending over one pool must give exactly what attending over two pools
gives. It is the assumption the prefill integration rests on, and nothing
covered it -- the aiter unit tests build each pool as its own contiguous
allocation, which is the one shape production never produces.
"""

import pytest
import torch

from vllm.platforms import current_platform

aiter_prefill = pytest.importorskip(
    "aiter.ops.triton.attention.pa_prefill_sparse",
    reason="needs aiter with the gfx1250 packed-fp8 prefill path",
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (  # noqa: E402
    _DSV4_REC_BYTES,
    _DSV4_UNIFIED_KV_BUFFER_STATE,
    _dsv4_merge_to_unified_kv_buffer,
    _dsv4_record_grid,
)

pa_prefill_sparse = aiter_prefill.pa_prefill_sparse

DIM_NOPE = 448
DIM_ROPE = 64
DIM_QK = DIM_NOPE + DIM_ROPE  # 512 B packed plane; the other 128 B are RoPE
TILE = 64
NUM_TILES = DIM_NOPE // TILE  # 7
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm()
    or current_platform.get_device_capability() is None
    or "gfx1250" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="DSv4 packed-fp8 prefill is gfx1250-only",
)


def _pack_2buff(x_bf16):
    """``[N, 512]`` bf16 -> the packed fp8 plane and the bf16 RoPE plane.

    Mirrors ``_v4_pack_2buff`` in aiter's op_tests. The UE8M0 scales must be
    real: a 0xFF pad byte is E8M0 NaN and ``0 * NaN`` poisons a whole score
    row, so this cannot be replaced with random bytes.
    """
    lead = x_bf16.shape[:-1]
    nope = x_bf16[..., :DIM_NOPE].float()
    rope = x_bf16[..., DIM_NOPE:].contiguous()
    tiled = nope.reshape(*lead, NUM_TILES, TILE)
    fp8_max = float(torch.finfo(FP8).max)
    scale = torch.pow(
        2.0, torch.clamp_min(tiled.abs().amax(dim=-1) / fp8_max, 1e-4).log2().ceil()
    )
    nope_fp8 = (tiled / scale.unsqueeze(-1)).to(FP8).reshape(*lead, DIM_NOPE)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    packed = torch.zeros((*lead, DIM_QK), dtype=torch.uint8, device=x_bf16.device)
    packed[..., :DIM_NOPE] = nope_fp8.view(torch.uint8)
    packed[..., DIM_NOPE : DIM_NOPE + 2 * NUM_TILES] = e8m0.repeat_interleave(2, dim=-1)
    return packed.view(FP8), rope


def _build_shared_allocation(rows_per_pool, page):
    """Two ``[nb, page, 640]`` cache views of ONE allocation, as vLLM does.

    Returns ``(caches, planes)``; ``planes[i]`` is pool i's ``(packed, rope)``
    as strided views -- row stride 640 B, aliasing the record grid.
    """
    nbs = [r // page for r in rows_per_pool]
    assert all(nb * page == r for nb, r in zip(nbs, rows_per_pool))
    n_rec = sum(rows_per_pool)
    buf = torch.zeros(n_rec * _DSV4_REC_BYTES, dtype=torch.int8, device="cuda")
    storage = buf.untyped_storage()

    caches, planes, off = [], [], 0
    for nb in nbs:
        c = torch.empty(0, dtype=torch.uint8, device="cuda")
        c.set_(
            storage,
            off * _DSV4_REC_BYTES,
            (nb, page, _DSV4_REC_BYTES),
            (page * _DSV4_REC_BYTES, _DSV4_REC_BYTES, 1),
        )
        caches.append(c)
        rows = c.reshape(nb * page, _DSV4_REC_BYTES)
        planes.append((rows[:, :DIM_QK], rows[:, DIM_QK:]))
        off += nb * page
    return caches, planes


def _fill(plane, values_bf16):
    packed, rope = _pack_2buff(values_bf16)
    plane[0][...] = packed.view(torch.uint8)
    plane[1][...] = rope.reshape(rope.shape[0], -1).view(torch.uint8)


def _ragged(num_tokens, dense_width, lens, num_rows, device):
    """Packed ragged indices in a DENSE-width allocation.

    ``build_ragged_indices_from_dense`` allocates ``num_tokens * max_width``
    and packs the valid prefix into it, so ``numel // num_tokens`` recovers
    the dense width exactly -- which is how the merge bounds its grid without
    a device->host sync. Reproduce that, not a tightly packed array.
    """
    flat = torch.zeros(num_tokens * dense_width, dtype=torch.int32, device=device)
    indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=device)
    torch.cumsum(lens, dim=0, out=indptr[1:])
    for t in range(num_tokens):
        n = int(lens[t])
        if n:
            flat[int(indptr[t]) : int(indptr[t]) + n] = torch.randint(
                0, num_rows, (n,), dtype=torch.int32, device=device
            )
    return flat, indptr


@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("main_width,extra_width", [(96, 32), (256, 128)])
def test_unified_merge_matches_two_pool_prefill(num_tokens, main_width, extra_width):
    torch.manual_seed(0)
    dev = "cuda"
    page = 64
    rows_main, rows_extra = 4096, 2048

    caches, planes = _build_shared_allocation([rows_main, rows_extra], page)
    _fill(planes[0], torch.randn(rows_main, DIM_QK, dtype=torch.bfloat16, device=dev))
    _fill(planes[1], torch.randn(rows_extra, DIM_QK, dtype=torch.bfloat16, device=dev))

    why: list[str] = []
    assert _dsv4_record_grid(caches, why) is not None, (
        f"record grid rejected a shared allocation: {why}"
    )

    heads = 32
    q = torch.randn(num_tokens, heads, DIM_QK, dtype=torch.bfloat16, device=dev)
    q_packed, q_rope = _pack_2buff(q)
    sink = torch.randn(heads, dtype=torch.float32, device=dev)
    scale = DIM_QK**-0.5

    # Ragged, variable per token -- the shape prefill actually produces.
    g = torch.Generator(device=dev).manual_seed(1)
    main_lens = torch.randint(
        1, main_width + 1, (num_tokens,), device=dev, generator=g, dtype=torch.int32
    )
    extra_lens = torch.randint(
        0, extra_width + 1, (num_tokens,), device=dev, generator=g, dtype=torch.int32
    )
    m_idx, m_indptr = _ragged(num_tokens, main_width, main_lens, rows_main, dev)
    e_idx, e_indptr = _ragged(num_tokens, extra_width, extra_lens, rows_extra, dev)

    two_pool = pa_prefill_sparse(
        q_packed,
        planes[0][0].view(FP8),
        m_idx,
        m_indptr,
        planes[1][0].view(FP8),
        e_idx,
        e_indptr,
        sink,
        scale,
        q_rope=q_rope,
        unified_kv_rope=planes[0][1].view(torch.bfloat16),
        kv_rope=planes[1][1].view(torch.bfloat16),
        has_invalid=False,
    )

    merged = _dsv4_merge_to_unified_kv_buffer(
        caches[0], m_idx, m_indptr, caches[1], e_idx, e_indptr
    )
    assert merged is not None, (
        f"merge declined: {_DSV4_UNIFIED_KV_BUFFER_STATE.get('why')}"
    )
    kv, kv_rope, u_idx, u_indptr = merged
    assert int(u_indptr[-1]) == int(m_indptr[-1]) + int(e_indptr[-1])

    one_pool = pa_prefill_sparse(
        q_packed,
        kv,
        u_idx,
        u_indptr,
        None,
        None,
        None,
        sink,
        scale,
        q_rope=q_rope,
        unified_kv_rope=kv_rope,
        has_invalid=False,
    )

    torch.testing.assert_close(
        one_pool.float(),
        two_pool.float(),
        atol=2e-2,
        rtol=2e-2,
        msg=lambda m: f"merged single-pool diverged from the two-pool path\n{m}",
    )

    # Negative control. The agreement above is only meaningful if the merged
    # stream actually reaches the extra pool: if its rebased indices silently
    # landed back in the main pool, or the extra segment were dropped, both
    # paths could still agree on main-only output and the test would pass
    # while proving nothing. Corrupting the extra pool must move the result.
    planes[1][0][...] = 0
    planes[1][1][...] = 0
    after = pa_prefill_sparse(
        q_packed,
        kv,
        u_idx,
        u_indptr,
        None,
        None,
        None,
        sink,
        scale,
        q_rope=q_rope,
        unified_kv_rope=kv_rope,
        has_invalid=False,
    )
    assert not torch.allclose(after.float(), one_pool.float(), atol=1e-3), (
        "zeroing the extra pool did not change the merged result -- the "
        "merged stream never reads it, so the equivalence above is vacuous"
    )


def test_paged_prefill_step_cache_is_per_metadata_object():
    """One table per metadata object, dying with it.

    Two properties, both of which have already been got wrong once:

    1. Alternating objects must each keep their own table. DSv4 interleaves
       HCA and CSA layers, which belong to separate KV-cache groups and so
       carry separate metadata objects. A single module-level slot keyed on
       "the current metadata" thrashes between them and never hits -- that is
       why trace_15 came out byte-identical to the unhoisted trace_14.
    2. An entry must not outlive its step, or a later step would attend to a
       previous step's rows without crashing.
    """
    from vllm.models.deepseek_v4.amd.rocm import _dsv4_paged_prefill_step_cache

    class Meta:  # stand-in for the per-step metadata object
        pass

    hca, csa = Meta(), Meta()
    _dsv4_paged_prefill_step_cache(hca)["geom"] = "hca-indices"
    _dsv4_paged_prefill_step_cache(csa)["geom"] = "csa-indices"

    # Interleaved exactly as the layers run: 128,128,4,128,4,...
    for i, obj in enumerate([hca, hca, csa, hca, csa, hca, csa] * 8):
        want = "hca-indices" if obj is hca else "csa-indices"
        got = _dsv4_paged_prefill_step_cache(obj)
        assert got["geom"] == want, f"thrashed at interleave step {i}"

    # Same object -> same table; different object -> its own.
    assert _dsv4_paged_prefill_step_cache(hca) is _dsv4_paged_prefill_step_cache(hca)
    assert _dsv4_paged_prefill_step_cache(hca) is not _dsv4_paged_prefill_step_cache(csa)

    # Next step's metadata is a new object, so it starts empty and the old
    # table is reachable only from the old object (which the step drops).
    nxt = Meta()
    assert "geom" not in _dsv4_paged_prefill_step_cache(nxt)

    # Frozen/slots metadata must degrade to no caching rather than raise.
    class Slotted:
        __slots__ = ()

    t = _dsv4_paged_prefill_step_cache(Slotted())
    assert t == {}
