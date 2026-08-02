# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`BatchedTritonExperts` must not see how many tokens its experts were handed.

This is the experts class behind `FusedMoEActivationFormat.BatchedExperts`, and
on ROCm it is the only one: DeepEP low latency and NIXL EP both force that
format (`FusedMoEParallelConfig.use_batched_activation_format`), and the
unquantized oracle's other two ROCm candidates -- AITER and `TritonExperts` --
are both `Standard`. So `_supports_batch_invariance()` on this one class decides
whether DeepEP LL can be brought up under the mode at all, and with it whether
`--enable-dbo` has a second admissible backend.

It runs a *different kernel* from `fused_moe_kernel` on a different layout
(`E x max_num_tokens x K`), so none of what was measured for the plain expert
GEMM carries over. What this file measures is the kernel it actually launches,
`batched_triton_kernel`, plus the whole class through the reference no-comms
`BatchedPrepareAndFinalize`.

The original sweep on gfx950 (E=8, K=1024, N=512, max_num_tokens=256) covered
four operand distributions -- bf16 with an exponent spread of +-20, bf16 flat,
fp16 +-14, fp32 flat. Every arm: 41 launch configurations collapsed to exactly 1
bitwise result, 17 token counts from 1 to 256 moved no row, a per-expert row
derangement relocating 68% of rows moved no bits once un-permuted, and 16
repeats at a fixed count gave 1 result. The whole class through
`BatchedPrepareAndFinalize` moved 0 of 2426 rows across 14 batch sizes, both by
appending tokens and by dropping them from the front (~1750 slot relocations).
End to end, `test_ep_all2all_batch_invariant.py` holds the needle bitwise over
DeepEP LL while the needle rank's padding goes 40 -> 220, against a mode-off
control that moves 31 of 32.

This file keeps the kernel-level half of that, which is the cheap half: it needs
one GPU and no model. Of the four distributions it keeps three, dropping bf16
flat as strictly the weakest -- it is the same dtype as the spread arm with less
accumulator slack to expose. The fp8 schemes are deliberately absent: they are
withheld under the mode by `_supports_quant_scheme` because that path is not
run-to-run reproducible, which is recorded there.

Three structural facts about the kernel, from reading it, that the numbers
below are testing rather than assuming:

  * There is no split-K and no `tl.atomic_*` of any kind in
    `fused_batched_moe.py`, and no `@triton.autotune`. Each CTA owns a disjoint
    `[BLOCK_M, BLOCK_N]` tile of one expert's output and runs the whole K loop
    into a single fp32 accumulator.
  * The grid is `(E, cdiv(max_num_tokens, BLOCK_M) * cdiv(N, BLOCK_N))` --
    keyed on the *buffer* size, a deployment constant, not on the runtime token
    count. Token r therefore always lands in tile `r // BLOCK_M` at lane
    `r % BLOCK_M`. The token count enters only through `mask_m` and two early
    exits.
  * `invoke_moe_batched_triton_kernel` forwards only the three block sizes;
    `num_warps` and `num_stages` are never passed, so Triton's defaults are
    used in production and the tuning table is doubly irrelevant here
    (`try_get_optimal_moe_config` is called with `M=max_num_tokens`, and under
    the mode `get_moe_configs` short-circuits to `None` anyway).

Non-vacuity. An fp32 accumulation that happens to be exact cannot detect a
reordering, so every arm below carries `_reorder_is_detectable`, which sums the
same products forward, backward and as a two-way split-K in fp32 and requires
the three to disagree bitwise. The operands are built with a widened exponent
spread for that reason (+-20 for bf16, +-14 for fp16 -- fp16 saturates at 65504
and a spread of 15+ starts producing infs, which read as differences and poison
the metric), and the spread is applied to the activation only so the fp16
*output* cannot overflow. fp32 with no spread has no accumulator headroom and is
the sharpest detector of the three.
"""

import hashlib
import itertools

import pytest
import torch
from utils import skip_if_not_cuda_alike

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    batched_triton_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

DEVICE_TYPE = current_platform.device_type

E = 8
K = 1024
N = 512
# max_num_tokens: the dispatch buffer's second dimension. A deployment
# constant (`FusedMoEConfig.max_num_tokens`, default 256 for batched DP), never
# the batch size -- pinning it here is what makes the token-count sweep below a
# test of the count rather than of the buffer shape.
T = 256

_TL = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}
# Spread applied to the activation only; see the module docstring.
CASES = [
    ("bf16", torch.bfloat16, 20),
    ("fp16", torch.float16, 14),
    ("fp32", torch.float32, 0),
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    return t.contiguous().view(view)


def _digest(t: torch.Tensor) -> str:
    return hashlib.blake2b(_bits(t).cpu().numpy().tobytes(), digest_size=8).hexdigest()


def _launch(A, B, C, ent, BM, BN, BK, num_warps=None, num_stages=None):
    """`invoke_moe_batched_triton_kernel`'s unquantized path, with num_warps
    and num_stages exposed. Production passes neither -- that is the point of
    exposing them: if they were live they would still have to be neutral."""
    e, t, k = A.shape
    n = C.shape[2]
    grid = (e, triton.cdiv(t, BM) * triton.cdiv(n, BN))
    extra = {}
    if num_warps is not None:
        extra["num_warps"] = num_warps
    if num_stages is not None:
        extra["num_stages"] = num_stages
    batched_triton_kernel[grid](
        A,
        B,
        C,
        ent,
        _TL[A.dtype],
        t,
        k,
        n,
        None,
        None,
        None,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        False,
        False,
        False,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        USE_TD=False,
        **extra,
    )
    return C


def _operands(dtype, spread, seed):
    device = torch.device(DEVICE_TYPE)
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((E, T, K), generator=g, device=device, dtype=torch.float32)
    if spread:
        ex = torch.randint(-spread, spread + 1, (E, T, K), generator=g, device=device)
        a = a * torch.exp2(ex.float())
        # fp16 saturates at 65504 and randn's tail over millions of draws
        # reaches ~5 sigma, so +-14 overflows on its own. Rescale by a power of
        # two, which is exact: every mantissa -- and so the whole reordering
        # sensitivity this spread exists to create -- survives, only the
        # exponents shift.
        lim = torch.finfo(dtype).max
        shift = torch.ceil(torch.log2(a.abs().max() / (lim * 0.25))).clamp(min=0)
        a = a / torch.exp2(shift)
    b = torch.randn((E, N, K), generator=g, device=device, dtype=torch.float32) / (
        K**0.5
    )
    A, B = a.to(dtype).contiguous(), b.to(dtype).contiguous()
    assert torch.isfinite(A).all() and torch.isfinite(B).all(), (
        "the exponent spread overflowed the operand dtype; a run with infs in "
        "it measures nothing"
    )
    return A, B


def _zeros(dtype):
    return torch.zeros((E, T, N), device=torch.device(DEVICE_TYPE), dtype=dtype)


def _counts(m):
    return torch.full((E,), m, device=torch.device(DEVICE_TYPE), dtype=torch.int32)


def _rows_that_differ(x, y, e, upto):
    ne = _bits(x[e, :upto]) != _bits(y[e, :upto])
    return torch.nonzero(ne.reshape(upto, -1).any(dim=1)).flatten()


def _reorder_is_detectable(A, B, e=1, m=7, n=3) -> bool:
    """Would *these* operands notice a change in K accumulation order?

    Forward, reverse and two-way split-K fp32 reductions over the same
    products. If all three agree bitwise the sum is exact and every invariance
    assertion in this file would pass without measuring anything.
    """
    p = (A[e, m].float() * B[e, n].float()).tolist()

    def acc(xs):
        s = torch.zeros((), dtype=torch.float32)
        for v in xs:
            s = s + v
        return int(_bits(s.reshape(1))[0])

    h = len(p) // 2
    fwd, rev = acc(p), acc(reversed(p))
    lo = torch.zeros((), dtype=torch.float32)
    for v in p[:h]:
        lo = lo + v
    hi = torch.zeros((), dtype=torch.float32)
    for v in p[h:]:
        hi = hi + v
    split = int(_bits((lo + hi).reshape(1))[0])
    return fwd != rev and fwd != split


def _assert_metric_sees_real_data(C, e, upto):
    x = C[e, :upto]
    assert torch.isfinite(x.float()).all(), "kernel output has non-finite values"
    assert len({_digest(x[i]) for i in range(upto)}) == upto, (
        "output rows are not all distinct, so a bitwise comparison over them "
        "is not measuring what it claims to"
    )


@pytest.fixture
def mode_on(enable_batch_invariant_mode):
    """Function-scoped, and explicitly dependent on the autouse fixture so a
    copy of this file that overrode it would still be running the mode."""
    assert envs.VLLM_BATCH_INVARIANT, "VLLM_BATCH_INVARIANT is not set"
    return True


# --------------------------------------------------------------------------- #
@skip_if_not_cuda_alike
@pytest.mark.parametrize("name,dtype,spread", CASES)
@torch.inference_mode()
def test_batched_expert_gemm_is_launch_config_invariant(mode_on, name, dtype, spread):
    """Every launch configuration must produce the same bits.

    One knob at a time off the batch-invariant default (BLOCK_M/N=64,
    BLOCK_K=32), plus a BLOCK_K x num_warps cross and a few extreme tiles --
    those are where a backend would be most tempted to distribute the dot along
    K and add a cross-warp reduction, which is the only thing here that could
    change the accumulation order.
    """
    A, B = _operands(dtype, spread, seed=1234)
    assert _reorder_is_detectable(A, B), (
        f"{name}: this operand distribution sums exactly, so it cannot detect "
        "a reordering and nothing below would mean anything"
    )
    ent = torch.tensor(
        [1, 7, 63, 64, 65, 128, 200, 256],
        device=torch.device(DEVICE_TYPE),
        dtype=torch.int32,
    )

    Cfg = tuple[int, int, int, int | None, int | None]
    cfgs: set[Cfg] = {(64, 64, 32, None, None)}
    for v in (16, 32, 128):
        cfgs.add((v, 64, 32, None, None))
        cfgs.add((64, v, 32, None, None))
    for bk, w in itertools.product((16, 32, 64, 128, 256), (1, 2, 4, 8)):
        cfgs.add((64, 64, bk, w, None))
    for s in (1, 2, 3, 4):
        cfgs.add((64, 64, 32, None, s))
    for (bm, bn), w in itertools.product(((16, 16), (16, 256), (128, 128)), (1, 4, 8)):
        cfgs.add((bm, bn, 32, w, None))

    buckets: dict[str, list] = {}
    ran = 0
    for bm, bn, bk, warps, stages in sorted(cfgs, key=str):
        C = _zeros(dtype)
        try:
            _launch(A, B, C, ent, bm, bn, bk, num_warps=warps, num_stages=stages)
            torch.accelerator.synchronize()
        except Exception:
            # out-of-resources / unsupported layout: not a determinism result
            continue
        ran += 1
        key = "|".join(_digest(C[e, : int(ent[e])]) for e in range(E))
        buckets.setdefault(key, []).append((bm, bn, bk, warps, stages))

    assert ran >= 20, f"{name}: only {ran} configurations compiled; too few to conclude"
    assert len(buckets) == 1, (
        f"{name}: {ran} launch configurations produced {len(buckets)} distinct "
        f"bit patterns, so the accumulation order depends on the tiling: "
        f"{ {k[:12]: v[:4] for k, v in buckets.items()} }"
    )


@skip_if_not_cuda_alike
@pytest.mark.parametrize("name,dtype,spread", CASES)
@torch.inference_mode()
def test_batched_expert_gemm_is_token_count_invariant(mode_on, name, dtype, spread):
    """A token's row must not change when its expert is handed more tokens.

    The operands are fixed and only `expert_num_tokens` moves, so this isolates
    the one thing the runtime batch controls in this kernel: `mask_m`, and with
    it how many rows of the last partial `BLOCK_M` tile are live. Counts either
    side of 64 and 128 are included because those are the tile boundaries at
    the batch-invariant `BLOCK_M`.
    """
    A, B = _operands(dtype, spread, seed=1234)
    assert _reorder_is_detectable(A, B), f"{name}: operands sum exactly"

    ref = _zeros(dtype)
    _launch(A, B, ref, _counts(T), 64, 64, 32)
    torch.accelerator.synchronize()
    _assert_metric_sees_real_data(ref, 1, 64)

    again = _zeros(dtype)
    _launch(A, B, again, _counts(T), 64, 64, 32)
    torch.accelerator.synchronize()
    assert torch.equal(_bits(ref), _bits(again)), (
        f"{name}: the kernel is not even run-to-run stable at a fixed count; "
        "nothing below is interpretable"
    )

    bad = {}
    for m in (1, 2, 7, 16, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256):
        C = _zeros(dtype)
        _launch(A, B, C, _counts(m), 64, 64, 32)
        torch.accelerator.synchronize()
        for e in range(E):
            d = _rows_that_differ(ref, C, e, m)
            if d.numel():
                bad[(m, e)] = d[:8].tolist()
    assert not bad, (
        f"{name}: rows moved when only the token count changed -- "
        f"{ {k: v for k, v in list(bad.items())[:6]} }"
    )


@skip_if_not_cuda_alike
@pytest.mark.parametrize("name,dtype,spread", CASES)
@torch.inference_mode()
def test_batched_expert_gemm_is_row_permutation_invariant(mode_on, name, dtype, spread):
    """Row order inside an expert's buffer is nondeterministic; ignore it.

    DeepEP low latency's dispatch assigns each token its slot with an atomic
    increment (`internode_ll.cu`), so whichever warp arrives first gets the low
    slot. The routing stays correct because the handle records each token's
    slot, which makes row-permutation invariance of this GEMM load bearing --
    the same assumption `test_moe_row_permutation_batch_invariant.py` measures
    for the `Standard` format kernel.
    """
    device = torch.device(DEVICE_TYPE)
    m = 200
    A, B = _operands(dtype, spread, seed=1234)
    assert _reorder_is_detectable(A, B), f"{name}: operands sum exactly"
    g = torch.Generator(device=device).manual_seed(99)
    ent = _counts(m)

    ref = _zeros(dtype)
    _launch(A, B, ref, ent, 64, 64, 32)

    Ap = A.clone()
    pis = []
    ar = torch.arange(m, device=device)
    for e in range(E):
        pi = torch.randperm(m, generator=g, device=device)
        for _ in range(64):
            if bool((pi != ar).all()):
                break
            pi = torch.randperm(m, generator=g, device=device)
        pis.append(pi)
        Ap[e, :m] = A[e, pi]
    perm = _zeros(dtype)
    _launch(Ap, B, perm, ent, 64, 64, 32)
    torch.accelerator.synchronize()

    # Vacuity guard: the rows must genuinely relocate to other BLOCK_M tiles.
    moved = sum(int(((p // 64) != (ar // 64)).sum()) for p in pis) / (E * m)
    assert moved > 0.5, (
        f"{name}: only {moved:.1%} of rows changed tile, so a pass would mean nothing"
    )

    bad = {}
    for e in range(E):
        inv = torch.empty_like(pis[e])
        inv[pis[e]] = ar
        un = perm[e, :m][inv]
        ne = _bits(ref[e, :m]) != _bits(un)
        d = torch.nonzero(ne.reshape(m, -1).any(dim=1)).flatten()
        if d.numel():
            bad[e] = d[:8].tolist()
        # Positive control: without the un-permute the same comparison must
        # fail, otherwise it is blind to row identity.
        assert _rows_that_differ(ref, perm, e, m).numel() > 0
    assert not bad, (
        f"{name}: {len(bad)} experts changed bits when their rows were "
        f"permuted (first offenders: {list(bad.items())[:4]})"
    )
