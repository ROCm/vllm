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

This is the kernel-level half of the certification, and the cheap half: it needs
one GPU and no model. The end-to-end half is in
`test_ep_all2all_batch_invariant.py`, which runs the same class over DeepEP LL.

The fp8 arms are what admit the fp8 pairs in `_supports_quant_scheme` under the
mode. They were originally withheld as *irreproducible*, which turned out to be
an unmasked out-of-bounds read of the weight scale in the launcher rather than
anything about the arithmetic; the regression guard for that read is
`test_batched_moe_weight_scale_is_read_as_a_broadcast` at the end. Note that
everything here goes through the no-comms `BatchedPrepareAndFinalize`, so it
cannot reach `use_fp8_dispatch`, where DeepEP LL quantizes inside `_do_quant`
and carries the scales through its buffers; that path has its own e2e arm,
`test_deepep_low_latency_fp8_block_dispatch_engages_end_to_end`.

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
from utils import bits, rows_that_differ, skip_if_not_cuda_alike

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
    batched_triton_kernel,
    invoke_moe_batched_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
    BatchedPrepareAndFinalize,
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


def _digest(t: torch.Tensor) -> str:
    return hashlib.blake2b(bits(t).cpu().numpy().tobytes(), digest_size=8).hexdigest()


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
        return int(bits(s.reshape(1))[0])

    h = len(p) // 2
    fwd, rev = acc(p), acc(reversed(p))
    lo = torch.zeros((), dtype=torch.float32)
    for v in p[:h]:
        lo = lo + v
    hi = torch.zeros((), dtype=torch.float32)
    for v in p[h:]:
        hi = hi + v
    split = int(bits((lo + hi).reshape(1))[0])
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

# A launch config that will not fit is not a determinism result and has to be
# skipped -- but a bare `except Exception` also swallows a compile error or a
# kernel fault, and this sweep only requires `ran >= 20` out of ~40 configs, so
# half of it could vanish and the test would still pass at the threshold. Name
# the two things that are legitimately skippable and re-raise everything else.
_SKIPPABLE_LAUNCH_ERRORS: tuple[type[BaseException], ...] = tuple(
    e
    for e in (
        getattr(
            __import__("triton.runtime.errors", fromlist=["OutOfResources"]),
            "OutOfResources",
            None,
        ),
        getattr(
            __import__("triton.compiler.errors", fromlist=["CompilationError"]),
            "CompilationError",
            None,
        ),
    )
    if e is not None
)
assert _SKIPPABLE_LAUNCH_ERRORS, "no Triton error types to narrow on"


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
    skipped: list[tuple] = []
    ran = 0
    for bm, bn, bk, warps, stages in sorted(cfgs, key=str):
        C = _zeros(dtype)
        try:
            _launch(A, B, C, ent, bm, bn, bk, num_warps=warps, num_stages=stages)
            torch.accelerator.synchronize()
        except _SKIPPABLE_LAUNCH_ERRORS as exc:
            skipped.append((bm, bn, bk, warps, stages, type(exc).__name__))
            continue
        ran += 1
        key = "|".join(_digest(C[e, : int(ent[e])]) for e in range(E))
        buckets.setdefault(key, []).append((bm, bn, bk, warps, stages))

    assert ran >= 20, (
        f"{name}: only {ran} configurations compiled; too few to conclude. "
        f"Skipped for want of resources: {skipped}"
    )
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
    assert torch.equal(bits(ref), bits(again)), (
        f"{name}: the kernel is not even run-to-run stable at a fixed count; "
        "nothing below is interpretable"
    )

    bad = {}
    for m in (1, 2, 7, 16, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256):
        C = _zeros(dtype)
        _launch(A, B, C, _counts(m), 64, 64, 32)
        torch.accelerator.synchronize()
        for e in range(E):
            d = rows_that_differ(ref[e, :m], C[e, :m])
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
        ne = bits(ref[e, :m]) != bits(un)
        d = torch.nonzero(ne.reshape(m, -1).any(dim=1)).flatten()
        if d.numel():
            bad[e] = d[:8].tolist()
        # Positive control: without the un-permute the same comparison must
        # fail, otherwise it is blind to row identity.
        assert rows_that_differ(ref[e, :m], perm[e, :m]).numel() > 0
    assert not bad, (
        f"{name}: {len(bad)} experts changed bits when their rows were "
        f"permuted (first offenders: {list(bad.items())[:4]})"
    )


# --------------------------------------------------------------------------- #
# fp8.
#
# `_supports_quant_scheme` advertises five (weight, activation) pairs, which
# reach `moe_mmk` as only four distinct arithmetics -- the scale shapes are what
# the kernel branches on, not the pair:
#
#   block             (kFp8Static128BlockSym, kFp8Dynamic128Sym)
#                     group_n/group_k > 0: a scale per (token, k-tile), a weight
#                     scale per (n-tile, k-tile), applied *inside* the K loop.
#   channel           (kFp8StaticChannelSym, kFp8DynamicTokenSym)
#                     per_act_token_quant with a genuine 3-D `[E, N, 1]` weight
#                     scale.
#   tensor_w_token_a  (kFp8StaticTensorSym, kFp8DynamicTokenSym), and the
#                     runtime form of (kFp8StaticTensorSym, kFp8DynamicTensorSym)
#                     once `maybe_promote_act_quant_for_batch_invariance` has
#                     promoted the activation. per_act_token_quant with a
#                     broadcast `[E]` weight scale -- the shape the launcher used
#                     to give stride 1 and read off the end of.
#   tensor            (kFp8StaticTensorSym, kFp8StaticTensorSym)
#                     one scalar scale per expert on each side.
#
# Only the first two carry over anything from the unquantized arms above: they
# are the only ones whose index arithmetic the unquantized path also exercises
# (it exercises neither, in fact -- there is no scale at all -- which is why all
# four are measured here rather than argued from the bf16 result).
#
# Non-vacuity is sharper here than above, not softer. fp8 e4m3 carries three
# mantissa bits, so a product of two fp8 values needs seven bits and 1024 of
# them fall on a common fp32 grid unless their exponents are spread: with
# `randn` operands quantized in the ordinary way, forward, reverse and split-K
# fp32 reductions over the K products agree *bitwise* in all four schemes, and
# every arm below would pass without measuring anything. `_FP8_A_SPREAD` and
# `_FP8_B_SPREAD` exist for that reason and `_assert_fp8_reorder_is_detectable`
# enforces it. Detectability is a property of the row, not of the scheme, so it
# is sampled over six (expert, token, out-channel) triples and a majority is
# required, which the spreads below achieve in every scheme.
# --------------------------------------------------------------------------- #

GROUP = 128
_FP8_A_SPREAD = 8
_FP8_B_SPREAD = 4

# per_act_token_quant, block_shape
_FP8_SCHEMES: dict[str, tuple[bool, list[int] | None]] = {
    "block": (False, [GROUP, GROUP]),
    "channel": (True, None),
    "tensor_w_token_a": (True, None),
    "tensor": (False, None),
}

# Sampled for the non-vacuity check. None of them is (0, 0, 0): expert 0, token
# 0 and channel 0 are all privileged positions in this kernel's indexing, and a
# scale bug that collapses everything to index 0 is invisible from there.
_FP8_TRIPLES = [(1, 7, 3), (5, 199, 511), (3, 64, 128), (7, 255, 64), (2, 1, 1)]

_fp8_skip = pytest.mark.skipif(
    not current_platform.supports_fp8(), reason="requires fp8 support"
)


def _fp8_spread_normal(shape, g, spread):
    a = torch.randn(shape, generator=g, device=torch.device(DEVICE_TYPE))
    if spread:
        ex = torch.randint(
            -spread, spread + 1, shape, generator=g, device=torch.device(DEVICE_TYPE)
        )
        a = a * torch.exp2(ex.float())
    return a


def _fp8_operands(scheme, seed=1234):
    """fp8 A `[E, T, K]` and B `[E, N, K]` with the scales the scheme implies."""
    fp8 = current_platform.fp8_dtype()
    fmax = torch.finfo(fp8).max
    g = torch.Generator(device=torch.device(DEVICE_TYPE)).manual_seed(seed)
    a = _fp8_spread_normal((E, T, K), g, _FP8_A_SPREAD)
    b = _fp8_spread_normal((E, N, K), g, _FP8_B_SPREAD) / (K**0.5)

    def q(x, scale):
        return (x / scale).clamp(-fmax, fmax).to(fp8)

    if scheme == "block":
        av = a.view(E, T, K // GROUP, GROUP)
        a_scale = (av.abs().amax(dim=-1, keepdim=True) / fmax).clamp(min=1e-12)
        A = q(av, a_scale).view(E, T, K)
        a_scale = a_scale.squeeze(-1).contiguous()
        bv = b.view(E, N // GROUP, GROUP, K // GROUP, GROUP)
        b_scale = (bv.abs().amax(dim=(2, 4), keepdim=True) / fmax).clamp(min=1e-12)
        B = q(bv, b_scale).view(E, N, K)
        b_scale = b_scale.reshape(E, N // GROUP, K // GROUP).contiguous()
    elif scheme in ("channel", "tensor_w_token_a"):
        a_scale = (a.abs().amax(dim=-1, keepdim=True) / fmax).clamp(min=1e-12)
        A = q(a, a_scale)
        a_scale = a_scale.contiguous()
        if scheme == "channel":
            b_scale = (b.abs().amax(dim=-1, keepdim=True) / fmax).clamp(min=1e-12)
            B = q(b, b_scale)
            b_scale = b_scale.contiguous()
        else:
            b_scale = (b.abs().amax(dim=(1, 2)) / fmax).clamp(min=1e-12)
            B = q(b, b_scale.view(E, 1, 1))
            b_scale = b_scale.contiguous()
    else:
        a_scale = (a.abs().amax(dim=(1, 2)) / fmax).clamp(min=1e-12)
        A = q(a, a_scale.view(E, 1, 1))
        a_scale = a_scale.view(E, 1, 1).contiguous()
        b_scale = (b.abs().amax(dim=(1, 2)) / fmax).clamp(min=1e-12)
        B = q(b, b_scale.view(E, 1, 1))
        b_scale = b_scale.contiguous()
    return A, B, a_scale, b_scale


def _launch_fp8(scheme, ops, ent, C, bm=64, bn=64, bk=32):
    """Through `invoke_moe_batched_triton_kernel`, not the kernel directly.

    The unquantized `_launch` above bypasses the launcher to expose num_warps.
    Here the launcher is part of what is under test: it is where the scale
    strides are computed, and that computation is what was wrong.
    """
    per_act_token_quant, block_shape = _FP8_SCHEMES[scheme]
    A, B, a_scale, b_scale = ops
    invoke_moe_batched_triton_kernel(
        A=A,
        B=B,
        C=C,
        expert_num_tokens=ent,
        compute_type=tl.bfloat16,
        A_scale=a_scale,
        B_scale=b_scale,
        B_zp=None,
        use_fp8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk},
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
    )
    return C


def _fp32_sum(values) -> torch.Tensor:
    s = torch.zeros((), dtype=torch.float32)
    for v in values:
        s = s + v
    return s


def _fp8_reorder_is_detectable(scheme, ops, e, m, n) -> bool:
    """Would the K reduction for this output element notice a reordering?

    Modelled on the reduction the kernel performs, which differs by scheme: for
    `block` the accumulator holds one scaled term per BLOCK_K chunk, for the
    others the raw fp8 products accumulate and the scales apply at the end.
    """
    A, B, a_scale, b_scale = ops
    aq, bq = A[e, m].float(), B[e, n].float()
    if scheme == "block":
        terms = []
        for k0 in range(0, K, 32):
            chunk = _fp32_sum((aq[k0 : k0 + 32] * bq[k0 : k0 + 32]).tolist())
            gk = k0 // GROUP
            terms.append(
                chunk * a_scale[e, m, gk].cpu() * b_scale[e, n // GROUP, gk].cpu()
            )
        p = terms
    else:
        p = (aq * bq).tolist()
    h = len(p) // 2
    fwd = int(bits(_fp32_sum(p).reshape(1))[0])
    rev = int(bits(_fp32_sum(list(reversed(p))).reshape(1))[0])
    split = int(bits((_fp32_sum(p[:h]) + _fp32_sum(p[h:])).reshape(1))[0])
    return fwd != rev and fwd != split


def _assert_fp8_reorder_is_detectable(scheme, ops, minimum=3):
    hits = [_fp8_reorder_is_detectable(scheme, ops, *t) for t in _FP8_TRIPLES]
    assert sum(hits) >= minimum, (
        f"{scheme}: only {sum(hits)} of {len(_FP8_TRIPLES)} sampled output "
        "elements have a K reduction that can tell accumulation orders apart, "
        "so this operand set is too close to exactly summable for the "
        f"assertions below to mean anything ({hits})"
    )


# --------------------------------------------------------------------------- #
@skip_if_not_cuda_alike
@_fp8_skip
@pytest.mark.parametrize("scheme", list(_FP8_SCHEMES))
@torch.inference_mode()
def test_batched_expert_fp8_gemm_is_reproducible(mode_on, scheme):
    """Repeatability before invariance.

    This is the arm that failed before the launcher gave a broadcast weight
    scale stride 0: `apply()` gave 2 distinct results over 4 identical calls
    because `moe_mmk` was reading whatever the caching allocator had left after
    an `[E]` tensor. A path that cannot repeat itself cannot be batch
    invariant, and calling it "variant" would understate the defect, so every
    other fp8 arm in this file is conditional on this one.
    """
    ops = _fp8_operands(scheme)
    _assert_fp8_reorder_is_detectable(scheme, ops)
    seen = set()
    for _ in range(16):
        C = _zeros(torch.bfloat16)
        _launch_fp8(scheme, ops, _counts(200), C)
        torch.accelerator.synchronize()
        _assert_metric_sees_real_data(C, 1, 64)
        seen.add(_digest(C))
    assert len(seen) == 1, (
        f"{scheme}: 16 identical calls produced {len(seen)} distinct results"
    )


@skip_if_not_cuda_alike
@_fp8_skip
@pytest.mark.parametrize("scheme", list(_FP8_SCHEMES))
@torch.inference_mode()
def test_batched_expert_fp8_gemm_is_launch_config_invariant(mode_on, scheme):
    """Every reachable launch configuration must produce the same bits.

    Reachable is narrower than for the unquantized path, and for one scheme it
    is decisive. Under the mode `get_default_config` returns 64/64/32 for every
    M, dtype and block shape (asserted below, since the rest of this test rests
    on it), and `try_get_optimal_moe_config` is called with `M=max_num_tokens`,
    a deployment constant. So exactly one configuration runs in production.

    Three of the four schemes are invariant across the whole grid anyway,
    BLOCK_K included: the scales multiply the accumulator once, after the K
    loop, and `tl.dot(..., acc=)` chains the MFMA steps in the same order
    whatever the K tile is. `block` is not, and cannot be: it applies
    `a_scale * b_scale` to each BLOCK_K chunk *inside* the loop, so the K tile
    decides how the scaled partials are grouped -- BLOCK_K 16, 32, 64 and 128
    give four different bit patterns here, all of them numerically correct
    (0.14% median relative error against a dequantized reference). BLOCK_K=256
    is a different matter: it exceeds `group_k` and reuses one k-group's scale
    for two groups, which is simply wrong (25% median relative error). Neither
    is a batch-invariance failure -- nothing in either depends on the batch --
    but the blockwise arm is therefore scoped to the reachable BLOCK_K, and the
    default-config assertion is what keeps that scoping honest.
    """
    for m in (1, 8, 64, 256):
        for block_shape in (None, [GROUP, GROUP]):
            cfg = try_get_optimal_moe_config(
                (E, 2 * N, K), (E, K, N), 2, "fp8_w8a8", m, block_shape=block_shape
            )
            assert (
                cfg["BLOCK_SIZE_M"],
                cfg["BLOCK_SIZE_N"],
                cfg["BLOCK_SIZE_K"],
            ) == (64, 64, 32), f"unexpected batch-invariant default config {cfg}"

    ops = _fp8_operands(scheme)
    _assert_fp8_reorder_is_detectable(scheme, ops)
    ent = torch.tensor(
        [1, 7, 63, 64, 65, 128, 200, 256],
        device=torch.device(DEVICE_TYPE),
        dtype=torch.int32,
    )

    cfgs = {(64, 64, 32)}
    for v in (16, 32, 128):
        cfgs.add((v, 64, 32))
        cfgs.add((64, v, 32))
    for bm, bn in ((16, 16), (16, 256), (128, 128)):
        cfgs.add((bm, bn, 32))
    if scheme != "block":
        for bk in (16, 64, 128, 256):
            cfgs.add((64, 64, bk))

    buckets: dict[str, list] = {}
    skipped: list[tuple] = []
    ran = 0
    for bm, bn, bk in sorted(cfgs):
        C = _zeros(torch.bfloat16)
        try:
            _launch_fp8(scheme, ops, ent, C, bm, bn, bk)
            torch.accelerator.synchronize()
        except _SKIPPABLE_LAUNCH_ERRORS as exc:
            skipped.append((bm, bn, bk, type(exc).__name__))
            continue
        ran += 1
        key = "|".join(_digest(C[e, : int(ent[e])]) for e in range(E))
        buckets.setdefault(key, []).append((bm, bn, bk))

    assert ran >= 9, (
        f"{scheme}: only {ran} configurations compiled. "
        f"Skipped for want of resources: {skipped}"
    )
    assert len(buckets) == 1, (
        f"{scheme}: {ran} launch configurations produced {len(buckets)} "
        f"distinct bit patterns: { {k[:12]: v[:4] for k, v in buckets.items()} }"
    )


@skip_if_not_cuda_alike
@_fp8_skip
@pytest.mark.parametrize("scheme", list(_FP8_SCHEMES))
@torch.inference_mode()
def test_batched_expert_fp8_gemm_is_token_count_invariant(mode_on, scheme):
    """A token's row must not change when its expert is handed more tokens.

    The operands and every scale are fixed; only `expert_num_tokens` moves. The
    counts straddle both BLOCK_M boundaries at the mode's BLOCK_M of 64.
    """
    ops = _fp8_operands(scheme)
    _assert_fp8_reorder_is_detectable(scheme, ops)

    ref = _zeros(torch.bfloat16)
    _launch_fp8(scheme, ops, _counts(T), ref)
    torch.accelerator.synchronize()
    _assert_metric_sees_real_data(ref, 1, 64)

    bad = {}
    for m in (1, 2, 7, 16, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256):
        C = _zeros(torch.bfloat16)
        _launch_fp8(scheme, ops, _counts(m), C)
        torch.accelerator.synchronize()
        for e in range(E):
            d = rows_that_differ(ref[e, :m], C[e, :m])
            if d.numel():
                bad[(m, e)] = d[:8].tolist()
    assert not bad, (
        f"{scheme}: rows moved when only the token count changed -- "
        f"{ {k: v for k, v in list(bad.items())[:6]} }"
    )


@skip_if_not_cuda_alike
@_fp8_skip
@pytest.mark.parametrize("scheme", list(_FP8_SCHEMES))
@torch.inference_mode()
def test_batched_expert_fp8_gemm_is_row_permutation_invariant(mode_on, scheme):
    """Row order inside an expert's buffer is nondeterministic; ignore it.

    As for the unquantized arm, except that the activation scale is indexed by
    row too and has to travel with it -- which is the point: a per-token or
    per-(token, k-tile) scale read at a stale row would show up here and
    nowhere else in this file.
    """
    device = torch.device(DEVICE_TYPE)
    m = 200
    ops = _fp8_operands(scheme)
    _assert_fp8_reorder_is_detectable(scheme, ops)
    A, B, a_scale, b_scale = ops
    g = torch.Generator(device=device).manual_seed(99)
    ent = _counts(m)

    ref = _zeros(torch.bfloat16)
    _launch_fp8(scheme, ops, ent, ref)

    ar = torch.arange(m, device=device)
    Ap = A.clone()
    a_scale_p = a_scale.clone()
    pis = []
    for e in range(E):
        pi = torch.randperm(m, generator=g, device=device)
        for _ in range(64):
            if bool((pi != ar).all()):
                break
            pi = torch.randperm(m, generator=g, device=device)
        pis.append(pi)
        Ap[e, :m] = A[e, pi]
        if scheme != "tensor":  # a per-tensor scale has no row to permute
            a_scale_p[e, :m] = a_scale[e, pi]
    perm = _zeros(torch.bfloat16)
    _launch_fp8(scheme, (Ap, B, a_scale_p, b_scale), ent, perm)
    torch.accelerator.synchronize()

    moved = sum(int(((p // 64) != (ar // 64)).sum()) for p in pis) / (E * m)
    assert moved > 0.5, (
        f"{scheme}: only {moved:.1%} of rows changed tile, so a pass would mean nothing"
    )

    bad = {}
    for e in range(E):
        inv = torch.empty_like(pis[e])
        inv[pis[e]] = ar
        un = perm[e, :m][inv]
        ne = bits(ref[e, :m]) != bits(un)
        d = torch.nonzero(ne.reshape(m, -1).any(dim=1)).flatten()
        if d.numel():
            bad[e] = d[:8].tolist()
        # Positive control: without the un-permute the comparison must fail.
        assert rows_that_differ(ref[e, :m], perm[e, :m]).numel() > 0
    assert not bad, (
        f"{scheme}: {len(bad)} experts changed bits when their rows were "
        f"permuted (first offenders: {list(bad.items())[:4]})"
    )


# --------------------------------------------------------------------------- #
# The whole class, through the reference no-comms `BatchedPrepareAndFinalize`.
#
# Everything above holds the quantized operands fixed and moves the batch
# around them. This half moves the batch and lets the class quantize, which is
# where the activation scales are actually decided: `prepare` quantizes a1 per
# expert and `batched_moe_kernel_quantize_input` quantizes a2 from the
# intermediate. A dynamic per-tensor scale on either is an amax over whatever
# rows arrived, so `_CLASS_SCHEMES` includes that scheme by name -- under the
# mode `maybe_promote_act_quant_for_batch_invariance` turns it into a per-token
# scale, and `..._moves_without_the_mode` below is the control that says the
# sweep would have caught it if it had not.
# --------------------------------------------------------------------------- #

C_K = 512  # hidden dim
C_N = 512  # intermediate size
C_TOPK = 2
C_MAXT = 256  # the dispatch buffer, a deployment constant -- never the batch
C_BATCHES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128, 170, 220, 256]
C_DROPS = [1, 3, 7, 20, 64, 100]

# Named by the advertised pair, not by the kernel arithmetic: `tensor_dynamic`
# and `tensor_w_token_a` reach the same kernel branch, and that they do is
# exactly what the promotion is for.
_CLASS_SCHEMES = [
    "block",
    "channel",
    "tensor_w_token_a",
    "tensor_static",
    "tensor_dynamic",
]


def _quantize_weights(scheme, w1, w2):
    fp8 = current_platform.fp8_dtype()
    fmax = torch.finfo(fp8).max

    def per_tensor(w):
        s = (w.abs().amax(dim=(1, 2)).float() / fmax).clamp(min=1e-12)
        return (w.float() / s.view(-1, 1, 1)).clamp(-fmax, fmax).to(fp8), s

    def per_channel(w):
        s = (w.abs().amax(dim=2, keepdim=True).float() / fmax).clamp(min=1e-12)
        return (w.float() / s).clamp(-fmax, fmax).to(fp8), s.contiguous()

    def per_block(w):
        e, r, c = w.shape
        v = w.float().view(e, r // GROUP, GROUP, c // GROUP, GROUP)
        s = (v.abs().amax(dim=(2, 4), keepdim=True) / fmax).clamp(min=1e-12)
        q = (v / s).clamp(-fmax, fmax).to(fp8).view(e, r, c)
        return q, s.reshape(e, r // GROUP, c // GROUP).contiguous()

    f = {"block": per_block, "channel": per_channel}.get(scheme, per_tensor)
    return (*f(w1), *f(w2))


def _class_quant_config(scheme, w1_scale, w2_scale, a_scale):
    if scheme == "block":
        return fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale, w2_scale=w2_scale, block_shape=[GROUP, GROUP]
        )
    if scheme == "channel":
        return fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            per_act_token_quant=True,
            per_out_ch_quant=True,
        )
    if scheme == "tensor_w_token_a":
        return fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale, w2_scale=w2_scale, per_act_token_quant=True
        )
    if scheme == "tensor_static":
        return fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a_scale, a2_scale=a_scale
        )
    assert scheme == "tensor_dynamic"
    return fp8_w8a8_moe_quant_config(w1_scale=w1_scale, w2_scale=w2_scale)


def _class_fixture(scheme, seed=7):
    """Weights, a token pool, and the kernel under test."""
    device = torch.device(f"{DEVICE_TYPE}:0")
    g = torch.Generator(device=device).manual_seed(seed)
    w1 = (
        torch.randn(E, 2 * C_N, C_K, generator=g, device=device, dtype=torch.bfloat16)
        / 8
    )
    w2 = torch.randn(E, C_K, C_N, generator=g, device=device, dtype=torch.bfloat16) / 8
    # Spread the per-expert magnitudes: a per-expert scale that was misshaped
    # or misindexed would send every expert to scale[0], and only a spread
    # makes that visible in the output rather than in the third decimal.
    for e in range(E):
        w1[e] *= 2.0 ** (e - 4)
        w2[e] *= 2.0 ** (4 - e)
    w1q, w1s, w2q, w2s = _quantize_weights(scheme, w1, w2)

    pool = torch.randn(
        max(C_BATCHES), C_K, generator=g, device=device, dtype=torch.float32
    )
    ex = torch.randint(-6, 7, (max(C_BATCHES), 1), generator=g, device=device)
    pool = ((pool * torch.exp2(ex.float())) / 4).to(torch.bfloat16)
    a_scale = (
        pool.abs().max().float() / torch.finfo(current_platform.fp8_dtype()).max
    ).reshape(1)

    cg = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.stack(
        [torch.randperm(E, generator=cg)[:C_TOPK] for _ in range(max(C_BATCHES))]
    ).to(device)
    weights = torch.full(
        (max(C_BATCHES), C_TOPK), 0.5, device=device, dtype=torch.float32
    )

    moe_config = FusedMoEConfig(
        num_experts=E,
        experts_per_token=C_TOPK,
        hidden_dim=C_K,
        intermediate_size=C_N,
        num_local_experts=E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=device.type,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=C_MAXT,
    )
    kernel = FusedMoEKernel(
        BatchedPrepareAndFinalize(
            C_MAXT, num_local_experts=E, num_dispatchers=1, rank=0
        ),
        BatchedTritonExperts(
            moe_config=moe_config,
            quant_config=_class_quant_config(scheme, w1s, w2s, a_scale),
            max_num_tokens=C_MAXT,
            num_dispatchers=1,
        ),
    )

    def run(lo, hi=None):
        sl = slice(lo, hi)
        return kernel.apply(
            pool[sl],
            w1q,
            w2q,
            weights[sl],
            ids[sl],
            activation=MoEActivation.SILU,
            global_num_experts=E,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    return kernel.fused_experts, run, ids


def _slots(ids, lo, hi):
    """The slot `prepare` gives each token inside each expert's buffer."""
    out = {}
    for e in range(E):
        hits = (ids[lo:hi] == e).any(dim=1)
        for slot, t in enumerate(torch.nonzero(hits).flatten().tolist()):
            out[(e, t + lo)] = slot
    return out


def _class_sweep(run, ids):
    """Returns (rows moved, rows checked, slot relocations)."""
    full = max(C_BATCHES)
    ref = run(0, full)
    torch.accelerator.synchronize()
    base = _slots(ids, 0, full)

    moved = checked = relocations = 0
    offenders: list[tuple[str, int, list[int]]] = []
    for m in C_BATCHES:  # append arm
        out = run(0, m)
        torch.accelerator.synchronize()
        d = (bits(ref[:m]) != bits(out)).any(dim=1)
        checked += m
        moved += int(d.sum())
        if int(d.sum()) and len(offenders) < 6:
            offenders.append(("append", m, torch.nonzero(d).flatten()[:5].tolist()))
    for j in C_DROPS:  # drop-from-front arm; this is what relocates slots
        out = run(j, full)
        torch.accelerator.synchronize()
        here = _slots(ids, j, full)
        relocations += sum(1 for k, v in here.items() if base[k] != v)
        d = (bits(ref[j:]) != bits(out)).any(dim=1)
        checked += full - j
        moved += int(d.sum())
        if int(d.sum()) and len(offenders) < 6:
            offenders.append(("drop", j, torch.nonzero(d).flatten()[:5].tolist()))
        # Positive control: the comparison is only meaningful because it lines
        # the rows up by token. Compared without the shift it must disagree.
        assert not torch.equal(bits(ref[: full - j]), bits(out)), (
            "dropping tokens from the front changed nothing at all, so this "
            "arm is not exercising the slot assignment it claims to"
        )
    return moved, checked, relocations, offenders


@skip_if_not_cuda_alike
@_fp8_skip
@pytest.mark.parametrize("scheme", _CLASS_SCHEMES)
@pytest.mark.parametrize("capturing", [False, True])
@torch.inference_mode()
def test_batched_experts_fp8_class_is_batch_invariant(
    mode_on, workspace_init, monkeypatch, scheme, capturing
):
    """`BatchedTritonExperts.apply` end to end, batch composition moving.

    Both branches of `batched_moe_kernel_quantize_input` are covered.
    `capturing=True` forces the one taken under cudagraphs and torch.compile,
    which quantizes the whole `E x max_num_tokens` buffer in one call instead
    of looping over the live rows of each expert -- a different amax domain,
    and the branch that actually runs in production behind DeepEP LL.
    """
    from vllm.model_executor.layers.fused_moe.experts import fused_batched_moe

    if capturing:
        monkeypatch.setattr(
            fused_batched_moe, "_is_capturing_or_compiling", lambda: True
        )

    experts, run, ids = _class_fixture(scheme)
    # `tensor_dynamic` is here because it was promoted; the rest are here as
    # constructed. Asserting the granularity is what stops a scheme from
    # silently collapsing into one already covered.
    per_token = {"channel", "tensor_w_token_a", "tensor_dynamic"}
    assert experts.per_act_token_quant == (scheme in per_token), (
        f"{scheme}: unexpected activation granularity "
        f"{experts.per_act_token_quant} under the mode"
    )
    assert (experts.block_shape is not None) == (scheme == "block")

    seen = {_digest(run(0, max(C_BATCHES))) for _ in range(4)}
    assert len(seen) == 1, f"{scheme}: {len(seen)} distinct results over 4 calls"

    moved, checked, relocations, offenders = _class_sweep(run, ids)
    assert relocations > 1000, (
        f"{scheme}: only {relocations} tokens changed slot, so the sweep is "
        "not exercising the buffer packing"
    )
    assert moved == 0, (
        f"{scheme}: {moved} of {checked} rows moved with the batch "
        f"({relocations} slot relocations); first offenders {offenders}"
    )


@skip_if_not_cuda_alike
@_fp8_skip
@torch.inference_mode()
def test_batched_experts_fp8_dynamic_per_tensor_moves_without_the_mode(
    workspace_init, monkeypatch
):
    """The sweep above has detection power; here is the proof.

    `tensor_dynamic` is the one advertised pair whose activation scale is an
    amax over the batch. With the mode off nothing promotes it and the same
    fixture moves hundreds of rows. Without this control, a sweep that passed
    because the whole harness had gone inert would read identically.
    """
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    experts, run, ids = _class_fixture("tensor_dynamic")
    assert not experts.per_act_token_quant

    moved, checked, relocations, _ = _class_sweep(run, ids)
    assert moved > 100, (
        f"only {moved} of {checked} rows moved without the mode, so the "
        "invariant version of this sweep is not measuring much"
    )


# --------------------------------------------------------------------------- #
@skip_if_not_cuda_alike
@_fp8_skip
@pytest.mark.parametrize("w8a16", [False, True])
@torch.inference_mode()
def test_batched_moe_weight_scale_is_read_as_a_broadcast(mode_on, w8a16):
    """A per-tensor weight scale must not be read past its end.

    `moe_mmk` loads the weight scale as `b_scale[e * stride_bse + offs_bn *
    stride_bsn]` with no mask and `offs_bn` spanning N, on both the
    per-act-token w8a8 path and the w8a16 path. A per-tensor scale is `[E]`,
    reshaped to `[E, 1, 1]` by the launcher, and a size-1 dimension must get
    stride 0 -- with the contiguous stride of 1 that read runs up to N elements
    past the tensor and picks up whatever the allocator left there, which moves
    between calls and is a correctness bug as well as a reproducibility one.

    So: put the scale at the front of a larger buffer, write two different
    poison patterns behind it, and require the output not to notice. The
    positive control drives `batched_triton_kernel` directly with the strides
    the launcher used to compute, and requires that it *does* notice -- without
    it this test would pass on any kernel that ignored the scale entirely.

    w8a16 is unreachable through `BatchedTritonExperts` (its constructor
    asserts NYI) and is covered here only because it is the same load with the
    same stride, one `tl.constexpr` away.
    """
    device = torch.device(DEVICE_TYPE)
    fp8 = current_platform.fp8_dtype()
    g = torch.Generator(device=device).manual_seed(5)
    if w8a16:
        A = (torch.randn((E, T, K), generator=g, device=device) / 4).to(torch.bfloat16)
        B = torch.randint(
            -127, 127, (E, N, K), generator=g, device=device, dtype=torch.int8
        )
        a_scale = None
    else:
        fmax = torch.finfo(fp8).max
        a = _fp8_spread_normal((E, T, K), g, _FP8_A_SPREAD)
        s = (a.abs().amax(dim=-1, keepdim=True) / fmax).clamp(min=1e-12)
        A = (a / s).clamp(-fmax, fmax).to(fp8)
        a_scale = s.contiguous()
        b = _fp8_spread_normal((E, N, K), g, _FP8_B_SPREAD) / (K**0.5)
        B = (b / (b.abs().amax() / fmax)).clamp(-fmax, fmax).to(fp8)

    # E scales followed by room for the out-of-bounds read the fix removed, so
    # the pre-fix behaviour is reproducible here without touching invalid
    # memory: the poison is a real allocation, just not this tensor's.
    buf = torch.empty(E + N + 64, device=device, dtype=torch.float32)
    b_scale = buf[:E]
    b_scale.copy_(torch.linspace(0.5, 1.5, E, device=device))
    ent = _counts(200)

    def with_poison(value):
        buf[E:] = value
        C = _zeros(torch.bfloat16)
        invoke_moe_batched_triton_kernel(
            A=A,
            B=B,
            C=C,
            expert_num_tokens=ent,
            compute_type=tl.bfloat16,
            A_scale=a_scale,
            B_scale=b_scale,
            B_zp=None,
            use_fp8_w8a8=not w8a16,
            use_int8_w8a16=w8a16,
            use_int4_w4a16=False,
            config={"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            per_act_token_quant=not w8a16,
            block_shape=None,
        )
        torch.accelerator.synchronize()
        return _digest(C)

    def with_poison_unfixed(value):
        """The launcher's pre-fix strides: `[E, 1, 1]` contiguous, all 1."""
        buf[E:] = value
        C = _zeros(torch.bfloat16)
        bs = b_scale.view(-1, 1, 1)
        asc = a_scale
        batched_triton_kernel[(E, triton.cdiv(T, 64) * triton.cdiv(N, 64))](
            A,
            B,
            C,
            ent,
            tl.bfloat16,
            T,
            K,
            N,
            asc,
            bs,
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
            0 if asc is None else asc.stride(0),
            0 if asc is None else asc.stride(1),
            0 if asc is None else asc.stride(2),
            bs.stride(0),
            bs.stride(2),
            bs.stride(1),
            0,
            0,
            not w8a16,
            w8a16,
            not w8a16,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            USE_TD=False,
        )
        torch.accelerator.synchronize()
        return _digest(C)

    assert with_poison_unfixed(7.0) != with_poison_unfixed(-3.0), (
        "the positive control did not fire: with the pre-fix strides the "
        "kernel must read the poison, otherwise this test cannot tell a "
        "correct broadcast from a scale that is never read"
    )
    assert with_poison(7.0) == with_poison(-3.0), (
        "the output changed when memory *after* the weight scale changed, so "
        "the launcher is reading past an [E] scale again"
    )


@skip_if_not_cuda_alike
@_fp8_skip
@pytest.mark.parametrize("requested_bk", [16, 32, 64, 128, 256])
@torch.inference_mode()
def test_batched_moe_block_quant_clamps_block_k_to_the_group(mode_on, requested_bk):
    """A K tile wider than the quantization group must be clamped, not obeyed.

    `moe_mmk` loads the block scale once per K tile, at `k_start // group_k`,
    and applies it to the whole tile. With `group_k=128` a BLOCK_K of 256
    therefore charges the first group's scale to both groups it spans, which
    is wrong arithmetic and not a rounding difference: 25% median relative
    error against a dequantized reference here, against 0.14% for every tile
    that fits inside a group.

    This is selectable, not hypothetical. `try_get_optimal_moe_config` keys
    the tuning tables on the *local* expert count and the unsharded
    intermediate size, which is what an EP deployment produces, and
    `E=32,N=2048` -- DeepSeek-V3/R1 block-fp8 at EP=8 -- has `BLOCK_SIZE_K:
    256` at M=256 on B200 and at M=2048 on H200. Nothing under
    `VLLM_BATCH_INVARIANT` reaches it, because `get_default_config`
    short-circuits to 64/64/32, which is why this is a correctness guard and
    not a batch-invariance one: BLOCK_K is chosen from `max_num_tokens`, a
    deployment constant, so it cannot vary with the batch either way.
    """
    ops = _fp8_operands("block")
    A, B, a_scale, b_scale = ops
    got = _zeros(torch.bfloat16)
    _launch_fp8("block", ops, _counts(T), got, 64, 64, requested_bk)
    torch.accelerator.synchronize()

    e, rows = 1, list(range(8))
    asc = a_scale[e, rows].repeat_interleave(GROUP, dim=-1)
    bsc = b_scale[e].repeat_interleave(GROUP, dim=0).repeat_interleave(GROUP, dim=1)
    dequant = (A[e, rows].float() * asc) @ (B[e].float() * bsc).T
    err = (
        (got[e, rows].float() - dequant).abs() / dequant.abs().clamp(min=1e-6)
    ).median()
    assert err < 0.01, (
        f"BLOCK_SIZE_K={requested_bk} with block_shape=[{GROUP}, {GROUP}] gave "
        f"{err:.3f} median relative error against a dequantized reference; the "
        "launcher is applying one k-group's scale to more than one group"
    )
    if requested_bk >= GROUP:
        # Clamped to the group, so it must land on the group-sized tile's bits.
        clamped = _zeros(torch.bfloat16)
        _launch_fp8("block", ops, _counts(T), clamped, 64, 64, GROUP)
        torch.accelerator.synchronize()
        assert torch.equal(bits(got), bits(clamped))
