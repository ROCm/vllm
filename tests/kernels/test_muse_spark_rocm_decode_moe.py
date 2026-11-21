# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm decode MoE kernels."""

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.models.muse_spark.amd.muse_moe_runner import MuseSparkAiterExperts
from vllm.models.muse_spark.amd.ops import decode_moe as dmoe
from vllm.models.muse_spark.amd.ops.compact_moe import MAX_TOKENS as COMPACT_MAX_TOKENS
from vllm.platforms import current_platform

from .moe.test_muse_spark_rocm_bf16_moe import (
    _moe_config,
    _mxfp4_weights,
    _parallel_config,
    _route_launcher,
)

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 4096
ROUTE_SCALE = 2.5
DIVISION_EPS = 1e-15
NORM_EPS = 1e-5


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires a ROCm GPU")
@pytest.mark.parametrize("num_tokens", [1, 7, 16, 48, 128, 513, 2047])
@pytest.mark.parametrize("routing", ["random", "ties"])
@pytest.mark.parametrize("split_k_partials", [True, False])
@torch.inference_mode()
def test_decode_moe_prep_matches_the_unfused_kernels(
    num_tokens: int, routing: str, split_k_partials: bool
) -> None:
    """One prep launch, from the router's split-K partials or its logits, picks
    the experts of the router split reduce + routing kernel of the per-route
    path, with its weights, and the rows of its pre-expert norm kernel."""
    from vllm.models.muse_spark.amd.ops.fused_norm_expand import fused_rmsnorm_expand
    from vllm.models.muse_spark.amd.ops.router_bf16x3 import (
        router_bf16x3_gemm,
        split_fp32_to_bf16x3,
    )
    from vllm.models.muse_spark.common.ops.fused_routing import (
        fused_muse_spark_routing,
    )

    torch.manual_seed(num_tokens)
    device = torch.device("cuda")
    router_weight = torch.randn(NUM_EXPERTS, 8192, device=device) * 0.02
    planes = split_fp32_to_bf16x3(router_weight)
    x = torch.randn(num_tokens, 8192, device=device, dtype=torch.bfloat16)
    pooled = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    norm_weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(NUM_EXPERTS, device=device) * 0.01
    if routing == "ties":
        x.zero_()
        bias = torch.randint(0, 4, (NUM_EXPERTS,), device=device).float() * 0.01

    weights, ids, normed = dmoe.decode_moe_prep(
        router_bf16x3_gemm(x, planes, reduce_splits=not split_k_partials),
        bias,
        TOP_K,
        ROUTE_SCALE,
        None,
        DIVISION_EPS,
        pooled,
        norm_weight,
        NORM_EPS,
        1.0,
        False,
    )
    ref_weights, ref_ids = fused_muse_spark_routing(
        router_bf16x3_gemm(x, planes),
        bias,
        TOP_K,
        ROUTE_SCALE,
        router_output_multiplier=None,
        router_score_normalization="sigmoid",
        router_score_division_eps=DIVISION_EPS,
    )
    ref_normed = fused_rmsnorm_expand(pooled, norm_weight, NORM_EPS, 1.0, False, TOP_K)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights)
    torch.testing.assert_close(normed, ref_normed)


ROCM_GFX950_AITER = False
if current_platform.is_rocm():
    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950_AITER = on_gfx950() and is_aiter_found_and_supported()


def _per_route_and_native(
    pooled: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_experts: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EP rank 1 with `local_experts` local experts at Muse Spark shapes: the
    per-route AITER experts on the repeated pre-expert-normed rows with the
    per-route finalize, and the decode MoE's experts on the T normed rows with its
    finalize. Returns `(per_route, native)` `[T, H]` outputs."""
    from vllm.models.muse_spark.amd import decode_moe
    from vllm.models.muse_spark.amd.ops.fused_norm_expand import fused_rmsnorm_expand
    from vllm.models.muse_spark.common.layers import MuseSparkRMSNormWithInputScale

    num_tokens, hidden_size = pooled.shape
    top_k = topk_ids.shape[1]
    device, dtype = pooled.device, pooled.dtype
    ep_size = NUM_EXPERTS // local_experts
    kernels = decode_moe._stage_kernels(
        num_tokens, top_k, local_experts, hidden_size, hidden_size
    )
    if kernels is None:
        pytest.skip("the installed AITER has no FlyDSL stages for this shape")
    local, expert_map, expert_mask = determine_expert_map(
        ep_size=ep_size,
        ep_rank=1,
        global_num_experts=NUM_EXPERTS,
        return_expert_mask=True,
    )
    assert expert_map is not None and expert_mask is not None
    expert_map, expert_mask = expert_map.to(device), expert_mask.to(device)
    w13, w2, quant_config = _mxfp4_weights(
        torch.randn(local, 2 * hidden_size, hidden_size, dtype=dtype, device=device)
        / hidden_size**0.5,
        torch.randn(local, hidden_size, hidden_size, dtype=dtype, device=device)
        / hidden_size**0.5,
    )
    post_norm = MuseSparkRMSNormWithInputScale(
        hidden_size=hidden_size,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
        output_norm_gain_center_type="zero",
    ).to(device)
    post_norm.weight.copy_(torch.randn(hidden_size, device=device) * 0.1)
    quant_config.post_expert_norm_fn = post_norm.forward
    config = _moe_config(
        parallel_config=_parallel_config(ep_size=ep_size),
        num_experts=NUM_EXPERTS,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
    )
    experts = MuseSparkAiterExperts(config, quant_config)
    norm_weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1

    route_buffer, launch = _route_launcher(
        experts,
        fused_rmsnorm_expand(pooled, norm_weight, NORM_EPS, 1.0, False, top_k),
        w13,
        w2,
        topk_weights,
        topk_ids,
        intermediate_size=hidden_size,
        num_experts=NUM_EXPERTS,
        num_local_experts=local,
        expert_map=expert_mask,
    )
    launch()
    per_route = experts.finalize_weight_and_reduce_impl().apply(
        torch.empty_like(pooled), route_buffer, topk_weights, topk_ids, False
    )

    native = decode_moe.native_routes_moe(
        fused_rmsnorm_expand(pooled, norm_weight, NORM_EPS, 1.0, False, 1),
        topk_weights,
        topk_ids,
        w13,
        w2,
        quant_config,
        expert_mask,
        expert_map,
        kernels,
        post_norm,
    )
    return per_route, native


@pytest.fixture
def rocm_aiter_moe(monkeypatch: pytest.MonkeyPatch):
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize("local_experts", [32, 64])
@pytest.mark.parametrize(
    "num_tokens", [1, 8, 48, 64, 128, 256, 512, 513, 1024, 2047, 2048]
)
@torch.inference_mode()
def test_native_routes_match_the_per_route_path(
    rocm_aiter_moe, local_experts: int, num_tokens: int
) -> None:
    """The decode MoE's experts and finalize give the per-route path's output at
    the EP4 and EP8 shapes, for every call size the decode MoE takes."""
    torch.manual_seed(num_tokens)
    device = torch.device("cuda")
    pooled = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    topk_ids = torch.topk(
        torch.randn(num_tokens, NUM_EXPERTS, device=device), TOP_K, dim=-1
    ).indices.to(torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, TOP_K, device=device), -1)
    per_route, native = _per_route_and_native(
        pooled, topk_ids, topk_weights, local_experts
    )
    assert torch.isfinite(native).all()
    # The per-route path may take other AITER kernels for its repeated rows,
    # e.g. a fused-quant stage 1 whose MXFP4 requantization rounds blocks
    # differently; wrong routes or weights would be off by far more.
    error = (native.float() - per_route.float()).norm()
    assert error <= 5e-2 * per_route.float().norm()


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize("num_tokens", [2, 37, 64, 256, 512])
@torch.inference_mode()
def test_native_routes_with_repeated_experts_and_nan_rows(
    rocm_aiter_moe, num_tokens: int
) -> None:
    """Tokens selecting one expert in several slots and a NaN row, as CUDA-graph
    padding rows with non-finite router logits produce: every other token's
    output is the per-route path's."""
    torch.manual_seed(num_tokens)
    device = torch.device("cuda")
    pooled = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    pooled[-1] = float("nan")
    topk_ids = torch.topk(
        torch.randn(num_tokens, NUM_EXPERTS, device=device), TOP_K, dim=-1
    ).indices.to(torch.int32)
    topk_ids[0] = torch.arange(64, 64 + TOP_K)
    topk_ids[3::3] = topk_ids[3::3, :1]
    topk_ids[1::3, 5] = topk_ids[1::3, 2]
    topk_weights = torch.softmax(torch.randn(num_tokens, TOP_K, device=device), -1)
    per_route, native = _per_route_and_native(pooled, topk_ids, topk_weights)
    slots = torch.arange(TOP_K, device=device)
    repeats = (
        (topk_ids[:, None, :] == topk_ids[:, :, None])
        & (slots[None, :] != slots[:, None])
    ).any(dim=(1, 2))
    compared = ~repeats & pooled.isfinite().all(dim=1)
    assert compared[0]
    torch.testing.assert_close(native[compared], per_route[compared])


# The compact experts and the FlyDSL stages share the quant rules and the output
# rounding; only fp32 accumulation order and SiLU ulps may differ, and a single
# requantized block that flips stays well below this.
COMPACT_REL_L2 = 1e-3
COMPACT_TOKENS = [1, 8, 24, 64, 65, 192, COMPACT_MAX_TOKENS]
REPLICA_TOKENS = [8, 64, COMPACT_MAX_TOKENS]


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize("local_experts", [32, 64, 128])
@pytest.mark.parametrize("num_tokens", COMPACT_TOKENS)
@torch.inference_mode()
def test_compact_experts_match_the_flydsl_stages(
    rocm_aiter_moe, monkeypatch: pytest.MonkeyPatch, local_experts: int, num_tokens: int
) -> None:
    """The compact experts give the FlyDSL stages' per-slot outputs on every
    local route."""
    from vllm.models.muse_spark.amd import decode_moe
    from vllm.models.muse_spark.amd.ops.compact_moe import supports_compact_moe

    torch.manual_seed(num_tokens)
    device = torch.device("cuda")
    local, expert_map, expert_mask = determine_expert_map(
        ep_size=NUM_EXPERTS // local_experts,
        ep_rank=1,
        global_num_experts=NUM_EXPERTS,
        return_expert_mask=True,
    )
    assert expert_map is not None and expert_mask is not None
    expert_map, expert_mask = expert_map.to(device), expert_mask.to(device)
    w13, w2, quant_config = _mxfp4_weights(
        torch.randn(local, 2 * HIDDEN, HIDDEN, dtype=torch.bfloat16, device=device)
        / HIDDEN**0.5,
        torch.randn(local, HIDDEN, HIDDEN, dtype=torch.bfloat16, device=device)
        / HIDDEN**0.5,
    )
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    topk_ids = torch.topk(
        torch.randn(num_tokens, NUM_EXPERTS, device=device), TOP_K, dim=-1
    ).indices.to(torch.int32)
    w1_scale, w2_scale = quant_config.w1_scale, quant_config.w2_scale
    assert supports_compact_moe(x, topk_ids, w13, w2, w1_scale, w2_scale)
    kernels = decode_moe._stage_kernels(num_tokens, TOP_K, local, HIDDEN, HIDDEN)
    assert kernels is not None

    def experts() -> torch.Tensor:
        return decode_moe._native_routes_experts(
            x, topk_ids, w13, w2, w1_scale, w2_scale, expert_mask, expert_map, kernels
        )[0]

    routes = expert_map[topk_ids.long()] >= 0

    def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a[routes].float(), b[routes].float()
        assert torch.isfinite(a).all()
        return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()

    compact = experts()
    monkeypatch.setattr(decode_moe, "supports_compact_moe", lambda *args: False)
    flydsl = experts()
    assert rel_l2(compact, flydsl) < COMPACT_REL_L2


# Expert-parallel layouts of the decode expert replicas: (ranks, copies per
# rank).
REPLICA_LAYOUTS = [(4, 4), (4, 8), (8, 8)]
REPLICA_INTER = 1024


def _replica_holders(
    step_indices: list[np.ndarray],
    plan: list[tuple[int, int, int]],
    local: int,
) -> np.ndarray:
    """The rank computing each expert's routes in one step, from the ranks'
    `[E]` step indices after the call. Asserts
    that each expert is computed by exactly one rank, either its owner in its
    own slot or a planned copy rank in the planned replica slot."""
    copies = {(e, c): local + s for e, c, s in plan}
    held = np.stack(step_indices) >= 0
    assert (held.sum(0) == 1).all(), "an expert has no holder or several"
    holders = held.argmax(0)
    for e, r in enumerate(holders):
        idx = step_indices[r][e]
        if r == e // local:
            assert idx == e - r * local, (e, r, idx)
        else:
            assert copies.get((e, int(r))) == idx, (e, r, idx)
    return holders


def _busiest_rank_cost(
    counts: np.ndarray, holders: np.ndarray, ranks: int, block_m: int
) -> int:
    """The largest per-rank cost of a step: expert weight reads, by blocks of
    routes, as the replica assignment counts them."""
    from vllm.models.muse_spark.amd.ops import compact_moe as cm

    blocks = (counts + block_m - 1) // block_m
    cost = np.where(
        counts > 0,
        cm.REPLICA_COST_UNIT + cm.REPLICA_COST_EXTRA_BLOCK * (blocks - 1),
        0,
    )
    return int(np.bincount(holders, weights=cost, minlength=ranks).max())


def _replica_rank_inputs(
    plan: list[tuple[int, int, int]], rank: int, ranks: int, replicas: int, device
):
    """One rank's replica table, its step index at rest and the global experts
    of its replica slots (0 for unused slots)."""
    from vllm.models.muse_spark.amd import decode_replicas as dr

    local = NUM_EXPERTS // ranks
    table = dr.layer_table(plan, rank, ranks, NUM_EXPERTS, replicas, device)
    index = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
    index[rank * local : (rank + 1) * local] = torch.arange(
        local, dtype=torch.int32, device=device
    )
    copies = [e for _, e in sorted((s, e) for e, c, s in plan if c == rank)]
    copies += [0] * (replicas - len(copies))
    return table, index, torch.tensor(copies, dtype=torch.long, device=device)


def _random_mxfp4_experts(device: torch.device, inter: int):
    """Shuffled-layout MXFP4 weights and E8M0 scales of every expert (random
    codes, scales 2^-6..2^-3 so that outputs stay well inside bf16)."""
    gen = torch.Generator(device=device).manual_seed(0)

    def rand(shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, dtype=torch.uint8, device=device, generator=gen
        )

    w13 = rand((NUM_EXPERTS, 2 * inter, HIDDEN // 2))
    w2 = rand((NUM_EXPERTS, HIDDEN, inter // 2))
    s13 = rand((NUM_EXPERTS, 2 * inter, HIDDEN // 32), 121, 125)
    s2 = rand((NUM_EXPERTS, HIDDEN, inter // 32), 121, 125)
    return w13, w2, s13, s2


def _shuffled(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    for t in tensors:
        t.is_shuffled = True
    return tensors


def _expert_load(ranks: int) -> np.ndarray:
    return np.random.default_rng(ranks).pareto(1.0, NUM_EXPERTS) + 0.01


def _routes(num_tokens: int, load: np.ndarray, skewed: bool, device) -> torch.Tensor:
    logits = torch.randn(num_tokens, NUM_EXPERTS, device=device)
    if skewed:
        logits += 3.0 * torch.from_numpy(np.log(load)).float().to(device)
    return torch.topk(logits, TOP_K, dim=-1).indices.to(torch.int32)


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize("ranks,replicas", REPLICA_LAYOUTS)
@pytest.mark.parametrize("num_tokens", REPLICA_TOKENS)
@pytest.mark.parametrize("routing", ["random", "skewed"])
@torch.inference_mode()
def test_compact_experts_with_replicas_compute_every_route_once(
    ranks: int, replicas: int, num_tokens: int, routing: str
) -> None:
    """Over two steps with the same step indices: every expert's routes are
    computed by exactly one rank (its owner, or a planned copy in the planned
    slot), the busiest rank's cost is no higher than with owners only, each
    computed route's output row is the owner's, and skewed routing moves some
    experts to their copies."""
    from vllm.models.muse_spark.amd import decode_replicas as dr
    from vllm.models.muse_spark.amd.ops import compact_moe as cm

    torch.manual_seed(num_tokens + ranks)
    device = torch.device("cuda")
    local = NUM_EXPERTS // ranks
    w13, w2, s13, s2 = _random_mxfp4_experts(device, REPLICA_INTER)
    load = _expert_load(ranks)
    plan = dr.plan_layer(load, ranks, replicas)
    owner = np.arange(NUM_EXPERTS) // local
    inputs = [
        _replica_rank_inputs(plan, r, ranks, replicas, device) for r in range(ranks)
    ]
    for step in range(2):
        topk_ids = _routes(num_tokens, load, routing == "skewed" and step == 0, device)
        x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
        counts = np.bincount(topk_ids.cpu().numpy().ravel(), minlength=NUM_EXPERTS)
        owners_out = []
        for r in range(ranks):
            own = torch.arange(r * local, (r + 1) * local, device=device)
            index = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
            index[own] = torch.arange(local, dtype=torch.int32, device=device)
            weights = _shuffled(w13[own], w2[own], s13[own], s2[own])
            owners_out.append(cm.compact_moe_experts(x, topk_ids, index, *weights))
        computed = torch.zeros(num_tokens, TOP_K, dtype=torch.int32, device=device)
        for r, (table, step_index, rows) in enumerate(inputs):
            own = torch.arange(r * local, (r + 1) * local, device=device)
            w13_r, w2_r, s13_r, s2_r = _shuffled(w13[own], w2[own], s13[own], s2[own])
            out = cm.compact_moe_experts(
                x,
                topk_ids,
                step_index,
                w13_r,
                w2_r,
                s13_r,
                s2_r,
                replicas=table,
                replica_weights=(w13[rows], w2[rows], s13[rows], s2[rows]),
            )
            mine = step_index[topk_ids.long()] >= 0
            computed += mine.int()
            for o in range(ranks):
                sel = mine & (topk_ids // local == o)
                if sel.any():
                    torch.testing.assert_close(out[sel], owners_out[o][sel])
        assert bool((computed == 1).all())
        step_indices = [i.cpu().numpy() for _, i, _ in inputs]
        holders = _replica_holders(step_indices, plan, local)
        block_m = cm.tile_config(num_tokens).block_m
        assert _busiest_rank_cost(counts, holders, ranks, block_m) <= (
            _busiest_rank_cost(counts, owner, ranks, block_m)
        )
        if routing == "skewed" and step == 0 and num_tokens >= 64:
            assert (holders != owner)[counts > 0].any(), "no expert moved"


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize("ranks,replicas", REPLICA_LAYOUTS)
@pytest.mark.parametrize("num_tokens,num_padding", [(8, 3), (64, 17), (192, 64)])
@torch.inference_mode()
def test_compact_experts_with_replicas_ignore_padding_rows(
    ranks: int, replicas: int, num_tokens: int, num_padding: int
) -> None:
    """Padding rows with arbitrary routes leave the replica assignment (every
    rank's step index) and the real rows' outputs unchanged."""
    from vllm.models.muse_spark.amd import decode_replicas as dr
    from vllm.models.muse_spark.amd.ops import compact_moe as cm

    torch.manual_seed(num_tokens + ranks)
    device = torch.device("cuda")
    local = NUM_EXPERTS // ranks
    w13, w2, s13, s2 = _random_mxfp4_experts(device, REPLICA_INTER)
    load = _expert_load(ranks)
    plan = dr.plan_layer(load, ranks, replicas)
    real = num_tokens - num_padding
    topk_ids = _routes(num_tokens, load, True, device)
    padding = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    padding[real:] = True
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    # Garbage routes: every padding row on rank 0's experts, then on the
    # hottest replicated experts.
    hot = torch.tensor([e for e, _, _ in plan[:TOP_K]], device=device)
    garbage = [
        torch.arange(TOP_K, device=device).expand(num_padding, TOP_K),
        hot[torch.arange(TOP_K, device=device) % len(hot)].expand(num_padding, TOP_K),
    ]
    results, indices = [], []
    for g in garbage:
        ids = topk_ids.clone()
        ids[real:] = g.to(torch.int32)
        per_rank, step_indices = [], []
        for r in range(ranks):
            table, step_index, rows = _replica_rank_inputs(
                plan, r, ranks, replicas, device
            )
            own = torch.arange(r * local, (r + 1) * local, device=device)
            w13_r, w2_r, s13_r, s2_r = _shuffled(w13[own], w2[own], s13[own], s2[own])
            out = cm.compact_moe_experts(
                x,
                ids,
                step_index,
                w13_r,
                w2_r,
                s13_r,
                s2_r,
                replicas=table,
                replica_weights=(w13[rows], w2[rows], s13[rows], s2[rows]),
                padding=padding,
            )
            step_indices.append(step_index.cpu().numpy())
            mine = step_index[ids[:real].long()] >= 0
            per_rank.append(torch.where(mine[..., None], out[:real], 0).float())
        _replica_holders(step_indices, plan, local)
        results.append(torch.stack(per_rank))
        indices.append(np.stack(step_indices))
    np.testing.assert_array_equal(indices[0], indices[1])
    torch.testing.assert_close(results[0], results[1])


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize("ranks,replicas", REPLICA_LAYOUTS)
@pytest.mark.parametrize("num_tokens", [8, 64, COMPACT_MAX_TOKENS])
@torch.inference_mode()
def test_native_routes_with_replicas_match_the_owners(
    ranks: int, replicas: int, num_tokens: int
) -> None:
    """The ranks' finalized outputs with replicas sum to the owners-only sum
    (each rank's partial is rounded to bf16, and replicas change which partial
    carries a route; one missing or repeated route is far above the bound)."""
    from types import SimpleNamespace

    from vllm.models.muse_spark.amd import decode_moe
    from vllm.models.muse_spark.amd import decode_replicas as dr
    from vllm.models.muse_spark.common.layers import MuseSparkRMSNormWithInputScale

    torch.manual_seed(num_tokens)
    device = torch.device("cuda")
    local = NUM_EXPERTS // ranks
    w13, w2, s13, s2 = _random_mxfp4_experts(device, REPLICA_INTER)
    load = _expert_load(ranks)
    plan = dr.plan_layer(load, ranks, replicas)
    post_norm = MuseSparkRMSNormWithInputScale(
        hidden_size=HIDDEN,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
        output_norm_gain_center_type="zero",
    ).to(device)
    post_norm.weight.copy_(torch.randn(HIDDEN, device=device) * 0.1)
    # The compact experts take every call here; the stage kernels are unused.
    kernels = decode_moe.StageKernels("", {}, 32)
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    topk_ids = _routes(num_tokens, load, True, device)
    topk_weights = torch.softmax(torch.randn(num_tokens, TOP_K, device=device), -1)
    owners = torch.zeros(num_tokens, HIDDEN, dtype=torch.float32, device=device)
    with_replicas = torch.zeros_like(owners)
    moved = 0
    for r in range(ranks):
        own = torch.arange(r * local, (r + 1) * local, device=device)
        w13_r, w2_r, s13_r, s2_r = _shuffled(w13[own], w2[own], s13[own], s2[own])
        quant_config = SimpleNamespace(w1_scale=s13_r, w2_scale=s2_r)
        table, step_index, rows = _replica_rank_inputs(plan, r, ranks, replicas, device)
        mask = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
        mask[own] = 1
        args = (x, topk_weights, topk_ids, w13_r, w2_r, quant_config, mask)
        owners += decode_moe.native_routes_moe(
            *args, step_index.clone(), kernels, post_norm
        ).float()
        layer = dr.LayerDecodeReplicas(
            table, step_index, (w13[rows], w2[rows], s13[rows], s2[rows])
        )
        with_replicas += decode_moe.native_routes_moe(
            *args, step_index, kernels, post_norm, replicas=layer
        ).float()
        # Routes this rank computes on a replica slot (not vacuous: the
        # routing is skewed toward the replicated experts).
        moved += int((step_index[topk_ids.long()] >= local).sum())
    assert moved > 0, "no route was computed on a replica"
    assert torch.isfinite(with_replicas).all()
    rel = (with_replicas - owners).norm() / owners.norm().clamp_min(1e-30)
    assert rel < 1e-2, f"relative L2 {rel:.2e}"


def _replica_vllm_config(additional_config: dict, **parallel):
    from types import SimpleNamespace

    parallel = {
        "tensor_parallel_size": 4,
        "data_parallel_size": 1,
        "enable_expert_parallel": True,
        "enable_eplb": False,
        **parallel,
    }
    return SimpleNamespace(
        additional_config=additional_config,
        parallel_config=SimpleNamespace(**parallel),
        kernel_config=SimpleNamespace(monokernel=parallel.pop("monokernel", False)),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=6, num_local_experts=32)
        ),
    )


def _write_routed_experts_responses(path, routed: list[np.ndarray]) -> None:
    import io
    import json

    import pybase64 as base64

    with open(path, "w") as f:
        for ids in routed:
            buf = io.BytesIO()
            np.save(buf, ids)
            encoded = base64.b64encode(buf.getvalue()).decode()
            f.write(json.dumps({"choices": [{"routed_experts": encoded}]}) + "\n")


def test_decode_replicas_from_a_routed_experts_profile(tmp_path) -> None:
    """A profile summed from returned routed experts loads through
    `additional_config` and plans copies of the hottest experts, each on a
    non-owner rank, at most `replicas` per rank."""
    from tools.expert_load_profile import expert_load_profile, routed_experts_of
    from vllm.models.muse_spark.amd import decode_replicas as dr

    rng = np.random.default_rng(0)
    hot = rng.pareto(1.0, 32) + 0.01
    routed = [
        np.stack(
            [
                np.stack([rng.choice(32, 4, replace=False, p=hot / hot.sum())])
                for _ in range(6)
            ],
            axis=1,
        ).repeat(n, axis=0)
        for n in (3, 5)
    ]
    responses = tmp_path / "responses.jsonl"
    _write_routed_experts_responses(responses, [r.astype(np.uint8) for r in routed])
    profile = expert_load_profile(routed_experts_of([str(responses)]), 32)
    want = np.zeros((6, 32))
    for ids in routed:
        for layer in range(6):
            np.add.at(want[layer], ids[:, layer].ravel(), 1)
    np.testing.assert_array_equal(profile, want)
    path = tmp_path / "profile.npy"
    np.save(path, profile)

    config = dr.DecodeReplicaConfig.from_vllm_config(
        _replica_vllm_config(
            {"decode_expert_replicas": 2, "decode_expert_load_profile": str(path)}
        )
    )
    assert config is not None and config.replicas == 2
    np.testing.assert_array_equal(config.profile, profile)
    for layer in range(6):
        load = config.profile[layer]
        plan = dr.plan_layer(load, 4, 2)
        assert plan
        assert all(load[e] > 0 for e, _, _ in plan)
        assert all(e // 8 != c for e, c, _ in plan)
    assert dr.DecodeReplicaConfig.from_vllm_config(_replica_vllm_config({})) is None


@pytest.mark.parametrize(
    "additional_config,parallel,match",
    [
        ({"decode_expert_replicas": 2}, {}, "needs decode_expert_load_profile"),
        ({"decode_expert_replicas": -1}, {}, "non-negative integer"),
        ({"decode_expert_load_profile": "p.npy"}, {}, "is 0"),
        ({"decode_expert_replicas": 9, "PROFILE": 1}, {}, "exceeds the 8 experts"),
        ({"decode_expert_replicas": 2, "PROFILE": 1}, {"enable_eplb": True}, "EPLB"),
        (
            {"decode_expert_replicas": 2, "PROFILE": 1},
            {"enable_expert_parallel": False},
            "expert parallelism",
        ),
        (
            {"decode_expert_replicas": 2, "PROFILE": 1},
            {"tensor_parallel_size": 16},
            "up to 8",
        ),
        ({"decode_expert_replicas": 2, "PROFILE": 1}, {"monokernel": True}, "mono"),
        ({"decode_expert_replicas": 2, "PROFILE": (5, 32)}, {}, "has shape"),
        ({"decode_expert_replicas": 2, "PROFILE": -1}, {}, "non-negative"),
        ({"decode_expert_replicas": 2, "PROFILE": "missing"}, {}, "cannot read"),
    ],
)
def test_decode_replica_config_rejects(
    tmp_path, additional_config: dict, parallel: dict, match: str
) -> None:
    from vllm.models.muse_spark.amd import decode_replicas as dr

    additional_config = dict(additional_config)
    kind = additional_config.pop("PROFILE", None)
    if kind is not None:
        path = tmp_path / "profile.npy"
        if kind != "missing":
            shape = kind if isinstance(kind, tuple) else (6, 32)
            np.save(path, np.full(shape, float(kind if kind == -1 else 1)))
        additional_config["decode_expert_load_profile"] = str(path)
    with pytest.raises(ValueError, match=match):
        dr.DecodeReplicaConfig.from_vllm_config(
            _replica_vllm_config(additional_config, **parallel)
        )


LOCAL_EXPERTS = 64
SORT_BLOCK_M = 32
SORT_QUANT_TOKENS = [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 257, 511, 512]
# AITER's HIP quant floors a 32-value block's amax at 1e-10 (scale 2^-35),
# `sort_quant` (AITER's Triton quant) at the smallest normal scale, so blocks
# below 1e-10 get other scales; both dequantize within 6 * 2^-35 of each other.
SCALE_FLOOR = 1e-10
SCALE_FLOOR_ATOL = 6 * 2**-35
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _ep_maps(
    placement: str, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """One EP4 rank's `[E]` global-to-local expert map and AITER's `[E + 1]`
    expert mask."""
    _, expert_map, expert_mask = determine_expert_map(
        ep_size=4,
        ep_rank=rank,
        global_num_experts=NUM_EXPERTS,
        expert_placement_strategy=placement,
        return_expert_mask=True,
    )
    assert expert_map is not None and expert_mask is not None
    return expert_map.to(device), expert_mask.to(device)


def _route_ids(
    num_tokens: int, kind: str, mask: torch.Tensor, g: torch.Generator
) -> torch.Tensor:
    """`[T, TOP_K]` routes; "repeat": tokens selecting an expert in several slots,
    as rows with non-finite router logits (CUDA-graph padding) do."""
    device = mask.device
    pool = NUM_EXPERTS
    if kind == "local":
        pool_ids = torch.nonzero(mask[:-1]).flatten()
        pool = pool_ids.numel()
    elif kind == "skewed":
        pool_ids = torch.randperm(NUM_EXPERTS, device=device, generator=g)[: TOP_K + 4]
        pool = pool_ids.numel()
    ids = torch.stack(
        [
            torch.randperm(pool, device=device, generator=g)[:TOP_K]
            for _ in range(num_tokens)
        ]
    )
    if kind in ("local", "skewed"):
        ids = pool_ids[ids]
    u = torch.rand(num_tokens, device=device, generator=g)
    if kind == "padding":
        ids[u < 0.3] = -1
    elif kind == "repeat":
        ids[u < 0.2] = ids[u < 0.2][:, :1]
        pair = (u >= 0.2) & (u < 0.5)
        ids[pair, 3] = ids[pair, 1]
        ids[pair, 7] = ids[pair, 1]
        ids[u >= 0.8] = 0
    return ids.to(torch.int32).contiguous()


def _rows(num_tokens: int, g: torch.Generator, device: torch.device) -> torch.Tensor:
    """bf16 rows over the exponent range, with zero, huge, tiny and NaN rows."""
    x = torch.randn(num_tokens, HIDDEN, device=device, generator=g)
    kind = num_tokens % 4
    if kind == 1:
        scale = torch.randint(-130, 125, (num_tokens, 1), device=device, generator=g)
        x = x * torch.exp2(scale.float())
    elif kind == 2:
        x[: num_tokens // 2] = 0
        if num_tokens > 2:
            x[num_tokens // 2] = 3.0e38 * torch.sign(x[num_tokens // 2])
            x[-1] = x[-1] * 1e-38
    elif kind == 3:
        x[torch.rand(num_tokens, HIDDEN, device=device, generator=g) < 0.05] = float(
            "nan"
        )
        x[: max(1, num_tokens // 4)] = float("nan")
    return x.to(torch.bfloat16).contiguous()


def _aiter_sort_quant(
    x: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    from aiter.fused_moe import moe_sorting
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort_hip

    num_tokens = x.shape[0]
    unused_weights = torch.empty(ids.shape, dtype=torch.float32, device=x.device)
    sorted_ids, _, sorted_eids, num_valid, _ = moe_sorting(
        ids,
        unused_weights,
        mask.numel(),
        HIDDEN,
        torch.bfloat16,
        SORT_BLOCK_M,
        mask,
        None,
        0,
        accumulate=False,
    )[:5]
    a1 = torch.empty(
        (num_tokens, HIDDEN // 2), dtype=torch.float4_e2m1fn_x2, device=x.device
    )
    scale = torch.empty(
        ((sorted_ids.shape[0] + 31) // 32 * 32, HIDDEN // 32),
        dtype=torch.float8_e8m0fnu,
        device=x.device,
    )
    fused_dynamic_mx_quant_moe_sort_hip(
        a1,
        scale,
        x,
        sorted_ids,
        num_valid,
        num_tokens,
        SORT_BLOCK_M,
        num_tokens * TOP_K,
    )
    return a1, scale, sorted_ids, sorted_eids, num_valid


def _sorted_row_scales(scale: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Scale bytes `[rows, HIDDEN // 32]` from AITER's swizzled e8m0 layout."""
    r = rows[:, None]
    k = torch.arange(HIDDEN // 32, device=rows.device)[None, :]
    n_pad = HIDDEN // 32
    offset = (
        (r // 32 * n_pad) * 32
        + (k // 8) * 256
        + (k % 4) * 64
        + (r % 16) * 4
        + (k % 8) // 4 * 2
        + (r % 32) // 16
    )
    return scale.view(torch.uint8).view(-1)[offset]


def _dequantize(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[N, HIDDEN // 2]` packed E2M1 bytes and `[N, HIDDEN // 32]` e8m0 bytes ->
    `[N, HIDDEN]` fp32."""
    nibbles = torch.stack([codes & 0xF, codes >> 4], dim=-1).flatten(1).long()
    lut = torch.tensor(E2M1, device=codes.device)
    values = lut[nibbles & 7] * torch.where(nibbles >= 8, -1.0, 1.0)
    scale = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)
    return values * scale


def _assert_sort_quant_matches_aiter(
    x: torch.Tensor, ids: torch.Tensor, expert_map: torch.Tensor, mask: torch.Tensor
) -> None:
    """Every entry the stage kernels read: num_valid, sorted ids / expert ids up to
    num_valid, the a1 rows of tokens with a local route and the scales of every
    valid sorted row. For a token repeating an expert only one slot is kept and
    which one is unspecified in AITER, so those entries compare the token only.
    Blocks under the scale floors compare their dequantized values."""
    num_tokens = x.shape[0]
    ref = _aiter_sort_quant(x, ids, mask)
    new = dmoe.sort_quant(x, ids, expert_map, LOCAL_EXPERTS, mask.numel(), 32)
    for r, n in zip(ref, new):
        assert r.shape == n.shape and r.dtype == n.dtype
    assert torch.equal(ref[4], new[4])
    num_valid = int(ref[4][0])
    ref_ids, new_ids = ref[2][:num_valid], new[2][:num_valid]
    token = ref_ids & 0xFFFFFF
    slots = torch.arange(TOP_K, device=ids.device)
    repeats = (
        (ids[:, None, :] == ids[:, :, None]) & (slots[None, :] != slots[:, None])
    ).any(dim=(1, 2))
    valid = token < num_tokens
    token_only = valid & repeats[token.clamp(max=num_tokens - 1).long()]
    assert torch.equal(
        torch.where(token_only, token, ref_ids),
        torch.where(token_only, new_ids & 0xFFFFFF, new_ids),
    )
    assert torch.equal(
        ref[3][: num_valid // SORT_BLOCK_M], new[3][: num_valid // SORT_BLOCK_M]
    )
    rows = torch.nonzero(valid).flatten()
    if rows.numel() == 0:
        return
    # NaN does not raise a block's amax in either quant.
    amax = x.float().nan_to_num(nan=0.0).abs().view(num_tokens, HIDDEN // 32, 32)
    amax = amax.amax(dim=-1)
    floor = amax < SCALE_FLOOR
    row_floor = floor[token[rows].long()]
    ref_scales = _sorted_row_scales(ref[1], rows)
    new_scales = _sorted_row_scales(new[1], rows)
    assert torch.equal(ref_scales[~row_floor], new_scales[~row_floor])
    tokens = torch.unique(token[rows].long())
    first = torch.zeros(num_tokens, dtype=torch.long, device=x.device)
    first[token[rows].long()] = torch.arange(rows.numel(), device=x.device)
    ref_a1, new_a1 = (a.view(torch.uint8)[tokens] for a in (ref[0], new[0]))
    byte_floor = floor[tokens].repeat_interleave(16, dim=1)
    assert torch.equal(ref_a1[~byte_floor], new_a1[~byte_floor])
    torch.testing.assert_close(
        _dequantize(new_a1, new_scales[first[tokens]]),
        _dequantize(ref_a1, ref_scales[first[tokens]]),
        atol=SCALE_FLOOR_ATOL,
        rtol=0,
        equal_nan=True,
    )


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize(
    "placement, rank", [("linear", 0), ("linear", 3), ("round_robin", 1)]
)
@pytest.mark.parametrize(
    "route_kind", ["random", "skewed", "local", "padding", "repeat"]
)
@torch.inference_mode()
def test_sort_quant_matches_the_aiter_sort_and_quant(
    placement: str, rank: int, route_kind: str
) -> None:
    """One `sort_quant` launch writes the buffers of AITER's moe_sorting +
    `fused_dynamic_mx_quant_moe_sort_hip` for T = 1..512: bit for bit, except
    blocks under the quant's scale floor."""
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)
    expert_map, mask = _ep_maps(placement, rank, device)
    for num_tokens in SORT_QUANT_TOKENS:
        ids = _route_ids(num_tokens, route_kind, mask, g)
        _assert_sort_quant_matches_aiter(
            _rows(num_tokens, g, device), ids, expert_map, mask
        )
