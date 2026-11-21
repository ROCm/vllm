# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark fused all-reduces on ROCm against the fp32 kernels they replace:
the decode MoE finalize and the attention boundary, each followed by an fp32
sum of the ranks' bf16 partials in rank order."""

import pytest
import torch

from vllm.models.muse_spark.amd.ops.fused_allreduce import WORLD_SIZES
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test, multi_gpu_marks
from .eplb_utils import distributed_run, set_env_vars_and_device

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Muse Spark ROCm kernels"
)

NUM_EXPERTS = 256
# One hop and two hops, row counts not divisible by the world size, and owners
# that change between calls of the same row (the boundary up to one row per CU).
ROWS = [1, 8, 9, 64, 192, 7, 320, 64]
BF16_EPS = 2.0**-8
# One bf16 ulp, relative: up to 2**-7 just above a power of two.
BF16_ULP = 2.0**-7
FP8_EPS = 2.0**-4


def _gather(partial: torch.Tensor) -> list[torch.Tensor]:
    world = torch.distributed.get_world_size()
    parts = [torch.empty_like(partial) for _ in range(world)]
    torch.distributed.all_gather(parts, partial)
    return parts


def _rank_order_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    total = parts[0].float()
    for part in parts[1:]:
        total = total + part.float()
    return total


def _check_reduced(out: torch.Tensor, parts: list[torch.Tensor]) -> None:
    """`out` is the bf16 sum of `parts` up to one bf16 ulp of each partial (the
    fused partial may round the other way) and one rounding of the sum."""
    expected = _rank_order_sum(parts)
    bound = BF16_ULP * sum(p.float().abs() for p in parts) + BF16_EPS * expected.abs()
    assert ((out.float() - expected).abs() <= bound).all()


def _check_fp8(q, q_scale, expected: torch.Tensor, bf16_bound: torch.Tensor):
    dequant = q.float() * q_scale[:, None]
    bound = FP8_EPS * expected.abs() + 2.0**-10 * q_scale[:, None] + bf16_bound
    assert ((dequant - expected).abs() <= bound).all()


def _one_block_mode(fused, rows: int, finalize: bool) -> int:
    """The communication mode the one-block-per-row kernels take at `rows`."""
    from vllm.models.muse_spark.amd.ops import fused_allreduce as far

    two_hop = far.FINALIZE_TWO_HOP_MODE[fused.world] if finalize else far._TWO_HOP
    return far._mode(rows, two_hop)


def _finalize_inputs(rows: int, rank: int, generator: torch.Generator) -> dict:
    """A finalize call's inputs and this rank's unfused partial, gathered
    (collective)."""
    from vllm.models.muse_spark.common.ops.fused_finalize import (
        fused_post_expert_finalize,
    )

    hidden, top_k = 4096, 8
    ids = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, device="cuda", generator=generator)[:top_k]
            for _ in range(rows)
        ]
    ).int()
    local = NUM_EXPERTS // torch.distributed.get_world_size()
    expert_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device="cuda")
    expert_map[rank * local : (rank + 1) * local] = torch.arange(
        local, dtype=torch.int32, device="cuda"
    )
    routes = torch.randn(rows, top_k, hidden, device="cuda", generator=generator)
    routes = (routes * (rank + 1)).bfloat16()
    # Rows of other ranks' experts are never written.
    routes[expert_map[ids.long()] < 0] = float("nan")
    weight = (0.1 * torch.randn(hidden, device="cuda", generator=generator)).bfloat16()
    topk_weights = torch.rand(rows, top_k, device="cuda", generator=generator)

    partial = torch.empty(rows, hidden, dtype=torch.bfloat16, device="cuda")
    fused_post_expert_finalize(
        routes, weight, topk_weights, 1e-8, 1.0, True, partial,
        topk_ids=ids, expert_map=expert_map,
    )  # fmt: skip
    return dict(
        args=(routes, weight, topk_weights, ids, expert_map, 1e-8, 1.0),
        parts=_gather(partial),
    )


def _check_finalize(case: dict, out, q, q_scale) -> None:
    parts = case["parts"]
    # The fused partial may differ from the Triton one by one rounding.
    if out is not None:
        _check_reduced(out, parts)
    if q is not None:
        expected = _rank_order_sum(parts)
        bf16_bound = (
            2 * BF16_EPS * (sum(p.float().abs() for p in parts) + expected.abs())
        )
        _check_fp8(q, q_scale, expected, bf16_bound)


def _finalize(fused, rows: int, rank: int, generator: torch.Generator) -> None:
    case = _finalize_inputs(rows, rank, generator)
    args = case["args"]
    out, _, _ = fused.finalize(*args, quant=False)
    _, q, q_scale = fused.finalize(*args, quant=True)
    _check_finalize(case, out, None, None)
    _check_finalize(case, None, q, q_scale)
    for t in (out, q.view(torch.uint8), q_scale):
        assert all(torch.equal(t, x) for x in _gather(t))
    # The one-block-per-row kernels give the same values as the multi-block
    # ones (FP8 zeros may differ in sign).
    mode = _one_block_mode(fused, rows, finalize=True)
    out_1, _, _ = fused.finalize(*args, mode=mode)
    _, q_1, q_scale_1 = fused.finalize(*args, quant=True, mode=mode)
    assert torch.equal(out_1, out) and torch.equal(q_scale_1, q_scale)
    assert torch.equal(q_1.float(), q.float())


def _boundary_inputs(
    rows: int,
    generator: torch.Generator,
    shared: torch.Generator,
    post_weight: bool = False,
    post_input_scale: bool = False,
    pre_input_scale: bool | None = None,
) -> dict:
    """A boundary call's inputs and the unfused kernel's outputs on the
    all-reduced partial (collective). By default the decoder's post-attention
    norm (no weight), pre-FFN norm and gated residual."""
    from vllm.models.muse_spark.common.layers import (
        MuseSparkNormPreservingResidualConnection,
        MuseSparkRMSNorm,
    )
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_quant_fwd,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    hidden = 8192
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        post_norm = MuseSparkRMSNorm(
            hidden, eps=1e-8, has_weight=post_weight, is_post_norm=True
        )
        pre_norm = MuseSparkRMSNorm(
            hidden,
            eps=1e-5,
            has_weight=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
        )
        connection = MuseSparkNormPreservingResidualConnection(dim=hidden)
    # Replicated across ranks, as in the model.
    pre_norm.weight.data.normal_(0, 0.1, generator=shared)
    post_w = None
    if post_weight:
        post_norm.weight.data.normal_(0, 0.1, generator=shared)
        post_w = post_norm.weight
    connection.gate.data.normal_(0, 1, generator=shared)
    connection.materialize_coefficients()
    stream = 4 * torch.randn(rows, hidden, device="cuda", generator=shared)

    partial = torch.randn(rows, hidden, device="cuda", generator=generator).bfloat16()
    kwargs = dict(
        post_gain_center=post_norm.fused_gain_center if post_weight else 0.0,
        pre_gain_center=pre_norm.fused_gain_center,
        post_input_scale=post_input_scale and post_weight,
        pre_input_scale=pre_norm.applies_weight_before_norm
        if pre_input_scale is None
        else pre_input_scale,
        out_dtype=torch.bfloat16,
    )
    eps = (post_norm.variance_epsilon, pre_norm.variance_epsilon)
    args = (partial, stream, connection.alpha, connection.beta, post_w, pre_norm.weight)
    reduced = _rank_order_sum(_gather(partial)).bfloat16()
    ref = fused_norm_residual_norm_quant_fwd(
        reduced, stream, post_w, pre_norm.weight, *eps,
        alpha=connection.alpha, beta=connection.beta, **kwargs,
    )  # fmt: skip
    return dict(args=args, eps=eps, kwargs=kwargs, ref=ref)


def _call_boundary(fused, case: dict, mode: int | None = None):
    return fused.boundary(*case["args"], *case["eps"], **case["kwargs"], mode=mode)


def _check_boundary(case: dict, outs) -> None:
    out, new_stream, q, q_scale = outs
    ref_out, ref_stream, ref_q, ref_scale = case["ref"]
    # Same inputs to the norms; only the fp32 reduction orders differ, which can
    # move the bf16 and FP8 outputs by one code.
    torch.testing.assert_close(new_stream, ref_stream, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out, ref_out, rtol=2 * BF16_EPS, atol=1e-5)
    torch.testing.assert_close(q_scale, ref_scale, rtol=2 * BF16_EPS, atol=0)
    torch.testing.assert_close(
        q.float() * q_scale[:, None],
        ref_q.float() * ref_scale[:, None],
        rtol=2 * FP8_EPS,
        atol=2.0**-9 * float(ref_scale.max()),
    )


def _boundary(
    fused, rows: int, generator: torch.Generator, shared: torch.Generator, **variant
) -> None:
    case = _boundary_inputs(rows, generator, shared, **variant)
    outs = _call_boundary(fused, case)
    _check_boundary(case, outs)
    out, new_stream, q, q_scale = outs
    for t in (new_stream, out, q.view(torch.uint8)):
        assert all(torch.equal(t, x) for x in _gather(t))
    # The one-block-per-row kernel reduces the same sum; its epilogue orders the
    # fp32 arithmetic differently, so the outputs may differ by one rounding.
    out_1, stream_1, q_1, q_scale_1 = _call_boundary(
        fused, case, mode=_one_block_mode(fused, rows, finalize=False)
    )
    torch.testing.assert_close(stream_1, new_stream, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_1, out, rtol=2 * BF16_EPS, atol=1e-5)
    torch.testing.assert_close(q_scale_1, q_scale, rtol=2 * BF16_EPS, atol=0)


def _row_counts(fused) -> list[int]:
    """Multi-block, one-hop and two-hop row counts, up to one row per CU."""
    rows = [1, 3, fused.mb_max_rows, 8, 9, 64, 130, fused.boundary_max_rows]
    return [r for r in dict.fromkeys(rows) if 0 < r <= fused.boundary_max_rows]


def _back_to_back(fused, rank, generator, shared) -> None:
    """Calls of varying row counts with no other collective between them: a
    rank may start a call while its peers are still in the previous one."""
    rows = _row_counts(fused)
    rows = rows + rows[::-1]
    fin = [_finalize_inputs(m, rank, generator) for m in rows]
    bnd = [_boundary_inputs(m, generator, shared) for m in rows]
    torch.accelerator.synchronize()
    torch.distributed.barrier()
    results = []
    for f, b in zip(fin, bnd):
        results.append(fused.finalize(*f["args"], quant=False))
        results.append(fused.finalize(*f["args"], quant=True))
        results.append(_call_boundary(fused, b))
    torch.accelerator.synchronize()
    for k, (f, b) in enumerate(zip(fin, bnd)):
        _check_finalize(f, results[3 * k][0], None, None)
        _check_finalize(f, None, *results[3 * k + 1][1:])
        _check_boundary(b, results[3 * k + 2])


def _graph_replay(fused, rank, generator, shared) -> None:
    """Calls captured in a CUDA graph and replayed with new inputs."""
    rows = [m for m in (1, 8, 64, 130) if m <= fused.boundary_max_rows]
    fin = [_finalize_inputs(m, rank, generator) for m in rows]
    bnd = [_boundary_inputs(m, generator, shared) for m in rows]
    torch.accelerator.synchronize()
    torch.distributed.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [
            (
                fused.finalize(*f["args"], quant=False),
                fused.finalize(*f["args"], quant=True),
                _call_boundary(fused, b),
            )
            for f, b in zip(fin, bnd)
        ]
    for _ in range(3):
        # New inputs in the captured tensors, and their references.
        for f, b, m in zip(fin, bnd, rows):
            new_f = _finalize_inputs(m, rank, generator)
            new_b = _boundary_inputs(m, generator, shared)
            for dst, src in zip(f["args"][:5], new_f["args"][:5]):
                dst.copy_(src)
            f["parts"] = new_f["parts"]
            for dst, src in zip(b["args"], new_b["args"]):
                if dst is not None:
                    dst.copy_(src)
            b["ref"] = new_b["ref"]
        torch.accelerator.synchronize()
        torch.distributed.barrier()
        graph.replay()
        torch.accelerator.synchronize()
        for (out, q, b_outs), f, b in zip(outs, fin, bnd):
            _check_finalize(f, out[0], None, None)
            _check_finalize(f, None, *q[1:])
            _check_boundary(b, b_outs)
    del graph


def _state_near_wrap(fused, rank, generator, shared) -> None:
    """Row states seeded where a 32-bit call counter would wrap give correct
    results, and the states stay bounded."""
    rows = _row_counts(fused)
    called = max(rows)  # every row below is called
    for op in (fused.finalize_op, fused.boundary_op):
        # [slot, previous owner] per row; owner 0: rank 0 owned the last call.
        state = op.state.view(-1, 2)[:called]
        state[:, 0] = -2  # 0xfffffffe as uint32
        state[:, 1] = 0
    for op in (fused.mb_finalize_op, fused.mb_boundary_op):
        op.state.fill_(2**31 - 2)
    for _ in range(4):
        for m in rows:
            _finalize(fused, m, rank, generator)
            _boundary(fused, m, generator, shared)
    for op in (fused.finalize_op, fused.boundary_op):
        slots = op.state.view(-1, 2)[:called, 0]
        assert ((slots >= 0) & (slots < 3)).all()
    for op in (fused.mb_finalize_op, fused.mb_boundary_op):
        assert ((op.state >= 0) & (op.state < 6)).all()


def _epilogue_variants(fused, generator, shared) -> None:
    """The boundary with a post-norm weight, applied before or after the norm,
    and either pre-norm weight order."""
    rows = [m for m in (fused.mb_max_rows, 64) if 0 < m <= fused.boundary_max_rows]
    for m in rows:
        for post_input_scale in (False, True):
            for pre_input_scale in (False, True):
                _boundary(
                    fused, m, generator, shared,
                    post_weight=True,
                    post_input_scale=post_input_scale,
                    pre_input_scale=pre_input_scale,
                )  # fmt: skip


def fused_allreduces_match_the_unfused_kernels(
    env: dict[str, str], world_size: int, inbox: str
) -> None:
    set_env_vars_and_device(env)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized
    from vllm.models.muse_spark.amd.ops import fused_allreduce

    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(world_size, 1)
    rank = int(env["RANK"])
    tp = get_tp_group()

    def build():
        fused = fused_allreduce.MuseFusedAllReduce(
            tp.cpu_group, tp.device_group, torch.device("cuda", rank), inbox=inbox
        )
        assert fused.inbox_kind == inbox
        return fused

    generator = torch.Generator(device="cuda").manual_seed(rank)
    shared = torch.Generator(device="cuda").manual_seed(world_size)
    with torch.inference_mode():
        # Seeded states need clean inboxes: a fresh instance.
        fused = build()
        _state_near_wrap(fused, rank, generator, shared)
        fused.close()
        fused.close()
        fused = build()
        for rows in ROWS:
            _finalize(fused, rows, rank, generator)
            if rows <= fused.boundary_max_rows:
                _boundary(fused, rows, generator, shared)
        _epilogue_variants(fused, generator, shared)
        _back_to_back(fused, rank, generator, shared)
        _graph_replay(fused, rank, generator, shared)
        fused.close()


@pytest.mark.parametrize("inbox", ["symm_mem", "ipc"])
@pytest.mark.parametrize(
    "world_size",
    [pytest.param(w, marks=multi_gpu_marks(num_gpus=w)) for w in WORLD_SIZES],
)
@create_new_process_for_each_test()
def test_muse_spark_fused_allreduces_match_the_unfused_kernels(world_size, inbox):
    """The fused finalize + all-reduce (bf16 and FP8 outputs) and the fused
    attention all-reduce + boundary match the kernels they replace, with the
    same bits on every rank, with either inbox allocation: across the epilogue
    variants, back to back, in a replayed CUDA graph, and from row states at the
    wrap of a 32-bit counter."""
    from vllm.models.muse_spark.amd.ops.fused_allreduce import is_supported

    if not is_supported():
        pytest.skip("the fused all-reduce kernels run on gfx950")
    distributed_run(fused_allreduces_match_the_unfused_kernels, world_size, inbox)
