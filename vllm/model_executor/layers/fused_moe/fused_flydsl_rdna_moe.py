# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL W4A16 grouped (MoE) GEMM for RDNA3 / RDNA3.5 (gfx11*, wave32).

One kernel serves both MoE GEMMs: ``out[r, :] = W[expert(r)] @ x[row(r), :]``
for every routed row ``r``, with rows grouped by expert into 16-row blocks by
``moe_align_block_size``.

A wave computes a 16 x (16 * REG_N) output tile with bf16 WMMA. Weights stay
packed int4 until the WMMA operand: a nibble ``q`` is placed in the mantissa
of the bf16 value ``1 + q / 16`` by shift/mask/or, so no int->float conversion
is needed. Per group of 32 inputs the kernel therefore accumulates
``P = sum_k x_k * (1 + q_k / 16)`` and ``S = sum_k x_k`` (WMMA against ones),
and folds scale and zero point in f32:

    sum_k x_k * s * (q_k - zp) = (16 * s) * P - s * (16 + zp) * S

``pack_w4a16_weights`` produces the kernel's weight and coefficient layout.
"""

import functools

import torch

GROUP_SIZE = 32
BLOCK_M = 16
REG_N = 4
TILE_N = 16 * REG_N

# Bit positions of the two nibbles of an element pair inside a packed dword
# (pair p at bits 4p and 4p + 16) and inside a bf16 pair (mantissa bits 3..6).
_NIBBLE_MASK = 0x00780078
_BF16_ONE_PAIR = 0x3F803F80

# DRAM channels interleave every 256 B on gfx115x. Waves of a launch read
# their tiles in lockstep, so tile strides that are a multiple of 512 B would
# make them all pull from the same channel at the same time.
_CHANNEL_BYTES = 256


def _tile_stride(payload_dwords: int) -> int:
    """Dwords per tile, padded to an odd multiple of the channel granule."""
    granules = -(-payload_dwords * 4 // _CHANNEL_BYTES)
    return (granules | 1) * _CHANNEL_BYTES // 4


def _w_tile_dwords(K: int) -> int:
    return (K // 16) * 16 * 2 * REG_N


def _coef_tile_dwords(K: int) -> int:
    return (K // GROUP_SIZE) * 16 * REG_N


def pack_w4a16_weights(
    q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack unsigned int4 weights for the kernel.

    Args:
        q: ``[E, N, K]`` unsigned nibbles (0..15).
        scale: ``[E, N, K // 32]`` group scales.
        zp: ``[E, N, K // 32]`` group zero points (0..15).

    Returns:
        Per ``(expert, 64-column tile)`` rows of dwords: ``w`` int32 with the
        nibbles ordered ``[K/16, 16, REG_N, 2]`` and ``coef`` float16 pairs
        ``(16 s, -s (16 + zp))`` ordered ``[K/32, 16, REG_N]``. Rows are padded
        to the tile stride.

    """
    E, N, K = q.shape
    G = K // GROUP_SIZE
    tiles = N // TILE_N
    assert N % TILE_N == 0 and K % GROUP_SIZE == 0
    # [E, tile, rn, lane, kstep, dword, half, pair] -> nibble at 4 * pair + 16 * half
    qv = q.to(torch.int64).view(E, tiles, REG_N, 16, K // 16, 2, 4, 2)
    pair = torch.arange(4, device=q.device).view(4, 1)
    half = torch.arange(2, device=q.device).view(1, 2)
    w = (qv << (4 * pair + 16 * half)).sum(dim=(-1, -2))
    w = torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)
    w = w.permute(0, 1, 4, 3, 2, 5).reshape(E, tiles, -1)

    s = scale.to(torch.float32)
    coef = torch.stack((16.0 * s, -s * (16.0 + zp.to(torch.float32))), dim=-1)
    coef = coef.view(E, tiles, REG_N, 16, G, 2).permute(0, 1, 4, 3, 2, 5)
    coef = coef.to(torch.float16).reshape(E, tiles, -1)

    def _pad(t: torch.Tensor, dwords: int) -> torch.Tensor:
        per_dword = 4 // t.element_size()
        extra = (_tile_stride(dwords) - dwords) * per_dword
        return torch.nn.functional.pad(t, (0, extra)).contiguous()

    return _pad(w, _w_tile_dwords(K)), _pad(coef, _coef_tile_dwords(K))


@functools.cache
def _build_gemm(K: int, N: int, x_row_div: int):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import buffer_ops, gpu, math, range_constexpr, rocdl
    from flydsl.expr.typing import Vector

    assert K % GROUP_SIZE == 0 and N % TILE_N == 0
    n_groups = K // GROUP_SIZE
    k_dwords = K // 2
    w_step = 16 * 2 * REG_N  # dwords per K-step of one output tile
    w_tile = _tile_stride(_w_tile_dwords(K))
    c_group = 16 * REG_N
    c_tile = _tile_stride(_coef_tile_dwords(K))

    @flyc.kernel
    def gemm_kernel(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid: fx.Tensor,
        n_rows: fx.Int32,
    ):
        lane = gpu.thread_idx.x
        tile_n = gpu.block_idx.x
        block_m = gpu.block_idx.y
        lane16 = lane % 16
        klane = lane // 16

        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_coef, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(arg_ids, max_size=True)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=True)
        nv_rsrc = buffer_ops.create_buffer_resource(arg_num_valid, max_size=True)

        num_valid = buffer_ops.buffer_load(nv_rsrc, 0, vec_width=1, dtype=fx.Int32)
        row0 = block_m * BLOCK_M
        # Padding blocks run zero groups and store nothing.
        block_valid = row0 < num_valid
        expert = block_valid.select(
            buffer_ops.buffer_load(eid_rsrc, block_m, vec_width=1, dtype=fx.Int32),
            fx.Int32(0),
        )
        rid = buffer_ops.buffer_load(
            ids_rsrc, row0 + lane16, vec_width=1, dtype=fx.Int32
        )
        x_row = (block_valid & (rid < n_rows)).select(rid // x_row_div, fx.Int32(0))
        x_base = x_row * k_dwords
        tile = expert * (N // TILE_N) + tile_n
        w_base = tile * w_tile + lane16 * (2 * REG_N)
        c_base = tile * c_tile + lane16 * REG_N

        ones = fx.full(16, _BF16_ONE_PAIR & 0xFFFF, fx.Int16)
        zero = fx.full(8, 0.0, fx.Float32)

        def load_group(g):
            """Issue the loads of group ``g``: 4 x, REG_N w, 1 coef."""
            vecs = [
                buffer_ops.buffer_load(
                    x_rsrc, x_base + g * 16 + j * 4, vec_width=4, dtype=fx.Int32
                )
                for j in range_constexpr(4)
            ]
            for h in range_constexpr(2):
                for j in range_constexpr(REG_N // 2):
                    vecs.append(
                        buffer_ops.buffer_load(
                            w_rsrc,
                            w_base + g * (2 * w_step) + (h * w_step + j * 4),
                            vec_width=4,
                            dtype=fx.Int32,
                        )
                    )
            vecs.append(
                buffer_ops.buffer_load(
                    c_rsrc, c_base + g * c_group, vec_width=REG_N, dtype=fx.Int32
                )
            )
            return vecs

        def unpack(lo, hi):
            words = []
            for w in (lo, hi):
                words.append(((w << 3) & _NIBBLE_MASK) | _BF16_ONE_PAIR)
                for p in range_constexpr(1, 4):
                    words.append(((w >> (4 * p - 3)) & _NIBBLE_MASK) | _BF16_ONE_PAIR)
            return Vector.from_elements(words, fx.Int32).bitcast(fx.Int16)

        def wmma(a, b, acc):
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a, b, acc).result

        # The next group's loads stay in flight while a group is computed;
        # their results are carried as loop state.
        n_iter = block_valid.select(fx.Int32(n_groups), fx.Int32(0))
        init = [zero for _ in range_constexpr(REG_N)] + load_group(fx.Int32(0))
        for i, state in range(0, n_iter, 1, init=init):  # type: ignore[call-overload]
            tot = list(state[:REG_N])
            cur = list(state[REG_N:])
            i_next = fx.Int32(i) + 1
            # The last iteration reloads group 0 so that nothing is read past
            # the tile.
            nxt = load_group((i_next < n_groups).select(i_next, fx.Int32(0)))

            acc = [zero for _ in range_constexpr(REG_N)]
            xsum = zero
            for h in range_constexpr(2):
                a = cur[2 * h].shuffle(cur[2 * h + 1], list(range(8)))
                a = a.bitcast(fx.Int16)
                xsum = wmma(a, ones, xsum)
                wv = [
                    cur[4 + h * (REG_N // 2) + j][e]
                    for j in range_constexpr(REG_N // 2)
                    for e in range_constexpr(4)
                ]
                for rn in range_constexpr(REG_N):
                    acc[rn] = wmma(a, unpack(wv[2 * rn], wv[2 * rn + 1]), acc[rn])
            cf = cur[4 + REG_N].bitcast(fx.Float16).to(fx.Float32)
            for rn in range_constexpr(REG_N):
                s16 = fx.full(8, cf[2 * rn], fx.Float32)
                szp = fx.full(8, cf[2 * rn + 1], fx.Float32)
                tot[rn] = math.fma(acc[rn], s16, math.fma(xsum, szp, tot[rn]))
            results = yield tot + nxt
        tot = list(results[:REG_N])

        for si in range_constexpr(8):
            out_rid = buffer_ops.buffer_load(
                ids_rsrc, row0 + 2 * si + klane, vec_width=1, dtype=fx.Int32
            )
            # RDNA buffer resources are not bounds checked, so padding rows
            # have to be skipped rather than masked.
            if block_valid & (out_rid < n_rows):
                out_base = out_rid * N + tile_n * TILE_N + lane16
                for rn in range_constexpr(REG_N):
                    buffer_ops.buffer_store(
                        tot[rn][si].to(fx.BFloat16), out_rsrc, out_base + rn * 16
                    )

    @flyc.jit
    def launch(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid: fx.Tensor,
        n_rows: fx.Int32,
        n_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        gemm_kernel(
            arg_out,
            arg_x,
            arg_w,
            arg_coef,
            arg_ids,
            arg_expert_ids,
            arg_num_valid,
            n_rows,
        ).launch(grid=(N // TILE_N, n_blocks, 1), block=(32, 1, 1), stream=stream)

    return launch


def flydsl_rdna_moe_gemm(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    coef: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_valid: torch.Tensor,
    x_row_div: int,
) -> None:
    """Run ``out[r] = W[expert(r)] @ x[r // x_row_div]`` for all routed rows.

    ``sorted_ids`` / ``expert_ids`` / ``num_valid`` come from
    ``moe_align_block_size`` with block size ``BLOCK_M``.
    """
    n_rows, N = out.shape
    launch = _build_gemm(x.shape[1], N, x_row_div)
    launch(
        out,
        x,
        w,
        coef,
        sorted_ids,
        expert_ids,
        num_valid,
        n_rows,
        expert_ids.numel(),
        torch.cuda.current_stream(),
    )
