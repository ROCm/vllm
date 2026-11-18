# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL W4A16 grouped (MoE) GEMM for RDNA3.5 (gfx115x, wave32).

One kernel serves both MoE GEMMs: ``out[r, :] = W[expert(r)] @ x[row(r), :]``
for every routed row ``r``, with rows grouped by expert into 16-row blocks by
``moe_align_block_size``.

A wave computes a 16 x (16 * REG_N) output tile with bf16 WMMA. Weights stay
packed int4 until the WMMA operand: a nibble ``q`` is placed in the mantissa
of the bf16 value ``1 + q / 16`` by shift/mask/or, so no int->float conversion
is needed. Per quantization group the kernel therefore accumulates
``P = sum_k x_k * (1 + q_k / 16)`` and ``S = sum_k x_k`` (WMMA against ones),
and folds scale and zero point in f32:

    sum_k x_k * s * (q_k - zp) = (16 * s) * P - s * (16 + zp) * S

``pack_w4a16_weights`` produces the kernel's weight and coefficient layout.
"""

import functools

import torch

BLOCK_M = 16
REG_N = 4
TILE_N = 16 * REG_N
# K-steps whose loads are kept in flight ahead of the one being computed.
_LOOKAHEAD = 6
# Rows of the one-token reduce kernel: every second row of a WMMA A operand.
MAX_REDUCE_ROWS = 8

# Bit positions of the two nibbles of an element pair inside a packed dword
# (pair p at bits 4p and 4p + 16) and inside a bf16 pair (mantissa bits 3..6).
_NIBBLE_MASK = 0x00780078
_BF16_ONE_PAIR = 0x3F803F80

# DRAM channels interleave every 256 B on gfx115x. Waves of a launch read
# their tiles in lockstep, so tile strides that are a multiple of 512 B would
# make them all pull from the same channel at the same time.
_CHANNEL_BYTES = 256
# The same holds across experts: the one-token reduce kernel reads the same
# tile offset of every routed expert at once, so the expert stride gets one
# more granule.
_EXPERT_PAD_DWORDS = _CHANNEL_BYTES // 4
# slc | dlc: weights of one token are read once, so they are loaded without
# being kept in the caches, like a non-temporal load.
_NON_TEMPORAL = 6
_LDS_BYTES = 65536


def _tile_stride(payload_dwords: int) -> int:
    """Dwords per tile, padded to an odd multiple of the channel granule."""
    granules = -(-payload_dwords * 4 // _CHANNEL_BYTES)
    return (granules | 1) * _CHANNEL_BYTES // 4


def pad_expert_stride(t: torch.Tensor) -> torch.Tensor:
    """Return ``t`` (``[E, tiles, tile stride]``) as a view of a buffer whose
    expert stride is ``_EXPERT_PAD_DWORDS`` longer."""
    E, flat = t.size(0), t[0].numel()
    extra = _EXPERT_PAD_DWORDS * 4 // t.element_size()
    buf = t.new_zeros((E, flat + extra))
    buf[:, :flat] = t.reshape(E, flat)
    return buf[:, :flat].view(t.shape)


def _expert_stride_dwords(t: torch.Tensor) -> int:
    return t.stride(0) * t.element_size() // 4


def _check_coef(coef: torch.Tensor, K: int, group: int, pairs: bool) -> None:
    stride = _tile_stride(_coef_tile_dwords(K, group, pairs))
    assert coef.size(-1) * coef.element_size() == stride * 4, "coefficient layout"


def _w_tile_dwords(K: int) -> int:
    return (K // 16) * 16 * 2 * REG_N


def _coef_tile_dwords(K: int, group: int, pairs: bool) -> int:
    return (K // group) * 16 * REG_N * (2 if pairs else 1) // 2


def _scale_terms(c, pairs: bool):
    """Float32 ``(16 s, -s (16 + zp))`` from the loaded coefficient(s)."""
    import flydsl.expr as fx

    if pairs:
        cf = c.to(fx.Float32)
        return cf[0], cf[1]
    s16 = fx.Float16(c).to(fx.Float32)
    return s16, s16 * -1.5


def gate_up_tile_order(n_out: int, device: torch.device) -> torch.Tensor:
    """Column order that puts each gate column 16 * REG_N / 2 columns before
    its up column inside one tile, as the fused SiLU-and-multiply needs."""
    half = TILE_N // 2
    cols = torch.arange(n_out, device=device).view(2, -1, half)
    return cols.transpose(0, 1).reshape(-1)


def pack_w4a16_weights(
    q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack unsigned int4 weights for the kernel.

    Args:
        q: ``[E, N, K]`` unsigned nibbles (0..15).
        scale: ``[E, N, K // group]`` group scales; the group size is a
            multiple of 16.
        zp: ``[E, N, K // group]`` group zero points (0..15), or None for
            symmetric weights (zero point 8).

    Returns:
        Per ``(expert, 64-column tile)`` rows of dwords: ``w`` int32 with the
        nibbles ordered ``[K/16, REG_N, 16, 2]`` and ``coef`` float16 pairs
        ``(16 s, -s (16 + zp))`` ordered ``[K/group, REG_N, 16]``. Rows are padded
        to the tile stride. Without ``zp`` only ``16 s`` is stored: the second
        coefficient is -1.5 times the first.

    """
    E, N, K = q.shape
    G = scale.shape[2]
    tiles = N // TILE_N
    assert N % TILE_N == 0 and K % G == 0 and (K // G) % 16 == 0
    # [E, tile, rn, lane, kstep, dword, half, pair] -> nibble at 4 * pair + 16 * half
    qv = q.to(torch.int64).view(E, tiles, REG_N, 16, K // 16, 2, 4, 2)
    pair = torch.arange(4, device=q.device).view(4, 1)
    half = torch.arange(2, device=q.device).view(1, 2)
    w = (qv << (4 * pair + 16 * half)).sum(dim=(-1, -2))
    w = torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)
    w = w.permute(0, 1, 4, 2, 3, 5).reshape(E, tiles, -1)

    s = scale.to(torch.float32)
    coef = (16.0 * s).unsqueeze(-1)
    if zp is not None:
        coef = torch.cat((coef, -s.unsqueeze(-1) * (16.0 + zp.unsqueeze(-1))), dim=-1)
    coef = coef.view(E, tiles, REG_N, 16, G, -1).permute(0, 1, 4, 2, 3, 5)
    coef = coef.to(torch.float16).reshape(E, tiles, -1)

    def _pad(t: torch.Tensor, dwords: int) -> torch.Tensor:
        per_dword = 4 // t.element_size()
        extra = (_tile_stride(dwords) - dwords) * per_dword
        return torch.nn.functional.pad(t, (0, extra)).contiguous()

    return (
        _pad(w, _w_tile_dwords(K)),
        _pad(coef, _coef_tile_dwords(K, K // G, zp is not None)),
    )


@functools.cache
def _build_gemm(
    K: int,
    N: int,
    x_row_div: int,
    group: int,
    silu_mul: bool,
    w_expert: int,
    c_expert: int,
    coef_pairs: bool,
):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import buffer_ops, gpu, math, range_constexpr, rocdl
    from flydsl.expr.typing import Vector

    assert K % group == 0 and group % 16 == 0 and N % TILE_N == 0
    n_groups = K // group
    steps = group // 16  # WMMA K-steps per group
    k_dwords = K // 2
    w_step = 16 * 2 * REG_N  # dwords per K-step of one output tile
    w_tile = _tile_stride(_w_tile_dwords(K))
    # Coefficient offsets are in float16 elements.
    cw = 2 if coef_pairs else 1
    c_group = 16 * REG_N * cw
    c_tile = 2 * _tile_stride(_coef_tile_dwords(K, group, coef_pairs))
    c_expert *= 2
    # With ``silu_mul`` a tile holds gate columns then the matching up columns.
    out_reg = REG_N // 2 if silu_mul else REG_N
    n_out = N // 2 if silu_mul else N

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
        # Padding blocks and blocks without an expert (id -1) run zero groups
        # and store nothing.
        expert = buffer_ops.buffer_load(eid_rsrc, block_m, vec_width=1, dtype=fx.Int32)
        block_valid = (row0 < num_valid) & (expert >= 0)
        expert = block_valid.select(expert, fx.Int32(0))
        rid = buffer_ops.buffer_load(
            ids_rsrc, row0 + lane16, vec_width=1, dtype=fx.Int32
        )
        x_row = (block_valid & (rid < n_rows)).select(rid // x_row_div, fx.Int32(0))
        x_base = x_row * k_dwords
        w_base = expert * w_expert + tile_n * w_tile + lane16 * 2
        c_base = expert * c_expert + tile_n * c_tile + lane16 * cw

        ones = fx.full(16, _BF16_ONE_PAIR & 0xFFFF, fx.Int16)
        zero = fx.full(8, 0.0, fx.Float32)

        def load_step(kstep):
            """Issue the loads of one K-step: 2 x and REG_N w vectors."""
            vecs = [
                buffer_ops.buffer_load(
                    x_rsrc, x_base + kstep * 8 + j * 4, vec_width=4, dtype=fx.Int32
                )
                for j in range_constexpr(2)
            ]
            for rn in range_constexpr(REG_N):
                vecs.append(
                    buffer_ops.buffer_load(
                        w_rsrc,
                        w_base + kstep * w_step + rn * 32,
                        vec_width=2,
                        dtype=fx.Int32,
                    )
                )
            return vecs

        def load_coef(g):
            return [
                buffer_ops.buffer_load(
                    c_rsrc,
                    c_base + g * c_group + rn * (16 * cw),
                    vec_width=cw,
                    dtype=fx.Float16,
                )
                for rn in range_constexpr(REG_N)
            ]

        def unpack(lo, hi):
            words = []
            for w in (lo, hi):
                words.append(((w << 3) & _NIBBLE_MASK) | _BF16_ONE_PAIR)
                for p in range_constexpr(1, 4):
                    words.append(((w >> (4 * p - 3)) & _NIBBLE_MASK) | _BF16_ONE_PAIR)
            return Vector.from_elements(words, fx.Int32).bitcast(fx.Int16)

        def wmma(a, b, acc):
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a, b, acc).result

        # The loads of the next _LOOKAHEAD K-steps stay in flight while a step
        # is computed; across groups they are carried as loop state.
        n_ksteps = K // 16
        n_step_loads = 2 + REG_N
        n_iter = block_valid.select(fx.Int32(n_groups), fx.Int32(0))
        init = [zero for _ in range_constexpr(REG_N)]
        for d in range_constexpr(_LOOKAHEAD):
            init = init + load_step(fx.Int32(d % n_ksteps))
        init = init + load_coef(fx.Int32(0))
        for i, state in range(0, n_iter, 1, init=init):  # type: ignore[call-overload]
            tot = list(state[:REG_N])
            queue = list(state[REG_N:-REG_N])
            coef = list(state[-REG_N:])
            g = fx.Int32(i)
            # Loads past the end repeat the last ones, which are cached.
            coef_next = load_coef((g + 1 < n_groups).select(g + 1, g))

            acc = [zero for _ in range_constexpr(REG_N)]
            xsum = zero
            for step in range_constexpr(steps):
                ahead = g * steps + (step + _LOOKAHEAD)
                ahead = (ahead < n_ksteps).select(ahead, fx.Int32(n_ksteps - 1))
                queue = queue + load_step(ahead)
                cur = queue[:n_step_loads]
                queue = queue[n_step_loads:]
                a = cur[0].shuffle(cur[1], list(range(8))).bitcast(fx.Int16)
                xsum = wmma(a, ones, xsum)
                for rn in range_constexpr(REG_N):
                    wv = cur[2 + rn]
                    acc[rn] = wmma(a, unpack(wv[0], wv[1]), acc[rn])
            for rn in range_constexpr(REG_N):
                s16, szp = _scale_terms(coef[rn], coef_pairs)
                s16 = fx.full(8, s16, fx.Float32)
                szp = fx.full(8, szp, fx.Float32)
                tot[rn] = math.fma(acc[rn], s16, math.fma(xsum, szp, tot[rn]))
            results = yield tot + queue + coef_next
        tot = list(results[:REG_N])
        if silu_mul:
            tot = [
                tot[rn] / (1.0 + math.exp(-tot[rn])) * tot[rn + out_reg]
                for rn in range_constexpr(out_reg)
            ]

        for si in range_constexpr(8):
            out_rid = buffer_ops.buffer_load(
                ids_rsrc, row0 + 2 * si + klane, vec_width=1, dtype=fx.Int32
            )
            # RDNA buffer resources are not bounds checked, so padding rows
            # have to be skipped rather than masked.
            if block_valid & (out_rid < n_rows):
                out_base = out_rid * n_out + tile_n * (16 * out_reg) + lane16
                for rn in range_constexpr(out_reg):
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


def _copy_to_lds(lds, rsrc, n_dwords: int, lane, base=0, lds0=0) -> None:
    """Copy ``n_dwords`` of a buffer from dword ``base`` to LDS dword ``lds0``,
    4 dwords per lane and round."""
    import flydsl.expr as fx
    from flydsl.expr import arith, buffer_ops, range_constexpr, vector
    from flydsl.expr.typing import T

    for j in range_constexpr(-(-n_dwords // 128)):
        idx = lane * 4 + j * 128
        idx = (idx < n_dwords).select(idx, fx.Int32(n_dwords - 4))
        v = buffer_ops.buffer_load(rsrc, base + idx, vec_width=4, dtype=fx.Int32)
        vector.store(v, lds, [arith.index_cast(T.index, lds0 + idx)])


@functools.cache
def _build_gemm_row(
    K: int,
    N: int,
    x_row_div: int,
    group: int,
    silu_mul: bool,
    w_expert: int,
    c_expert: int,
    coef_pairs: bool,
):
    """``out[r] = W[expert_ids[r]] @ x[r // x_row_div]``, one row per block.

    A wave needs each weight in two lanes, as WMMA operands are laid out, but
    it loads a group of K-steps with 32 distinct lanes (512 B per load) into
    LDS and reads the operands from there. The activations are in LDS too.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, math, range_constexpr, rocdl, vector
    from flydsl.expr.typing import T, Vector
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

    assert K % group == 0 and group % 16 == 0 and N % TILE_N == 0
    n_groups = K // group
    steps = group // 16
    k_dwords = K // 2
    w_step = 16 * 2 * REG_N
    chunk = steps * w_step  # dwords of one group of one tile
    w_tile = _tile_stride(_w_tile_dwords(K))
    # Coefficient offsets are in float16 elements.
    cw = 2 if coef_pairs else 1
    c_group = 16 * REG_N * cw
    c_tile = 2 * _tile_stride(_coef_tile_dwords(K, group, coef_pairs))
    c_expert *= 2
    out_reg = REG_N // 2 if silu_mul else REG_N
    n_out = N // 2 if silu_mul else N
    allocator = SmemAllocator(None, arch=get_rocm_arch(), global_sym_name="moe_row")
    lds_wave = k_dwords + chunk  # LDS dwords per wave
    # Waves per workgroup. GEMM1 of a token runs next to another kernel of the
    # layer, and workgroups of several waves keep their share of the GPU.
    wg = next(
        w
        for w in (8, 4, 2, 1)
        if (N // TILE_N) % w == 0 and lds_wave * 4 * w <= _LDS_BYTES
    )
    allocator.ptr = lds_wave * 4 * wg

    @flyc.kernel(known_block_size=[32 * wg, 1, 1])
    def row_kernel(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_expert_ids: fx.Tensor,
    ):
        lane = gpu.thread_idx.x % 32
        wave = gpu.thread_idx.x // 32
        tile_n = gpu.block_idx.x * wg + wave
        row = gpu.block_idx.y
        lane16 = lane % 16
        klane = lane // 16

        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_coef, max_size=True)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=True)

        # Rows without an expert (id -1) run zero groups and store nothing.
        expert = buffer_ops.buffer_load(eid_rsrc, row, vec_width=1, dtype=fx.Int32)
        valid = expert >= 0
        expert = valid.select(expert, fx.Int32(0))
        w_base = expert * w_expert + tile_n * w_tile
        c_base = expert * c_expert + tile_n * c_tile + lane16 * cw

        lds = SmemPtr(allocator.get_base(), 0, T.i32, shape=(lds_wave * wg,)).get()
        lds0 = wave * lds_wave
        _copy_to_lds(lds, x_rsrc, k_dwords, lane, (row // x_row_div) * k_dwords, lds0)

        def lds_at(dword):
            return [arith.index_cast(T.index, lds0 + dword)]

        def load_chunk(start, spread):
            """Loads of one group; with ``spread`` 0 all lanes load the same
            16 B, for the group after the last."""
            return [
                buffer_ops.buffer_load(
                    w_rsrc,
                    w_base + start + (lane * 4 + s * w_step) * spread,
                    vec_width=4,
                    dtype=fx.Int32,
                    cache_modifier=_NON_TEMPORAL,
                )
                for s in range_constexpr(steps)
            ]

        def load_coef(g):
            return [
                buffer_ops.buffer_load(
                    c_rsrc,
                    c_base + g * c_group + rn * (16 * cw),
                    vec_width=cw,
                    dtype=fx.Float16,
                )
                for rn in range_constexpr(REG_N)
            ]

        def unpack(wv):
            words = []
            for i in range_constexpr(2):
                w = wv[i]
                words.append(((w << 3) & _NIBBLE_MASK) | _BF16_ONE_PAIR)
                for p in range_constexpr(1, 4):
                    words.append(((w >> (4 * p - 3)) & _NIBBLE_MASK) | _BF16_ONE_PAIR)
            return Vector.from_elements(words, fx.Int32).bitcast(fx.Int16)

        def wmma(a, b, acc):
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a, b, acc).result

        ones = fx.full(16, _BF16_ONE_PAIR & 0xFFFF, fx.Int16)
        zero = fx.full(8, 0.0, fx.Float32)
        n_iter = valid.select(fx.Int32(n_groups), fx.Int32(0))
        init = [zero for _ in range_constexpr(REG_N)]
        init = init + load_chunk(fx.Int32(0), fx.Int32(1)) + load_coef(fx.Int32(0))
        for i, state in range(0, n_iter, 1, init=init):  # type: ignore[call-overload]
            tot = list(state[:REG_N])
            cur = list(state[REG_N:-REG_N])
            coef = list(state[-REG_N:])
            g = fx.Int32(i)
            for s in range_constexpr(steps):
                vector.store(cur[s], lds, lds_at(lane * 4 + (k_dwords + s * w_step)))
            more = g + 1 < n_groups
            nxt = load_chunk(
                more.select((g + 1) * chunk, fx.Int32(0)),
                more.select(fx.Int32(1), fx.Int32(0)),
            )
            coef_next = load_coef(more.select(g + 1, g))

            acc = [zero for _ in range_constexpr(REG_N)]
            xsum = zero
            for s in range_constexpr(steps):
                a = vector.load_op(
                    T.vec(8, T.i32), lds, lds_at((g * steps + s) * 8)
                ).bitcast(fx.Int16)
                xsum = wmma(a, ones, xsum)
                for rn in range_constexpr(REG_N):
                    wv = vector.load_op(
                        T.vec(2, T.i32),
                        lds,
                        lds_at(lane16 * 2 + (k_dwords + s * w_step + rn * 32)),
                    )
                    acc[rn] = wmma(a, unpack(wv), acc[rn])
            for rn in range_constexpr(REG_N):
                s16, szp = _scale_terms(coef[rn], coef_pairs)
                s16 = fx.full(8, s16, fx.Float32)
                szp = fx.full(8, szp, fx.Float32)
                tot[rn] = math.fma(acc[rn], s16, math.fma(xsum, szp, tot[rn]))
            results = yield tot + nxt + coef_next
        tot = list(results[:REG_N])
        if silu_mul:
            tot = [
                tot[rn] / (1.0 + math.exp(-tot[rn])) * tot[rn + out_reg]
                for rn in range_constexpr(out_reg)
            ]

        # Row 0 of the WMMA result is element 0 of the first 16 lanes.
        if valid & (klane == 0):
            out_base = row * n_out + tile_n * (16 * out_reg) + lane16
            for rn in range_constexpr(out_reg):
                buffer_ops.buffer_store(
                    tot[rn][0].to(fx.BFloat16), out_rsrc, out_base + rn * 16
                )

    @flyc.jit
    def launch(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        row_kernel(arg_out, arg_x, arg_w, arg_coef, arg_expert_ids).launch(
            grid=(N // TILE_N // wg, n_rows, 1), block=(32 * wg, 1, 1), stream=stream
        )

    return launch


@functools.cache
def _build_gemm_reduce(
    K: int,
    N: int,
    group: int,
    top_k: int,
    w_expert: int,
    c_expert: int,
    coef_pairs: bool,
):
    """One token: ``out[n] = sum_e weight_e * (W[expert_e] @ x[e])[n]``.

    Row ``2 e`` of the WMMA A operand holds ``x[e]``, so one WMMA per expert
    yields its 16 output columns in the lanes of the first half-wave and a
    single WMMA against ones yields all activation sums.

    ``x`` is copied to LDS first. Buffer loads complete in order, so loading
    ``x`` when it is needed would wait for the weight loads issued ahead.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, math, range_constexpr, rocdl, vector
    from flydsl.expr.typing import T, Vector
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

    assert K % group == 0 and group % 16 == 0 and N % TILE_N == 0
    assert top_k <= MAX_REDUCE_ROWS
    n_groups = K // group
    steps = group // 16
    n_ksteps = K // 16
    k_dwords = K // 2
    w_step = 16 * 2 * REG_N
    w_tile = _tile_stride(_w_tile_dwords(K))
    # Coefficient offsets are in float16 elements.
    cw = 2 if coef_pairs else 1
    c_group = 16 * REG_N * cw
    c_tile = 2 * _tile_stride(_coef_tile_dwords(K, group, coef_pairs))
    c_expert *= 2
    x_dwords = top_k * k_dwords
    allocator = SmemAllocator(None, arch=get_rocm_arch(), global_sym_name="moe_x")
    allocator.ptr = x_dwords * 4

    @flyc.kernel
    def reduce_kernel(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_weights: fx.Tensor,
    ):
        lane = gpu.thread_idx.x
        col16 = gpu.block_idx.x
        lane16 = lane % 16
        klane = lane // 16

        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_coef, max_size=True)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=True)
        tw_rsrc = buffer_ops.create_buffer_resource(arg_weights, max_size=True)

        lds = SmemPtr(allocator.get_base(), 0, T.i32, shape=(x_dwords,)).get()
        _copy_to_lds(lds, x_rsrc, x_dwords, lane)
        x_row = lane16 // 2
        x_base = (x_row < top_k).select(x_row, fx.Int32(0)) * k_dwords
        tile_n = col16 // REG_N
        rn = col16 % REG_N
        weight = []
        w_base = []
        c_base = []
        for e in range_constexpr(top_k):
            expert = buffer_ops.buffer_load(
                eid_rsrc, fx.Int32(e), vec_width=1, dtype=fx.Int32
            )
            valid = expert >= 0
            tw = buffer_ops.buffer_load(
                tw_rsrc, fx.Int32(e), vec_width=1, dtype=fx.Float32
            )
            # Experts with id -1 contribute nothing (the engine warm-up).
            weight.append(valid.select(tw, fx.Float32(0.0)))
            expert = valid.select(expert, fx.Int32(0))
            w_base.append(expert * w_expert + tile_n * w_tile + rn * 32 + lane16 * 2)
            c_base.append(expert * c_expert + tile_n * c_tile + (rn * 16 + lane16) * cw)

        ones = fx.full(16, _BF16_ONE_PAIR & 0xFFFF, fx.Int16)
        zero = fx.full(8, 0.0, fx.Float32)

        def load_w(kstep):
            return [
                buffer_ops.buffer_load(
                    w_rsrc,
                    w_base[e] + kstep * w_step,
                    vec_width=2,
                    dtype=fx.Int32,
                    cache_modifier=_NON_TEMPORAL,
                )
                for e in range_constexpr(top_k)
            ]

        def load_coef(g):
            return [
                buffer_ops.buffer_load(
                    c_rsrc, c_base[e] + g * c_group, vec_width=cw, dtype=fx.Float16
                )
                for e in range_constexpr(top_k)
            ]

        def unpack(wv):
            words = []
            for i in range_constexpr(2):
                w = wv[i]
                words.append(((w << 3) & _NIBBLE_MASK) | _BF16_ONE_PAIR)
                for p in range_constexpr(1, 4):
                    words.append(((w >> (4 * p - 3)) & _NIBBLE_MASK) | _BF16_ONE_PAIR)
            return Vector.from_elements(words, fx.Int32).bitcast(fx.Int16)

        def wmma(a, b, acc):
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a, b, acc).result

        init = [fx.Float32(0.0)]
        for d in range_constexpr(_LOOKAHEAD):
            init = init + load_w(fx.Int32(d % n_ksteps))
        init = init + load_coef(fx.Int32(0))
        for i, state in range(0, n_groups, 1, init=init):  # type: ignore[call-overload]
            tot = fx.Float32(state[0])
            queue = list(state[1:-top_k])
            coef = list(state[-top_k:])
            g = fx.Int32(i)
            coef_next = load_coef((g + 1 < n_groups).select(g + 1, g))

            acc = [zero for _ in range_constexpr(top_k)]
            xsum = zero
            for step in range_constexpr(steps):
                ahead = g * steps + (step + _LOOKAHEAD)
                ahead = (ahead < n_ksteps).select(ahead, fx.Int32(n_ksteps - 1))
                queue = queue + load_w(ahead)
                a = vector.load_op(
                    T.vec(8, T.i32),
                    lds,
                    [arith.index_cast(T.index, x_base + (g * steps + step) * 8)],
                ).bitcast(fx.Int16)
                xsum = wmma(a, ones, xsum)
                for e in range_constexpr(top_k):
                    acc[e] = wmma(a, unpack(queue[e]), acc[e])
                queue = queue[top_k:]
            for e in range_constexpr(top_k):
                s16, szp = _scale_terms(coef[e], coef_pairs)
                val = math.fma(acc[e][e], s16, xsum[e] * szp)
                tot = math.fma(val, weight[e], tot)
            results = yield [tot] + queue + coef_next
        tot = fx.Float32(results[0])

        if klane == 0:
            buffer_ops.buffer_store(tot.to(fx.BFloat16), out_rsrc, col16 * 16 + lane16)

    @flyc.jit
    def launch(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_weights: fx.Tensor,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        reduce_kernel(
            arg_out, arg_x, arg_w, arg_coef, arg_expert_ids, arg_weights
        ).launch(grid=(N // 16, 1, 1), block=(32, 1, 1), stream=stream)

    return launch


def flydsl_rdna_moe_gemm_reduce(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    coef: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    group_size: int,
    coef_pairs: bool = True,
) -> None:
    """One token: ``out[0] = sum_e weights[e] * W[expert_ids[e]] @ x[e]``.

    ``weights`` is float32; at most ``MAX_REDUCE_ROWS`` experts. ``coef_pairs``
    is False for weights packed without zero points.
    """
    _check_coef(coef, x.shape[1], group_size, coef_pairs)
    launch = _build_gemm_reduce(
        x.shape[1],
        out.shape[1],
        group_size,
        x.shape[0],
        _expert_stride_dwords(w),
        _expert_stride_dwords(coef),
        coef_pairs,
    )
    launch(out, x, w, coef, expert_ids, weights, torch.cuda.current_stream())


def flydsl_rdna_moe_gemm(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    coef: torch.Tensor,
    sorted_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_valid: torch.Tensor | None,
    x_row_div: int,
    group_size: int,
    silu_mul: bool = False,
    coef_pairs: bool = True,
) -> None:
    """Run ``out[r] = W[expert(r)] @ x[r // x_row_div]`` for all routed rows.

    ``sorted_ids`` / ``expert_ids`` / ``num_valid`` come from
    ``moe_align_block_size`` with block size ``BLOCK_M``. When ``sorted_ids``
    is None, ``expert_ids[r]`` is the expert of row ``r`` and no grouping is
    needed; this is the cheap routing for a few rows with distinct experts.

    With ``silu_mul`` the weights are in ``gate_up_tile_order`` and ``out``
    receives ``silu(gate) * up``, half as many columns as the weights have.
    ``coef_pairs`` is False for weights packed without zero points.
    """
    _check_coef(coef, x.shape[1], group_size, coef_pairs)
    n_rows, N = out.shape
    if silu_mul:
        N *= 2
    strides = (_expert_stride_dwords(w), _expert_stride_dwords(coef), coef_pairs)
    stream = torch.cuda.current_stream()
    if sorted_ids is None:
        launch = _build_gemm_row(
            x.shape[1], N, x_row_div, group_size, silu_mul, *strides
        )
        launch(out, x, w, coef, expert_ids, expert_ids.numel(), stream)
        return
    launch = _build_gemm(x.shape[1], N, x_row_div, group_size, silu_mul, *strides)
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
        stream,
    )
