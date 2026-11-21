# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from SGLang's compact A4 MoE kernels
# (python/sglang/kernels/ops/moe/mxfp4_compact_a4_moe.py, Apache-2.0).
"""Compact MXFP4 experts for small Muse Spark decode calls (ROCm gfx950).

For the `[T, top_k]` routes of T MXFP4-quantized rows over this rank's local
experts, in three launches:

1. `_route_quant_kernel`: per local expert, its routes in token order, placed in
   `BM`-row blocks after the lower experts' blocks (unused blocks get expert -1),
   and the per-row MXFP4 quant of the T rows (the rounding of `sort_quant`);
2. `_expert_kernel` (gate/up): `dot_scaled` gate and up GEMMs, SiLU-and-mul and
   the MXFP4 quant of each route's product (the bf16 rounding and the quant of
   the bf16-output FlyDSL stage 1 followed by the stage-2 input quant);
3. `_expert_kernel` (down): one unweighted BF16 row per local (token, slot).

Every program reads one N tile of its block's expert weights once, streamed
past the caches. Weights and scales are in AITER's preshuffled MXFP4 layout, so
the AITER expert stages and this path share one copy.

With expert replicas (`ReplicaTable`), `_route_quant_replica_kernel` replaces
step 1: it also decides, from the step's routes, which replicated experts run
on their copy instead of their owner (`_replica_assignment`), and the expert
kernels read replica slots from separate replica weights.
"""

import functools
from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton

# AITER's Triton MXFP4 quant (private); without it this path is off.
try:
    from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

    if not {"SCALING_MODE", "USE_ASM"} <= set(
        getattr(_mxfp4_quant_op, "arg_names", ())
    ):
        raise ImportError
except ImportError:
    _mxfp4_quant_op = None

# Largest call (tokens) that takes this path.
MAX_TOKENS = 448
TOP_K = 8

# Relative cost of an expert on a rank, by blocks of its routes: the first
# block streams the expert's weights; a further block of the same expert
# rereads them, partly from the caches.
REPLICA_COST_UNIT = 4
REPLICA_COST_EXTRA_BLOCK = 3
# Single-move rounds of the per-step replica assignment.
REPLICA_ROUNDS = 3
# The assignment packs per-rank costs in 16-bit fields of two int64 words.
REPLICA_MAX_RANKS = 8


class TileConfig(NamedTuple):
    """Rows per expert block, and per stage: N tile, K tile (elements), warps,
    pipeline stages."""

    block_m: int
    bn1: int
    bk1: int
    warps1: int
    stages1: int
    bn2: int
    bk2: int
    warps2: int
    stages2: int


# By the largest token count each row covers (gfx950, Muse Spark expert
# shapes). With few routes per touched expert, 16-row blocks and narrow N tiles
# spread each expert's weights over more programs; with more routes, 32-row
# blocks keep an expert's weights from being read twice.
_TILE_TABLE: tuple[tuple[int, TileConfig], ...] = (
    (64, TileConfig(16, 32, 1024, 2, 2, 32, 1024, 2, 2)),
    (MAX_TOKENS, TileConfig(32, 64, 512, 4, 2, 64, 512, 4, 2)),
)


def tile_config(num_tokens: int) -> TileConfig:
    for last, cfg in _TILE_TABLE:
        if num_tokens <= last:
            return cfg
    raise ValueError(f"no compact MoE tiles for {num_tokens} tokens")


@functools.cache
def _on_gfx950() -> bool:
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def supports_compact_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> bool:
    """Whether `compact_moe_experts` takes these bf16 rows `x` (`[T, H]`), their
    int32 routes `topk_ids` (`[T, top_k]`) and preshuffled MXFP4 expert weights
    (`w13`: `[E, 2 * I, H / 2]`, `w2`: `[E, H, I / 2]`, E8M0 scales per 32
    columns)."""
    if x.dim() != 2 or topk_ids.dim() != 2 or w13.dim() != 3 or w2.dim() != 3:
        return False
    num_tokens, hidden = x.shape
    e, inter = w13.shape[0], w2.shape[2] * 2
    return (
        _mxfp4_quant_op is not None
        and _on_gfx950()
        and 0 < num_tokens <= MAX_TOKENS
        and topk_ids.shape == (num_tokens, TOP_K)
        and x.dtype == torch.bfloat16
        and topk_ids.dtype == torch.int32
        and x.is_contiguous()
        and topk_ids.is_contiguous()
        and hidden % 1024 == 0
        and inter % 1024 == 0
        and 0 < e <= 128
        and e & (e - 1) == 0
        and w13.shape == (e, 2 * inter, hidden // 2)
        and w2.shape == (e, hidden, inter // 2)
        and w13_scale.shape == (e, 2 * inter, hidden // 32)
        and w2_scale.shape == (e, hidden, inter // 32)
        and getattr(w13, "is_shuffled", False)
        and getattr(w2, "is_shuffled", False)
        and all(
            t.element_size() == 1 and t.is_contiguous()
            for t in (w13, w2, w13_scale, w2_scale)
        )
    )


@triton.jit(do_not_specialize=["M", "NB"])
def _route_quant_kernel(
    ids_ptr,
    lmap_ptr,
    sorted_ptr,
    block_expert_ptr,
    x_ptr,
    q_ptr,
    qs_ptr,
    M,
    NB,
    LE: tl.constexpr,
    S: tl.constexpr,
    PER_E: tl.constexpr,
    BM: tl.constexpr,
    NBB: tl.constexpr,
    TOPK: tl.constexpr,
    D: tl.constexpr,
    DC: tl.constexpr,
):
    """Programs `[0, LE)`: local expert `pid`'s routes; the rest: one `DC`-column
    chunk of one row's MXFP4 quant."""
    pid = tl.program_id(0)
    if pid < LE:
        e = pid
        le = tl.arange(0, LE)
        s = tl.arange(0, S)
        valid = s < M * TOPK
        gid = tl.load(ids_ptr + s, mask=valid, other=-1)
        lid = tl.load(lmap_ptr + gid, mask=valid & (gid >= 0), other=-1)
        # Other ranks' routes and the padding go to bins [LE, 2 LE) (a masked
        # tl.histogram miscounts on ROCm Triton).
        h2 = tl.reshape(tl.histogram(tl.where(lid >= 0, lid, LE), 2 * LE), [2, LE])
        counts = tl.sum(tl.where(tl.arange(0, 2)[:, None] == 0, h2, 0), axis=0)
        blocks_all = (counts + BM - 1) // BM
        first = tl.sum(tl.where(le < e, blocks_all, 0), axis=0)
        if e == LE - 1:
            b = tl.arange(0, NBB)
            tl.store(
                block_expert_ptr + b,
                -1,
                mask=(b >= tl.sum(blocks_all, axis=0)) & (b < NB),
            )
        hit = lid == e
        count = tl.sum(hit.to(tl.int32), axis=0)
        blocks = (count + BM - 1) // BM
        if blocks > 0:
            eb = tl.arange(0, PER_E)
            tl.store(block_expert_ptr + first + eb, e, mask=eb < blocks)
            rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
            base = first * BM
            c = tl.arange(0, PER_E * BM)
            # Padding and routes write disjoint slots of the expert's blocks.
            tl.store(
                sorted_ptr + base + c, M * TOPK, mask=(c < blocks * BM) & (c >= count)
            )
            tl.store(sorted_ptr + base + rank, s, mask=hit)
    else:
        chunks: tl.constexpr = D // DC
        row = (pid - LE) // chunks
        chunk = (pid - LE) % chunks
        col = chunk * DC + tl.arange(0, DC)
        x = tl.load(x_ptr + row * D + col).to(tl.float32)
        q, qs = _quant_rows(tl.reshape(x, [1, DC]), 1, DC)
        tl.store(
            q_ptr + row * (D // 2) + chunk * (DC // 2) + tl.arange(0, DC // 2),
            tl.reshape(q, [DC // 2]),
        )
        tl.store(
            qs_ptr + row * (D // 32) + chunk * (DC // 32) + tl.arange(0, DC // 32),
            tl.reshape(qs, [DC // 32]),
        )


class ReplicaTable(NamedTuple):
    """One layer's expert replicas on this rank (see `replica_table`).

    `table` (int32) holds, for E global experts, NE replica entries (padded to
    NEP) and NS local slots (the rank's own experts, then its replica slots;
    padded to NSP):

    * `owner[E]`: the rank that owns each expert;
    * `entry[E]`: each expert's replica entry, -1 for experts without a copy;
    * per entry (-1 ranks for padding): `entry_owner[NEP]`, `entry_copy[NEP]`
      (the rank holding the copy), `entry_local[NEP]` (the expert's local
      slot on this rank, as owner or copy, else -1), `entry_expert[NEP]`;
    * per local slot: `slot_expert[NSP]` (-1 for padding) and
      `slot_entry[NSP]` (the replica entry of the slot's expert, or -1).

    Entries are in priority order: when two moves compete for a rank, the
    lower entry wins.
    """

    table: torch.Tensor
    ranks: int
    rank: int
    num_entries: int
    num_slots: int
    entries_padded: int
    slots_padded: int


def replica_table(
    owner: list[int],
    entries: list[tuple[int, int, int]],
    rank: int,
    ranks: int,
    slot_expert: list[int],
    device: torch.device,
) -> ReplicaTable:
    """The `ReplicaTable` of one layer on `rank`. `entries`: `(expert, copy rank,
    copy slot)` in priority order, identical on every rank; `slot_expert`: the
    global expert of each local slot (own experts, then replica slots, each
    expert at most once)."""
    assert 1 < ranks <= REPLICA_MAX_RANKS
    num_experts = len(owner)
    nep = max(16, triton.next_power_of_2(max(1, len(entries))))
    nsp = triton.next_power_of_2(len(slot_expert))
    entry = [-1] * num_experts
    for k, (e, _, _) in enumerate(entries):
        entry[e] = k
    pad = nep - len(entries)
    fields = [
        owner,
        entry,
        [owner[e] for e, _, _ in entries] + [-1] * pad,
        [c for _, c, _ in entries] + [-1] * pad,
        [
            slot_expert.index(e) if owner[e] == rank or c == rank else -1
            for e, c, _ in entries
        ]
        + [-1] * pad,
        [e for e, _, _ in entries] + [0] * pad,
        slot_expert + [-1] * (nsp - len(slot_expert)),
        [entry[e] if e >= 0 else -1 for e in slot_expert]
        + [-1] * (nsp - len(slot_expert)),
    ]
    table = torch.tensor(sum(fields, []), dtype=torch.int32, device=device)
    return ReplicaTable(table, ranks, rank, len(entries), len(slot_expert), nep, nsp)


# Per-rank quantities of up to 8 ranks are packed in two int64 words, 16 bits
# per rank (ranks 0-3, then 4-7), so that per-rank sums and prefix sums over the
# replica entries are single 1-D reductions and scans.
_FIELD = tl.constexpr(16)


@triton.jit
def _pack(v, rank, HI: tl.constexpr):
    """`v` in the 16-bit field of `rank` (ranks 4-7 when HI, else 0-3)."""
    shift = (rank - 4 * HI) * _FIELD
    inside = (rank >= 4 * HI) & (rank < 4 * HI + 4)
    return tl.where(inside, v.to(tl.int64) << shift.to(tl.int64), 0)


@triton.jit
def _field(lo, hi, rank, TWO: tl.constexpr):
    """The 16-bit field of `rank` from the packed pair `(lo, hi)`."""
    word = tl.where(rank >= 4, hi, lo) if TWO else lo
    shift = (tl.maximum(rank, 0) % 4) * _FIELD
    return ((word >> shift.to(tl.int64)) & 0xFFFF).to(tl.int32)


@triton.jit
def _packed_sum(v, rank, TWO: tl.constexpr):
    lo = tl.sum(_pack(v, rank, 0), axis=0)
    hi = tl.sum(_pack(v, rank, 1), axis=0) if TWO else lo * 0
    return lo, hi


@triton.jit
def _excl_prefix(v, rank, TWO: tl.constexpr):
    """Per entry: the sum of `v` over earlier entries of the same rank."""
    lo = tl.cumsum(_pack(v, rank, 0), axis=0)
    hi = tl.cumsum(_pack(v, rank, 1), axis=0) if TWO else lo
    return _field(lo, hi, rank, TWO) - v


@triton.jit
def _scalar_field(lo, hi, p: tl.constexpr):
    word = hi if p >= 4 else lo
    return ((word >> ((p % 4) * _FIELD)) & 0xFFFF).to(tl.int32)


@triton.jit
def _max_field(lo, hi, P: tl.constexpr):
    """The largest of the P packed per-rank values (scalar)."""
    top = _scalar_field(lo, hi, 0)
    for p in tl.static_range(1, P):
        top = tl.maximum(top, _scalar_field(lo, hi, p))
    return top


@triton.jit
def _replica_assignment(
    gid,
    valid,
    tab_ptr,
    E: tl.constexpr,
    NEP: tl.constexpr,
    P: tl.constexpr,
    BM: tl.constexpr,
    UNIT: tl.constexpr,
    EXTRA: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    """The step's per-expert route counts and each replica entry's placement
    (1: its copy computes the expert's routes), identical in every program of
    every rank.

    Cost of an expert on a rank: `UNIT` for its first block of routes plus
    `EXTRA` per further block (its weights are read once per block).

    First, all at once: a selected replicated expert on an owner above the mean
    cost moves to its copy if the copy's rank is below the mean, while the
    earlier such candidates for the copy's rank stay within its room below the
    mean. Then `ROUNDS` single moves: the first entry on the busiest rank whose
    move to its other holder lowers that pair's larger cost."""
    TWO: tl.constexpr = P > 4
    h2 = tl.reshape(tl.histogram(tl.where(valid & (gid >= 0), gid, E), 2 * E), [2, E])
    cnt = tl.sum(tl.where(tl.arange(0, 2)[:, None] == 0, h2, 0), axis=0)
    e = tl.arange(0, E)
    owner = tl.load(tab_ptr + e)
    w = tl.where(cnt > 0, UNIT + EXTRA * ((cnt + BM - 1) // BM - 1), 0)
    Llo, Lhi = _packed_sum(w, owner, TWO)
    tot = tl.sum(w, axis=0)
    k = tl.arange(0, NEP)
    base = tab_ptr + 2 * E
    eo = tl.load(base + k)
    ec = tl.load(base + NEP + k)
    ex = tl.load(base + 3 * NEP + k)
    we = tl.where(eo >= 0, tl.gather(w, ex, 0), 0)
    lo = _field(Llo, Lhi, eo, TWO)
    lc = _field(Llo, Lhi, ec, TWO)
    cand = (we > 0) & (lo * P > tot) & (lc * P < tot)
    wc = tl.where(cand, we, 0)
    take = cand & (_excl_prefix(wc, ec, TWO) * P <= tot - lc * P)
    mv = tl.where(take, we, 0)
    dlo, dhi = _packed_sum(mv, ec, TWO)
    slo, shi = _packed_sum(mv, eo, TWO)
    Llo, Lhi = Llo + dlo - slo, Lhi + dhi - shi
    on_copy = take.to(tl.int32)
    for _ in tl.static_range(ROUNDS):
        cur = tl.where(on_copy != 0, ec, eo)
        oth = tl.where(on_copy != 0, eo, ec)
        lcur = _field(Llo, Lhi, cur, TWO)
        top = _max_field(Llo, Lhi, P)
        c1 = (we > 0) & (lcur == top) & (_field(Llo, Lhi, oth, TWO) + we < lcur)
        # The first candidate, with its ranks and cost, in one reduction.
        key = (k << 19) | (cur << 16) | (oth << 13) | tl.minimum(we, 8191)
        pick = tl.min(tl.where(c1, key, 1 << 30), axis=0)
        found = pick < (1 << 30)
        kp = pick >> 19
        wp = tl.where(found, pick & 8191, 0)
        cp = (pick >> 16) & 7
        op = (pick >> 13) & 7
        Llo = Llo + _pack(wp, op, 0) - _pack(wp, cp, 0)
        if TWO:
            Lhi = Lhi + _pack(wp, op, 1) - _pack(wp, cp, 1)
        on_copy = tl.where(found & (k == kp), 1 - on_copy, on_copy)
    return cnt, on_copy


@triton.jit(do_not_specialize=["M", "NB"])
def _route_quant_replica_kernel(
    ids_ptr,
    tab_ptr,
    index_ptr,
    pad_ptr,
    sorted_ptr,
    block_expert_ptr,
    x_ptr,
    q_ptr,
    qs_ptr,
    M,
    NB,
    E: tl.constexpr,
    LE: tl.constexpr,
    NS: tl.constexpr,
    NSP: tl.constexpr,
    NEP: tl.constexpr,
    P: tl.constexpr,
    RANK: tl.constexpr,
    UNIT: tl.constexpr,
    EXTRA: tl.constexpr,
    ROUNDS: tl.constexpr,
    HAS_PAD: tl.constexpr,
    S: tl.constexpr,
    PER_E: tl.constexpr,
    BM: tl.constexpr,
    NBB: tl.constexpr,
    TOPK: tl.constexpr,
    D: tl.constexpr,
    DC: tl.constexpr,
):
    """`_route_quant_kernel` over the rank's own experts and its replica slots.
    Programs `[0, NS)`: local slot `pid`'s routes under the step's replica
    assignment (program 0 also rewrites the replicated experts' entries of the
    `[E]` local index `index_ptr` for the finalize); the rest: one `DC`-column
    chunk of one row's MXFP4 quant. HAS_PAD: rows flagged in `pad_ptr` (one
    byte per row) are computed, but their routes do not count toward the
    assignment."""
    pid = tl.program_id(0)
    if pid < NS:
        s = tl.arange(0, S)
        valid = s < M * TOPK
        gid = tl.load(ids_ptr + s, mask=valid, other=-1)
        counted = valid
        if HAS_PAD:
            pad = tl.load(pad_ptr + s // TOPK, mask=valid, other=1)
            counted = valid & (pad == 0)
        cnt, on_copy = _replica_assignment(
            gid, counted, tab_ptr, E, NEP, P, BM, UNIT, EXTRA, ROUNDS
        )
        if HAS_PAD:
            # The slots' route counts for the block layout include padding rows.
            cnt = tl.sum(
                tl.where(
                    tl.arange(0, 2)[:, None] == 0,
                    tl.reshape(
                        tl.histogram(tl.where(valid & (gid >= 0), gid, E), 2 * E),
                        [2, E],
                    ),
                    0,
                ),
                axis=0,
            )
        sl = tl.arange(0, NSP)
        sbase = tab_ptr + 2 * E + 4 * NEP
        sx = tl.load(sbase + sl)
        sent = tl.load(sbase + NSP + sl)
        moved = tl.where(sent >= 0, tl.gather(on_copy, tl.maximum(sent, 0), 0), 0)
        scnt = tl.where(sx >= 0, tl.gather(cnt, tl.maximum(sx, 0), 0), 0)
        # Own slots run unless their expert moved; replica slots only if it did.
        act = (scnt > 0) & tl.where(sl >= LE, moved != 0, moved == 0)
        blocks_all = tl.where(act, (scnt + BM - 1) // BM, 0)
        first = tl.sum(tl.where(sl < pid, blocks_all, 0), axis=0)
        if pid == 0:
            # Only the replicated experts' entries of the local index change.
            k = tl.arange(0, NEP)
            ebase = tab_ptr + 2 * E
            eo = tl.load(ebase + k)
            ec = tl.load(ebase + NEP + k)
            eloc = tl.load(ebase + 2 * NEP + k)
            ex = tl.load(ebase + 3 * NEP + k)
            index = tl.where((eo == RANK) == (on_copy == 0), eloc, -1)
            tl.store(index_ptr + ex, index, mask=(eo == RANK) | (ec == RANK))
        if pid == NS - 1:
            b = tl.arange(0, NBB)
            tl.store(
                block_expert_ptr + b,
                -1,
                mask=(b >= tl.sum(blocks_all, axis=0)) & (b < NB),
            )
        mine = tl.sum(tl.where(sl == pid, sx, 0), axis=0)
        hit = (gid == mine) & (tl.sum(tl.where(sl == pid, act.to(tl.int32), 0)) > 0)
        count = tl.sum(hit.to(tl.int32), axis=0)
        blocks = (count + BM - 1) // BM
        if blocks > 0:
            eb = tl.arange(0, PER_E)
            tl.store(block_expert_ptr + first + eb, pid, mask=eb < blocks)
            rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
            base = first * BM
            c = tl.arange(0, PER_E * BM)
            # Padding and routes write disjoint slots of the slot's blocks.
            tl.store(
                sorted_ptr + base + c, M * TOPK, mask=(c < blocks * BM) & (c >= count)
            )
            tl.store(sorted_ptr + base + rank, s, mask=hit)
    else:
        chunks: tl.constexpr = D // DC
        row = (pid - NS) // chunks
        chunk = (pid - NS) % chunks
        col = chunk * DC + tl.arange(0, DC)
        x = tl.load(x_ptr + row * D + col).to(tl.float32)
        q, qs = _quant_rows(tl.reshape(x, [1, DC]), 1, DC)
        tl.store(
            q_ptr + row * (D // 2) + chunk * (DC // 2) + tl.arange(0, DC // 2),
            tl.reshape(q, [DC // 2]),
        )
        tl.store(
            qs_ptr + row * (D // 32) + chunk * (DC // 32) + tl.arange(0, DC // 32),
            tl.reshape(qs, [DC // 32]),
        )


@triton.jit
def _load_w(W, n0, k0, KROW: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    """`[KB bytes, BN rows]` tile from row `n0`, byte `k0` of one expert's
    preshuffled `[N, KROW]` packed E2M1 rows: `[N / 16, KROW / 16, 16 rows,
    16 bytes]`, loaded as whole 16-row groups and permuted in registers."""
    g = tl.arange(0, BN // 16)
    kc = tl.arange(0, KB // 16)
    r = tl.arange(0, 16)
    b = tl.arange(0, 16)
    offset = (
        ((n0 // 16 + g) * (16 * KROW))[:, None, None, None]
        + ((k0 // 16 + kc) * 256)[None, :, None, None]
        + (r * 16)[None, None, :, None]
        + b[None, None, None, :]
    )
    x = tl.permute(tl.load(W + offset, cache_modifier=".cs"), (1, 3, 0, 2))
    return tl.reshape(x, (KB, BN))


@triton.jit
def _s_offset(n, c, SC: tl.constexpr):
    """`[N, scale col]` offsets in one expert's preshuffled `[N, SC]` E8M0
    scales: `[N / 32, SC / 8, 4 (c % 4), 16 (n % 16), 2 ((c % 8) // 4),
    2 ((n % 32) // 16)]`."""
    n2, c2 = n[:, None], c[None, :]
    return (
        ((((n2 // 32) * (SC // 8) + c2 // 8) * 4 + c2 % 4) * 16 + n2 % 16) * 4
        + ((c2 % 8) // 4) * 2
        + (n2 % 32) // 16
    )


@triton.jit
def _quant_rows(x, BM: tl.constexpr, BN: tl.constexpr):
    """MXFP4 quant of `[BM, BN]` fp32 rows per 32 columns, as sort_quant: E8M0
    scale 2^ceil(log2(amax / 6)) (all-zero blocks 2^-126), round-half-even
    E2M1 codes packed two per byte, the even column in the low nibble."""
    return _mxfp4_quant_op(x, BN, BM, 32, SCALING_MODE=1, USE_ASM=True)


@triton.jit
def _silu_mul_quant(gate, up, BM: tl.constexpr, BN: tl.constexpr):
    """SiLU(gate) * up, rounded to bf16 as a bf16 stage-1 output is, and its
    MXFP4 quant per 32 columns (`_quant_rows`, the stage-2 input quant of the
    bf16-output expert stages)."""
    # 1 / (1 + 2^(-gate * log2(e))) with the hardware exp2 and reciprocal, one
    # instruction per asm block so the compiler resolves their hazards.
    exp = tl.inline_asm_elementwise(
        "v_exp_f32 $0, $1",
        "=v,v",
        args=[gate * -1.4426950408889634],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    sig = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        "=v,v",
        args=[exp + 1.0],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    act = (gate * sig * up).to(tl.bfloat16).to(tl.float32)
    return _quant_rows(act, BM, BN)


@triton.jit(do_not_specialize=["rows"])
def _expert_kernel(
    a_ptr,
    as_ptr,
    w_ptr,
    ws_ptr,
    sorted_ptr,
    block_expert_ptr,
    out_ptr,
    os_ptr,
    rows,
    wr_ptr,
    wrs_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    GATE_UP: tl.constexpr,
    TOPK: tl.constexpr,
    LOCAL: tl.constexpr = 0,
):
    """Program `(n tile, block)`. GATE_UP: `a` holds one MXFP4 row per token and
    `w` `[E, 2 N, K / 2]` (gate rows, then up rows); the output is MXFP4 per
    route. Else `a` holds one MXFP4 row per route and the output is BF16.

    LOCAL > 0: slots from LOCAL on are rows of the replica weights `wr` /
    `wrs` (same layout)."""
    block, n_tile = tl.program_id(1), tl.program_id(0)
    expert = tl.load(block_expert_ptr + block)
    if expert >= 0:
        slot = tl.load(sorted_ptr + block * BM + tl.arange(0, BM))
        valid = slot < rows
        src = slot // TOPK if GATE_UP else slot
        n = n_tile * BN + tl.arange(0, BN)
        kp = tl.arange(0, BK // 2)
        ks = tl.arange(0, BK // 32)
        WN: tl.constexpr = 2 * N if GATE_UP else N
        w_e = w_ptr + expert.to(tl.int64) * (WN * (K // 2))
        ws_e = ws_ptr + expert.to(tl.int64) * (WN * (K // 32))
        if LOCAL > 0:  # noqa: SIM102 (a compile-time branch around a runtime one)
            if expert >= LOCAL:
                w_e = wr_ptr + (expert - LOCAL).to(tl.int64) * (WN * (K // 2))
                ws_e = wrs_ptr + (expert - LOCAL).to(tl.int64) * (WN * (K // 32))
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if GATE_UP:
            up = tl.zeros((BM, BN), dtype=tl.float32)
        for tile in range(K // BK):
            a = tl.load(
                a_ptr + src[:, None] * (K // 2) + tile * (BK // 2) + kp[None, :],
                mask=valid[:, None],
                other=0,
            )
            a_s = tl.load(
                as_ptr + src[:, None] * (K // 32) + tile * (BK // 32) + ks[None, :],
                mask=valid[:, None],
                other=127,
            )
            w = _load_w(w_e, n_tile * BN, tile * (BK // 2), K // 2, BN, BK // 2)
            w_s = tl.load(ws_e + _s_offset(n, tile * (BK // 32) + ks, K // 32))
            acc = tl.dot_scaled(a, a_s, "e2m1", w, w_s, "e2m1", acc)
            if GATE_UP:
                wu = _load_w(
                    w_e, n_tile * BN + N, tile * (BK // 2), K // 2, BN, BK // 2
                )
                wu_s = tl.load(ws_e + _s_offset(n + N, tile * (BK // 32) + ks, K // 32))
                up = tl.dot_scaled(a, a_s, "e2m1", wu, wu_s, "e2m1", up)
        if GATE_UP:
            q, q_s = _silu_mul_quant(acc, up, BM, BN)
            pn = n_tile * (BN // 2) + tl.arange(0, BN // 2)
            sn = n_tile * (BN // 32) + tl.arange(0, BN // 32)
            tl.store(
                out_ptr + slot[:, None] * (N // 2) + pn[None, :], q, mask=valid[:, None]
            )
            tl.store(
                os_ptr + slot[:, None] * (N // 32) + sn[None, :],
                q_s,
                mask=valid[:, None],
            )
        else:
            tl.store(
                out_ptr + slot[:, None] * N + n[None, :],
                acc.to(out_ptr.dtype.element_ty),
                mask=valid[:, None],
            )


def compact_moe_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    local_index: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    replicas: ReplicaTable | None = None,
    replica_weights: tuple[torch.Tensor, ...] | None = None,
    padding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unweighted per-(token, slot) expert outputs `[T, top_k, H]` (BF16) of the T
    bf16 rows `x` on their local routes; rows of other ranks' routes are left
    unwritten.

    With `replicas`, `replica_weights` (w13, w2, w13 scales, w2 scales of the
    rank's replica slots, in slot order, same layouts) follow the local
    experts, and the step's routes are split between owners and copies (every
    rank computes the same split). `local_index` is then the layer's step
    index: on entry the rank's own experts' local index, as left by the
    previous call with the same table; the call rewrites the replicated
    experts' entries, so afterwards it marks exactly the routes this rank
    computed.

    Args:
        x: `[T, H]` bf16 rows.
        topk_ids: `[T, top_k]` int32 global expert ids (< 0: padding route).
        local_index: `[E]` int32 local expert of each global expert (-1:
            another rank's).
        w13: the local experts' preshuffled MXFP4 gate/up weights.
        w2: the local experts' preshuffled MXFP4 down weights.
        w13_scale: `w13`'s E8M0 scales.
        w2_scale: `w2`'s E8M0 scales (shapes: `supports_compact_moe`).
        replicas: this layer's replica table on this rank, or None.
        replica_weights: with `replicas`, the replica slots' weights.
        padding: with `replicas`, optional `[T]` bool flags of padding rows
            (e.g. CUDA graph padding): computed as any row, but left out of
            the assignment so that their contents cannot change it.

    """
    T, H = x.shape
    top_k = topk_ids.shape[1]
    num_local, inter = w13.shape[0], w2.shape[2] * 2
    assert supports_compact_moe(x, topk_ids, w13, w2, w13_scale, w2_scale), (
        "compact MoE does not support these rows, routes or weights"
    )
    if replicas is not None:
        assert replica_weights is not None and len(replica_weights) == 4
        nrep = replicas.num_slots - num_local
        for t, own in zip(replica_weights, (w13, w2, w13_scale, w2_scale)):
            assert t.shape == (nrep, *own.shape[1:]) and t.is_contiguous()
        assert local_index.dtype == torch.int32
        if padding is not None:
            assert padding.dtype == torch.bool and padding.shape == (T,)
            padding = padding.contiguous()
        num_local = replicas.num_slots
    cfg = tile_config(T)
    bm = cfg.block_m
    dev = x.device
    rows = T * top_k
    num_blocks = triton.cdiv(rows + num_local * (bm - 1), bm)
    sorted_slots = torch.empty(num_blocks * bm, dtype=torch.int32, device=dev)
    block_expert = torch.empty(num_blocks, dtype=torch.int32, device=dev)
    q = torch.empty((T, H // 2), dtype=torch.uint8, device=dev)
    q_scale = torch.empty((T, H // 32), dtype=torch.uint8, device=dev)
    inter_q = torch.empty((rows, inter // 2), dtype=torch.uint8, device=dev)
    inter_scale = torch.empty((rows, inter // 32), dtype=torch.uint8, device=dev)
    out = torch.empty((T, top_k, H), dtype=torch.bfloat16, device=dev)
    dc = 1024
    common = dict(
        S=triton.next_power_of_2(rows),
        PER_E=triton.next_power_of_2(triton.cdiv(rows, bm)),
        BM=bm,
        NBB=triton.next_power_of_2(num_blocks),
        TOPK=top_k,
        D=H,
        DC=dc,
        num_warps=4,
    )
    if replicas is None:
        _route_quant_kernel[(num_local + T * (H // dc),)](
            topk_ids,
            local_index,
            sorted_slots,
            block_expert,
            x,
            q,
            q_scale,
            T,
            num_blocks,
            LE=num_local,
            **common,
        )
    else:
        num_experts = local_index.numel()
        _route_quant_replica_kernel[(num_local + T * (H // dc),)](
            topk_ids,
            replicas.table,
            local_index,
            local_index if padding is None else padding.view(torch.uint8),
            sorted_slots,
            block_expert,
            x,
            q,
            q_scale,
            T,
            num_blocks,
            E=num_experts,
            LE=num_experts // replicas.ranks,
            NS=replicas.num_slots,
            NSP=replicas.slots_padded,
            NEP=replicas.entries_padded,
            P=replicas.ranks,
            RANK=replicas.rank,
            UNIT=REPLICA_COST_UNIT,
            EXTRA=REPLICA_COST_EXTRA_BLOCK,
            ROUNDS=REPLICA_ROUNDS,
            HAS_PAD=padding is not None,
            **common,
        )
    w13_u8, w2_u8 = w13.view(torch.uint8), w2.view(torch.uint8)
    s13_u8, s2_u8 = w13_scale.view(torch.uint8), w2_scale.view(torch.uint8)
    if replica_weights is None:
        local_slots = 0
        w13r_u8, s13r_u8, w2r_u8, s2r_u8 = w13_u8, s13_u8, w2_u8, s2_u8
    else:
        local_slots = w13.shape[0]
        w13r_u8, w2r_u8, s13r_u8, s2r_u8 = (
            t.view(torch.uint8) for t in replica_weights
        )
    _expert_kernel[(inter // cfg.bn1, num_blocks)](
        q,
        q_scale,
        w13_u8,
        s13_u8,
        sorted_slots,
        block_expert,
        inter_q,
        inter_scale,
        rows,
        w13r_u8,
        s13r_u8,
        K=H,
        N=inter,
        BN=cfg.bn1,
        BK=cfg.bk1,
        BM=bm,
        GATE_UP=True,
        TOPK=top_k,
        LOCAL=local_slots,
        num_warps=cfg.warps1,
        num_stages=cfg.stages1,
        matrix_instr_nonkdim=16,
    )
    _expert_kernel[(H // cfg.bn2, num_blocks)](
        inter_q,
        inter_scale,
        w2_u8,
        s2_u8,
        sorted_slots,
        block_expert,
        out,
        out,
        rows,
        w2r_u8,
        s2r_u8,
        K=inter,
        N=H,
        BN=cfg.bn2,
        BK=cfg.bk2,
        BM=bm,
        GATE_UP=False,
        TOPK=top_k,
        LOCAL=local_slots,
        num_warps=cfg.warps2,
        num_stages=cfg.stages2,
        matrix_instr_nonkdim=16,
    )
    return out
