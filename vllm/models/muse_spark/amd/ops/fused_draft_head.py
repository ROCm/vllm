# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Muse Spark DSpark draft head as one persistent launch (`--monokernel`, 1-2 requests).

`markov_vocab_shard.MarkovVocabShard` runs the draft head on each TP rank's vocabulary
shard as the lm_head GEMM, then per draft step the bias GEMM on the rank's `markov_w2`
rows, the sampling kernel, an all-gather of the per-block keys and the combine kernel.
Here the lm_head and the steps are one launch: program p owns columns
[p BV, p BV + BV) of the shard and

  * computes its columns of the base logits in place (bf16 MFMAs over K, bf16 logits as
    the GEMM's output), and before the first step everything that does not depend on the
    chain (Gumbel noise of every step, positions, seeds, temperatures),
  * per step: the previous token's `markov_w1` row, the bias of its columns (one bf16
    MFMA for both requests), the transform, the staged logits and their Gumbel-max key;
    the rank's winner meets in a relaxed atomic max + ticket, and the ticket's last
    program puts one 64-bit word (score | 14-bit tag | 18-bit complement of the token id)
    into every rank's mailbox with one vector system-scope atomic; every program polls
    its own mailbox with system-scope atomic loads and takes the maximum word: the token
    of the step, on every rank.

The draws are `MarkovVocabShard`'s (Gumbel noise keyed by the global token id, maximum
score then lowest id), and the staged logits feed its cache all-gather and scatter. A NaN
logit samples as -inf here. Tags are epoch-based (the epoch word bumped by the last
ticket of a launch), so CUDA graph replays need no reset. A poll gives up after
POLL_LIMIT iterations and sets health words, which `poll_health` checks.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    _log1p_neg_stable,
    murmur3_uniform32,
    spec_step_seed,
)

logger = init_logger(__name__)

POLL_LIMIT = tl.constexpr(1 << 22)
BV = 128  # vocabulary columns a program: 198 programs (all resident) at 25256 columns
HKCH = 256  # lm_head K chunk: the hidden size is a multiple of it
NUM_WARPS = 8
WORLD = 8


@triton.jit
def _row_prep(pos_ptr, idx_map_ptr, temp_ptr, seeds_ptr, keys, ii, NSPEC: tl.constexpr):
    """A request's chain-independent values: Gumbel noise [8, BV] of every step (the round
    kernel's draw at the step's seed / position) and temperature a step."""
    im = ii < NSPEC
    pos = tl.load(pos_ptr + ii, mask=im, other=1) - 1
    req = tl.load(idx_map_ptr + ii, mask=im, other=-1).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid & im, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req, mask=valid & im, other=0)
    seed = spec_step_seed(seed, ii)
    u = murmur3_uniform32(
        seed[:, None], (pos + _DRAFT_NOISE_SALT)[:, None], keys[None, :]
    )
    noise = -tl.log(-_log1p_neg_stable(u))
    return noise, temp


@triton.jit
def _pick(mat, ii, i):
    return tl.sum(tl.where((ii == i)[:, None], mat, 0.0), axis=0)


@triton.jit
def _pick1(vec, ii, i):
    return tl.sum(tl.where(ii == i, vec, 0), axis=0)


@triton.jit
def _head_base(hid_ptr, hid_stride, sidx_ptr, wh_ptr, wh_stride, v, vmask,
               R: tl.constexpr, NSPEC: tl.constexpr, K: tl.constexpr, KCH: tl.constexpr):  # fmt: skip
    """The draft lm_head on the program's columns: request r's sample rows in MFMA operand
    columns 8 r .., one bf16 MFMA [BV, KCH] x [KCH, 16] per K chunk, the logits rounded to
    bf16 as the GEMM's output. [8, BV] fp32 a request."""
    jj = tl.arange(0, 16)
    rq = jj // 8
    ri = jj % 8
    jm = (ri < NSPEC) & (rq < R)
    rows = tl.load(sidx_ptr + rq * NSPEC + ri, mask=jm, other=0).to(tl.int64)
    acc = tl.zeros([v.shape[0], 16], dtype=tl.float32)
    for kc in tl.range(0, K, KCH):
        kk = kc + tl.arange(0, KCH)
        w = tl.load(
            wh_ptr + v[:, None] * wh_stride + kk[None, :],
            mask=vmask[:, None],
            other=0.0,
        )
        h = tl.load(
            hid_ptr + rows[None, :] * hid_stride + kk[:, None],
            mask=jm[None, :],
            other=0.0,
        )
        acc = tl.dot(w, h.to(tl.bfloat16), acc)
    lg = acc.to(tl.bfloat16).to(tl.float32)
    lg = tl.where(jm[None, :], lg, 0.0)
    b0, b1 = tl.split(
        tl.permute(tl.reshape(tl.trans(lg), (2, 8, v.shape[0])), (1, 2, 0))
    )
    return b0, b1


@triton.jit
def _exchange(keys, rr, rmask, i, epoch, slot_ptr, count_ptr, epoch_ptr, mbox_ptr, peers_ptr,
              health_ptr, rank, vocab_full, G, NSPEC: tl.constexpr, R: tl.constexpr,
              W: tl.constexpr):  # fmt: skip
    """The step's token of each row: the maximum key over every program of every rank."""
    tag14 = ((epoch.to(tl.uint64) * NSPEC + i + 1) & 0x3FFF).to(tl.uint64)
    sl = slot_ptr + (i * 2 + rr) * 16
    old = tl.atomic_max(sl, keys, mask=rmask, sem="relaxed")
    inc = ((tl.min(tl.where(rmask, old, 0), axis=0) != 1) & (epoch >= 0)).to(tl.int32)
    ticket = tl.atomic_add(count_ptr + i * 32, inc, sem="relaxed")
    jj = tl.arange(0, W)
    if ticket == G - 1:
        win = tl.atomic_and(sl, tl.zeros_like(keys), mask=rmask, sem="relaxed")
        tl.atomic_xchg(count_ptr + i * 32, 0, sem="relaxed")
        tid = 0xFFFFFFFF - (win & 0xFFFFFFFF)
        word = ((win >> 32) << 32) | (tag14 << 18) | (0x3FFFF - tid)
        pj = tl.load(peers_ptr + jj).to(tl.pointer_type(tl.uint64))
        dst = pj[None, :] + ((i * R + rr[:, None]) * W + rank)
        wv = word[:, None] + tl.zeros([2, W], dtype=tl.uint64)
        tl.atomic_xchg(dst, wv, mask=rmask[:, None], sem="relaxed", scope="sys")
        if i == NSPEC - 1:
            tl.atomic_add(epoch_ptr, 1, sem="relaxed")
    mb = mbox_ptr + (i * R + rr[:, None]) * W + jj[None, :]
    mm = rmask[:, None] & (jj[None, :] >= 0)
    zw = tl.zeros([2, W], dtype=tl.uint64)
    got = tl.atomic_add(mb, zw, mask=mm, sem="relaxed", scope="sys")
    n = 0
    while (tl.sum(tl.sum(((((got >> 18) & 0x3FFF) == tag14) | ~mm).to(tl.int32), axis=1),
                  axis=0) < 2 * W) & (n < POLL_LIMIT):  # fmt: skip
        got = tl.atomic_add(mb, zw, mask=mm, sem="relaxed", scope="sys")
        n += 1
    if n >= POLL_LIMIT:
        tl.atomic_xchg(health_ptr, 1, sem="relaxed")
        tl.atomic_xchg(health_ptr + 1, i, sem="relaxed")
    best = tl.max(tl.where(mm, got, 0), axis=1)
    t = (0x3FFFF - (best & 0x3FFFF)).to(tl.int64)
    return tl.minimum(tl.maximum(t, 0), vocab_full - 1)


@triton.jit
def _draft_head_kernel(
    hid_ptr, hid_stride, sidx_ptr, wh_ptr, wh_stride, w2_ptr, w2_stride,
    w1_ptr, w1_stride, prev_ptr, idx_map_ptr, idx_stride, pos_ptr, pos_stride, temp_ptr,
    seeds_ptr, stage_ptr, stage_s0, stage_s1, tokens_ptr, tokens_stride, slot_ptr, count_ptr,
    epoch_ptr, mbox_ptr, peers_ptr, health_ptr, rank, mult, inv_cap, cap, vocab, vocab_start,
    vocab_full, R: tl.constexpr, NSPEC: tl.constexpr, W: tl.constexpr,
    RANK: tl.constexpr, BV: tl.constexpr, HK: tl.constexpr, HKCH: tl.constexpr,
    HAS_CAP: tl.constexpr,
):  # fmt: skip
    p = tl.program_id(0)
    G = tl.num_programs(0)
    epoch = tl.load(epoch_ptr, volatile=True)
    v = p * BV + tl.arange(0, BV)
    vmask = v < vocab
    keys = vocab_start + v
    kk = tl.arange(0, RANK)
    ii = tl.arange(0, 8)
    rr = tl.arange(0, 2)
    rmask = rr < R
    cols = tl.arange(0, 16)
    w = tl.load(
        w2_ptr + v[:, None] * w2_stride + kk[None, :], mask=vmask[:, None], other=0.0
    )
    n0, t0 = _row_prep(pos_ptr, idx_map_ptr, temp_ptr, seeds_ptr, keys, ii, NSPEC)
    n1, t1 = n0, t0
    if R > 1:
        n1, t1 = _row_prep(pos_ptr + pos_stride, idx_map_ptr + idx_stride, temp_ptr,
                               seeds_ptr, keys, ii, NSPEC)  # fmt: skip
    b0, b1 = _head_base(hid_ptr, hid_stride, sidx_ptr, wh_ptr, wh_stride, v, vmask, R, NSPEC,
                        HK, HKCH)  # fmt: skip
    tok0 = tl.load(prev_ptr + 0).to(tl.int64)
    tok1 = tok0
    if R > 1:
        tok1 = tl.load(prev_ptr + 1).to(tl.int64)
    for i in tl.static_range(NSPEC):
        e0 = tl.load(w1_ptr + tok0 * w1_stride + kk).to(tl.float32)
        e1 = tl.load(w1_ptr + tok1 * w1_stride + kk).to(tl.float32)
        # bf16 markov_w1 row x bf16 markov_w2 rows (the stock unquantized bias GEMM)
        qm = tl.where(
            cols[None, :] == 0,
            e0[:, None],
            tl.where(cols[None, :] == 1, e1[:, None], 0.0),
        )
        acc16 = tl.dot(w, qm.to(tl.bfloat16))
        # rows as columns of one [BV, 2] pass
        acc = tl.join(tl.sum(tl.where(cols[None, :] == 0, acc16, 0.0), axis=1),
                      tl.sum(tl.where(cols[None, :] == 1, acc16, 0.0), axis=1))  # fmt: skip
        bias = acc.to(tl.bfloat16)
        base = tl.join(_pick(b0, ii, i), _pick(b1, ii, i))
        x = (base + bias.to(tl.float32)).to(tl.bfloat16)
        x = x.to(tl.float32) * mult
        if HAS_CAP:
            x = cap * libdevice.tanh(x * inv_cap)
        temp = tl.where(rr == 0, _pick1(t0, ii, i), _pick1(t1, ii, i))
        vm2 = vmask[:, None] & rmask[None, :]
        # cached and sampled in the cache dtype
        x = x.to(stage_ptr.dtype.element_ty)
        tl.store(stage_ptr + rr[None, :] * stage_s0 + i * stage_s1 + v[:, None], x,
                 mask=vm2)  # fmt: skip
        lf = x.to(tl.float32)
        lf = tl.where(vm2 & (lf == lf), lf, float("-inf"))
        nz = tl.join(_pick(n0, ii, i), _pick(n1, ii, i))
        tp = temp[None, :] != 0.0
        lf = tl.where(
            tp,
            tl.where(vm2, lf / tl.where(tp, temp[None, :], 1.0) + nz, float("-inf")),
            lf,
        )
        value, idx = tl.max(lf, axis=0, return_indices=True)
        token = (vocab_start + p * BV + idx).to(tl.int64)
        bits = (value + 0.0).to(tl.uint32, bitcast=True)
        ordered = tl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)
        low = (0xFFFFFFFF - token.to(tl.uint32)).to(tl.uint64)
        kv = (ordered.to(tl.uint64) << 32) | low
        t = _exchange(kv, rr, rmask, i, epoch, slot_ptr, count_ptr, epoch_ptr, mbox_ptr,
                      peers_ptr, health_ptr, rank, vocab_full, G, NSPEC, R, W)  # fmt: skip
        tok0 = tl.sum(tl.where(rr == 0, t, 0), axis=0)
        tok1 = tl.sum(tl.where(rr == 1, t, 0), axis=0)
        if p == 0:
            tl.store(tokens_ptr + rr * tokens_stride + i, t, mask=rmask)


def _rows(w: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """A bf16 weight as row-major [n, k] (a view), stored [n, k] or as a [k, n] view."""
    t = w if tuple(w.shape) == (n, k) else w.t()
    if tuple(t.shape) != (n, k) or t.stride(1) != 1:
        raise ValueError(f"bf16 rows {tuple(w.shape)} {tuple(w.stride())}")
    return t


class _Ptr:
    """A raw device address passed to Triton as a uint64 pointer."""

    def __init__(self, addr: int) -> None:
        self.addr = addr
        self.dtype = torch.uint64

    def data_ptr(self) -> int:
        return self.addr


class MuseSparkFusedDraftHead:
    """The vocabulary-sharded DSpark draft head of `--monokernel` steps of 1-2
    requests as one launch, for `MarkovVocabShard` (`shard`).

    Set up at the first call (the warm-up step, on every rank). `sample` returns False
    for the steps it does not take, which the shard's own chain then runs."""

    def __init__(self, model: torch.nn.Module, shard: Any) -> None:
        self.model = model
        self.shard = shard
        self._state: str | None = None  # None: not set up; "on" or the reason it is off
        self.health: torch.Tensor | None = None
        self._steps = 0
        self._health_event: torch.cuda.Event | None = None

    def _setup(self, spec) -> str:
        model, shard = self.model, self.shard
        if not model.vllm_config.kernel_config.monokernel:
            return "not --monokernel"
        if shard.greedy:
            return "greedy drafts"
        if spec.num_speculative_steps > 8:
            return "more than 8 draft steps"
        if model.logits_processor.head_dtype not in (None, torch.bfloat16):
            return "an fp32 lm_head"
        from vllm.distributed import get_tp_group

        from ..mono.peer import PeerBuffer

        group = get_tp_group()
        if group.world_size != WORLD:
            return f"tensor parallel {group.world_size}"
        lm_head = model.lm_head.weight
        w2 = shard.w2_shard.weight
        hidden, rank_dim = lm_head.shape[-1], shard.w1.shape[1]
        if hidden % HKCH != 0 or rank_dim < 16:
            return f"hidden size {hidden} / Markov rank {rank_dim}"
        dtypes = {lm_head.dtype, w2.dtype, shard.w1.dtype}
        try:
            if dtypes != {torch.bfloat16}:
                raise ValueError(f"dtypes {sorted(map(str, dtypes))}, not bf16")
            self.wh = _rows(lm_head, lm_head.shape[0], hidden)
            self.w2 = _rows(w2, shard.shard_size, rank_dim)
        except ValueError as e:
            return f"head / Markov rows: {e}"
        if self.wh.shape[0] < shard.shard_size:
            return "lm_head rows"
        device = self.wh.device
        # The programs wait on each other: all of them must be resident, one a CU.
        programs = triton.cdiv(shard.shard_size, BV)
        cus = torch.cuda.get_device_properties(device).multi_processor_count
        if programs > cus:
            return f"{programs} programs on {cus} CUs"
        self.rank = group.rank_in_group
        n = spec.num_speculative_steps
        self.peers = PeerBuffer(n * 2 * WORLD * 8 + 256, group.cpu_group)
        self.peer_ptrs = torch.tensor(
            self.peers.pointers, dtype=torch.int64, device=device
        )
        self.mbox = _Ptr(self.peers.pointers[self.rank])
        self.slot = torch.zeros(n * 2 * 16, dtype=torch.uint64, device=device)
        self.count = torch.zeros(n * 32, dtype=torch.int32, device=device)
        self.epoch = torch.zeros(4, dtype=torch.int32, device=device)
        self.health = torch.zeros(8, dtype=torch.int32, device=device)
        self._health_host = torch.zeros(8, dtype=torch.int32).pin_memory()
        soft_cap = model.output_soft_cap_temp
        self.cap = float(soft_cap) if soft_cap is not None else 1.0
        self.has_cap = soft_cap is not None
        self.mult = float(model.output_multiplier)
        # torch divides by a scalar as a multiply by its fp32 reciprocal.
        self.inv_cap = float(
            torch.tensor(1.0, dtype=torch.float32)
            / torch.tensor(self.cap, dtype=torch.float32)
        )
        from ..mono.decoder import register_health_check

        register_health_check(id(self), self.poll_health)
        return "on"

    def sample(
        self, spec, num_reqs: int, head_hidden: torch.Tensor, stage: torch.Tensor | None
    ) -> bool:
        """Draft tokens into `spec.draft_tokens` and the staged logits into `stage`
        (`[num_reqs, steps, shard]`); False for a step left to the shard's chain."""
        if self._state is None:
            # collective (IPC mailboxes): every rank sets up on the same first call
            self._state = self._setup(spec)
            logger.info(
                "Muse Spark DSpark fused draft head (lm_head + %d Markov steps, one "
                "launch at 1-2 requests): %s",
                spec.num_speculative_steps,
                self._state,
            )
            return False
        if (
            self._state != "on"
            or num_reqs not in (1, 2)
            or stage is None
            or spec.use_fp64_gumbel
            or spec.use_confidence_head
        ):
            return False
        shard = self.shard
        n = spec.num_speculative_steps
        num_sample = num_reqs * n
        idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n)
        pos = spec.sample_pos[:num_sample].view(num_reqs, n)
        tokens = spec.draft_tokens[:num_reqs]
        prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
        sidx = spec.sample_indices[:num_sample]
        w1 = shard.w1
        grid = (triton.cdiv(shard.shard_size, BV),)
        _draft_head_kernel[grid](
            head_hidden, head_hidden.stride(0), sidx, self.wh, self.wh.stride(0),
            self.w2, self.w2.stride(0), w1, w1.stride(0), prev, idx_map,
            idx_map.stride(0), pos, pos.stride(0), spec.temperature, spec.seeds, stage,
            stage.stride(0), stage.stride(1), tokens, tokens.stride(0), self.slot,
            self.count, self.epoch, self.mbox, self.peer_ptrs, self.health, self.rank,
            self.mult, self.inv_cap, self.cap, shard.shard_size, shard.vocab_start,
            shard.vocab_size, R=num_reqs, NSPEC=n, W=WORLD, RANK=w1.shape[1], BV=BV,
            HK=head_hidden.shape[1], HKCH=HKCH, HAS_CAP=self.has_cap, num_warps=NUM_WARPS,
        )  # fmt: skip
        return True

    def poll_health(self) -> None:
        """Raise if a launch timed out waiting for a peer rank (an asynchronous copy every
        64 calls, checked one interval later)."""
        if self.health is None or torch.cuda.is_current_stream_capturing():
            return
        self._steps += 1
        if self._health_event is not None and self._health_event.query():
            self._health_event = None
            if int(self._health_host[0]) != 0:
                raise RuntimeError(
                    "Muse Spark DSpark fused draft head: a launch timed out waiting for "
                    f"a peer rank (step {int(self._health_host[1])})"
                )
        if self._health_event is None and self._steps % 64 == 0:
            self._health_host.copy_(self.health, non_blocking=True)
            self._health_event = torch.cuda.Event()
            self._health_event.record()
