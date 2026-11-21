# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark TP all-reduces fused with their neighbouring kernels (ROCm gfx950).

The kernels (`_rocm_C.muse_fused_allreduce_*`) push each rank's partial rows
into the peers' inboxes, whose data is its own flag. One block carries one token
row, or, for small row counts at TP8, a sender, an owner and a row block share
it. This module owns the inboxes and the per-row state and picks the
communication mode of each call. The inboxes are torch symmetric memory, or,
where that is unavailable, vLLM's IPC-shared buffers: device memory of each rank
mapped into every peer, which the kernels reach with system-scope,
cache-bypassing accesses.

Every rank must issue the same sequence of calls with the same row counts, one
call at a time: the calls share the per-row state and the inboxes, so the ops
are not for concurrent callers (micro-batches on separate streams).
"""

import functools
from collections.abc import Callable
from typing import TypeVar

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

WORLD_SIZES = (4, 8)
MAX_ROWS = 512
FINALIZE_HIDDEN = 4096
BOUNDARY_HIDDEN = 8192
TOP_K = 8
# Communication modes of the kernels.
_TWO_HOP, _ONE_HOP, _NO_COMM, _TWO_HOP_CHUNKED, _MULTI_BLOCK = 0, 1, 2, 3, 4
# Up to this many rows every rank pushes its partial to every peer (one hop, 2x
# the link bytes); above, partials go to the row's owner, which pushes the sum.
ONE_HOP_MAX_ROWS = 8
# Up to this many rows the finalize loads all route rows without waiting on the
# expert ids.
FINALIZE_READ_ALL_MAX_ROWS = 64
# Per world size, up to this many rows each row is served by a sender, an owner
# and a row block; 0: never.
MULTI_BLOCK_MAX_ROWS = {4: 0, 8: 64}
# Per world size, the smallest row count above the multi-block range that the
# one-block-per-row kernels serve; below it, the layers keep their unfused
# kernels and all-reduce.
FINALIZE_MIN_ROWS = {4: 1, 8: 64}
BOUNDARY_MIN_ROWS = {4: 16, 8: 128}
# The finalize's two-hop mode per world size. Chunked, every rank owns a slice
# of every row, which spreads the second hop over all blocks; unchunked, a row
# owner can push the FP8 rows of its sum instead (half the bytes).
FINALIZE_TWO_HOP_MODE = {4: _TWO_HOP, 8: _TWO_HOP_CHUNKED}

_FP8 = torch.float8_e4m3fn
# Empty inbox records: bf16 -0.0 pairs; FP8 records: FP8 -0 bytes.
_EMPTY_BF16 = 0x80008000
_EMPTY_FP8 = 0x80808080

_T = TypeVar("_T")


def is_supported() -> bool:
    """Whether this build and device have the fused all-reduce kernels."""
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950() and hasattr(torch.ops._rocm_C, "muse_fused_allreduce_fill")


def _mode(rows: int, two_hop: int = _TWO_HOP) -> int:
    return _ONE_HOP if rows <= ONE_HOP_MAX_ROWS else two_hop


def _row_bytes(world: int, hidden: int, max_rows: int) -> int:
    """Two hops of bf16 rows in 3 slots: `[3][world]` stage-1 and `[3]` stage-2
    records per row."""
    return 3 * (world + 1) * max_rows * hidden * 2


class _Op:
    """One fused op's inbox region and per-row state."""

    def __init__(
        self,
        world: int,
        hidden: int,
        max_rows: int,
        multi_block: bool,
        fp8_hop: bool,
        device: torch.device,
    ):
        self.max_rows = max_rows
        self.bf16_bytes = _row_bytes(world, hidden, max_rows)
        # FP8 second hop: FP8 rows plus a 16-byte scale record, in 3 slots.
        self.fp8_bytes = 3 * max_rows * (hidden + 16) if fp8_hop else 0
        self.nbytes = self.bf16_bytes + self.fp8_bytes
        if multi_block:
            # Call counts of the sender, owner and row block of every row.
            self.state = torch.zeros(3 * max_rows, dtype=torch.int32, device=device)
        else:
            # Row-call count and owner of the previous row-call (-1: none).
            self.state = torch.tensor([0, -1], dtype=torch.int32, device=device).repeat(
                max_rows
            )
        self.inboxes: list[int] = []

    def place(self, peers: list[int], offset: int, rank: int, device: int) -> None:
        """Point at this op's region of every rank's inbox and empty this
        rank's."""
        self.inboxes = [p + offset for p in peers]
        own = self.inboxes[rank]
        ops.muse_fused_allreduce_fill(own, self.bf16_bytes, _EMPTY_BF16, device)
        if self.fp8_bytes:
            ops.muse_fused_allreduce_fill(
                own + self.bf16_bytes, self.fp8_bytes, _EMPTY_FP8, device
            )


class _InitFailed(RuntimeError):
    """A construction step failed on at least one rank (on every rank)."""


class MuseFusedAllReduce:
    """The fused all-reduces of one process; collective over the TP group.

    Construction either succeeds on every rank or raises on every rank: each
    step that can fail locally (allocations, the inbox mapping) is agreed
    across the group before the next collective."""

    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device_group: dist.ProcessGroup,
        device: torch.device,
        inbox: str | None = None,
    ):
        """`inbox` forces the inbox allocation ("symm_mem" or "ipc"); by default
        torch symmetric memory, else vLLM's IPC-shared buffers."""
        if inbox not in (None, "symm_mem", "ipc"):
            raise ValueError(f"unknown inbox allocation {inbox!r}")
        world = dist.get_world_size(group=cpu_group)
        if world not in WORLD_SIZES:
            raise ValueError(f"fused all-reduce needs TP in {WORLD_SIZES}")
        self.world = world
        self.rank = dist.get_rank(group=cpu_group)
        if device.index is None:
            device = torch.device(device.type, torch.accelerator.current_device_index())
        self.device = device
        self._cpu_group = cpu_group
        self.closed = False
        self._symm_buffer: torch.Tensor | None = None
        self._symm_handle = None
        self._ipc_ptr = 0
        self._ipc_peers: list[int] = []
        self._ops: tuple[_Op, ...] = ()
        try:
            self._build(cpu_group, device_group, device, inbox)
        except BaseException:
            self.close()
            raise

    def _all_ranks_true(self, ok: bool) -> bool:
        flag = torch.tensor(int(ok), dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._cpu_group)
        return bool(flag.item())

    def _agreed(self, what: str, fn: Callable[[], _T]) -> tuple[bool, _T | None]:
        """Run the local step `fn` and agree its success across the group:
        `(True, result)` on every rank, or `(False, None)` on every rank."""
        result = None
        try:
            result = fn()
            ok = True
        except Exception as e:
            logger.warning("Muse Spark fused all-reduces: %s failed: %s", what, e)
            ok = False
        if not self._all_ranks_true(ok):
            return False, None
        return True, result

    def _step(self, what: str, fn: Callable[[], _T]) -> _T:
        """`fn` on every rank, or `_InitFailed` on every rank."""
        ok, result = self._agreed(what, fn)
        if not ok:
            raise _InitFailed(f"{what} failed on a rank")
        return result  # type: ignore[return-value]

    def _build(
        self,
        cpu_group: dist.ProcessGroup,
        device_group: dist.ProcessGroup,
        device: torch.device,
        inbox: str | None,
    ) -> None:
        world = self.world
        # One block per row, all resident at once (one per CU, on the rank with
        # the fewest CUs), so that every row's exchange is in flight together.
        cus = torch.tensor(
            torch.cuda.get_device_properties(device).multi_processor_count,
            dtype=torch.int64,
        )
        dist.all_reduce(cus, op=dist.ReduceOp.MIN, group=cpu_group)
        self.boundary_max_rows = min(MAX_ROWS, int(cus.item()))
        # Three blocks per row, all resident at once.
        self.mb_max_rows = min(MULTI_BLOCK_MAX_ROWS[world], self.boundary_max_rows // 3)

        def make_ops() -> tuple[_Op, ...]:
            return (
                _Op(world, FINALIZE_HIDDEN, MAX_ROWS, False, True, device),
                _Op(
                    world, BOUNDARY_HIDDEN, self.boundary_max_rows, False, False, device
                ),
                _Op(world, FINALIZE_HIDDEN, self.mb_max_rows, True, False, device),
                _Op(world, BOUNDARY_HIDDEN, self.mb_max_rows, True, False, device),
            )

        self._ops = self._step("allocating the row state", make_ops)
        (
            self.finalize_op,
            self.boundary_op,
            self.mb_finalize_op,
            self.mb_boundary_op,
        ) = self._ops
        nbytes = sum(op.nbytes for op in self._ops)
        peers = None
        if inbox in (None, "symm_mem"):
            peers = self._symm_mem_inbox(nbytes, device_group, device)
            if peers is None and inbox == "symm_mem":
                raise _InitFailed("torch symmetric memory is unavailable")
            self.inbox_kind = "symm_mem"
        if peers is None:
            peers = self._ipc_inbox(nbytes, cpu_group)
            self.inbox_kind = "ipc"

        def place() -> None:
            offset = 0
            for op in self._ops:
                if op.nbytes:
                    op.place(peers, offset, self.rank, device.index)
                offset += op.nbytes
            # Stands in for a missing norm weight.
            self._ones = torch.ones(
                BOUNDARY_HIDDEN, dtype=torch.bfloat16, device=device
            )
            torch.accelerator.synchronize(device)

        self._step("emptying the inboxes", place)
        dist.barrier(group=cpu_group)
        self.nbytes = nbytes

    def _symm_mem_inbox(
        self,
        nbytes: int,
        device_group: dist.ProcessGroup,
        device: torch.device,
    ) -> list[int] | None:
        """The inboxes in torch symmetric memory, or None (on every rank) if a
        rank cannot allocate or map it."""

        def allocate() -> torch.Tensor:
            import torch.distributed._symmetric_memory as symm_mem

            return symm_mem.empty(nbytes, dtype=torch.uint8, device=device)

        ok, buffer = self._agreed("allocating symmetric memory", allocate)
        if not ok:
            return None
        self._symm_buffer = buffer

        def rendezvous() -> list[int]:
            import torch.distributed._symmetric_memory as symm_mem

            handle = symm_mem.rendezvous(buffer, device_group.group_name)
            self._symm_handle = handle
            return [int(p) for p in handle.buffer_ptrs]

        ok, peers = self._agreed("mapping symmetric memory", rendezvous)
        if not ok:
            self._symm_handle = self._symm_buffer = None
            return None
        return peers

    def _ipc_inbox(self, nbytes: int, cpu_group: dist.ProcessGroup) -> list[int]:
        """The inboxes in vLLM's IPC-shared buffers."""

        def allocate() -> tuple[int, torch.Tensor]:
            ptr, handle = ops.allocate_shared_buffer_and_handle(nbytes)
            self._ipc_ptr = ptr  # freed by `close` if another rank failed
            return ptr, handle

        ptr, handle = self._step("allocating the IPC inbox", allocate)
        handles: list[torch.Tensor | None] = [None] * self.world
        dist.all_gather_object(handles, handle, group=cpu_group)

        def open_handles() -> list[int]:
            peers = []
            for i, h in enumerate(handles):
                if i == self.rank:
                    peers.append(ptr)
                else:
                    peers.append(ops.open_mem_handle(h))
                    self._ipc_peers.append(peers[-1])
            return peers

        return self._step("opening the peers' IPC inboxes", open_handles)

    def close(self) -> None:
        """Release the inboxes (local; idempotent). Every call issued on any
        stream, including from CUDA graphs, must have been issued before, and
        none after: graphs that captured these ops must not be replayed."""
        if self.closed:
            return
        self.closed = True
        for op in self._ops:
            op.inboxes = []
        if not (self._symm_buffer is not None or self._ipc_ptr or self._ipc_peers):
            return
        # Every peer write into this rank's inbox belongs to a call this rank
        # waits on, so once this rank's calls are done no peer writes to it.
        torch.accelerator.synchronize(self.device)
        self._symm_handle = self._symm_buffer = None
        for ptr in self._ipc_peers:
            ops.muse_fused_allreduce_close_ipc_handle(ptr, self.device.index)
        self._ipc_peers = []
        if self._ipc_ptr:
            ops.free_shared_buffer(self._ipc_ptr)
            self._ipc_ptr = 0

    def applies(self, rows: int) -> bool:
        return 0 < rows <= self.mb_max_rows or (
            FINALIZE_MIN_ROWS[self.world] <= rows <= MAX_ROWS
        )

    def boundary_applies(self, rows: int) -> bool:
        return 0 < rows <= self.mb_max_rows or (
            BOUNDARY_MIN_ROWS[self.world] <= rows <= self.boundary_max_rows
        )

    def finalize(
        self,
        routes: torch.Tensor,
        weight: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor,
        eps: float,
        gain: float,
        *,
        quant: bool = False,
        mode: int | None = None,
    ) -> tuple[torch.Tensor, None, None] | tuple[None, torch.Tensor, torch.Tensor]:
        """The all-reduced weight-first post-expert finalize of `routes`
        `[M, 8, 4096]` (rows of routes whose `expert_map` entry is negative are
        not read): `(out, None, None)` in bf16, or with `quant` its per-token FP8
        rows and scales `(None, q, q_scale)`. `mode` forces a communication mode
        of the one-block-per-row kernel (tests)."""
        m = topk_ids.shape[0]
        assert not self.closed
        assert 0 < m <= MAX_ROWS and routes.shape == (m, TOP_K, FINALIZE_HIDDEN)
        assert routes.dtype == weight.dtype == torch.bfloat16
        assert topk_weights.dtype == torch.float32 and topk_ids.dtype == torch.int32
        assert expert_map.dtype == torch.int32
        for t in (routes, weight, topk_weights, topk_ids, expert_map):
            assert t.is_contiguous()
        out = q = qs = None
        if quant:
            q = routes.new_empty((m, FINALIZE_HIDDEN), dtype=_FP8)
            qs = routes.new_empty((m,), dtype=torch.float32)
        else:
            out = routes.new_empty((m, FINALIZE_HIDDEN))
        if mode is None:
            mode = (
                _MULTI_BLOCK
                if m <= self.mb_max_rows
                else _mode(m, FINALIZE_TWO_HOP_MODE[self.world])
            )
        op = self.mb_finalize_op if mode == _MULTI_BLOCK else self.finalize_op
        ops.muse_fused_allreduce_finalize(
            routes, weight, topk_weights, topk_ids, expert_map, out, q, qs,
            eps, gain, op.inboxes, op.state, self.rank, op.max_rows, mode,
            m <= FINALIZE_READ_ALL_MAX_ROWS,
        )  # fmt: skip
        return out, q, qs  # type: ignore[return-value]

    def boundary(
        self,
        partial: torch.Tensor,
        stream: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        post_weight: torch.Tensor | None,
        pre_weight: torch.Tensor | None,
        post_eps: float,
        pre_eps: float,
        *,
        post_gain_center: float,
        pre_gain_center: float,
        post_input_scale: bool,
        pre_input_scale: bool,
        out_dtype: torch.dtype | None,
        mode: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        """All-reduce `partial` `[M, 8192]` into one residual boundary:
        `new_stream = alpha * stream + beta * post_norm(sum)`, then the pre-norm
        of `new_stream` quantized per token to FP8, as
        `fused_norm_residual_norm_quant_fwd`: `(out | None, new_stream, q,
        q_scale)`, with `out` in `out_dtype` (bf16) if given. `mode` forces a
        communication mode of the one-block-per-row kernel (tests)."""
        m = partial.shape[0]
        assert not self.closed
        assert 0 < m <= self.boundary_max_rows and partial.shape == (m, BOUNDARY_HIDDEN)
        assert stream.shape == partial.shape and stream.dtype == torch.float32
        assert partial.dtype == torch.bfloat16 and out_dtype in (None, torch.bfloat16)
        assert alpha.dtype == beta.dtype == torch.float32
        assert alpha.shape == beta.shape == (BOUNDARY_HIDDEN,)
        for t in (partial, stream, alpha, beta, post_weight, pre_weight):
            assert t is None or t.is_contiguous()
        for w in (post_weight, pre_weight):
            assert w is None or (w.dtype == torch.bfloat16 and w.shape == alpha.shape)
        assert mode != _TWO_HOP_CHUNKED
        new_stream = torch.empty_like(stream)
        out = None if out_dtype is None else torch.empty_like(partial)
        q = partial.new_empty(partial.shape, dtype=_FP8)
        qs = partial.new_empty((m,), dtype=torch.float32)
        if mode is None:
            mode = _MULTI_BLOCK if m <= self.mb_max_rows else _mode(m)
        op = self.mb_boundary_op if mode == _MULTI_BLOCK else self.boundary_op
        ops.muse_fused_allreduce_boundary(
            partial, stream, alpha, beta,
            post_weight if post_weight is not None else self._ones,
            pre_weight if pre_weight is not None else self._ones,
            new_stream, out, q, qs, post_eps, pre_eps,
            post_gain_center if post_weight is not None else 0.0,
            pre_gain_center if pre_weight is not None else 0.0,
            post_input_scale and post_weight is not None,
            pre_input_scale and pre_weight is not None,
            op.inboxes, op.state, self.rank, op.max_rows, mode,
        )  # fmt: skip
        return out, new_stream, q, qs


_instance: MuseFusedAllReduce | None = None
# The TP group coordinator `_instance` was built over.
_instance_group: object | None = None


def get() -> MuseFusedAllReduce | None:
    """The process's fused all-reduces, if `init` built them."""
    return _instance


def destroy() -> None:
    """Release the process's fused all-reduces (local; runs when their TP group
    is destroyed)."""
    global _instance, _instance_group
    instance, _instance, _instance_group = _instance, None, None
    if instance is not None:
        try:
            instance.close()
        except Exception as e:
            logger.warning("Releasing the Muse Spark fused all-reduces: %s", e)


def _on_group_destroy(group: object) -> None:
    if _instance_group is group:
        destroy()


def init(
    device: torch.device, micro_batched: bool = False
) -> MuseFusedAllReduce | None:
    """Build the fused all-reduces of the TP group (collective; outside graph
    capture), or return those already built over it. None for other TP sizes,
    under micro-batching (the calls of concurrent micro-batches would share the
    row state and inboxes), or where the kernels are unavailable on any rank."""
    global _instance, _instance_group
    from vllm.distributed import get_tp_group

    tp = get_tp_group()
    if _instance is not None and _instance_group is tp:
        return _instance
    destroy()
    if tp.world_size not in WORLD_SIZES:
        return None
    if micro_batched:
        logger.info_once(
            "Muse Spark fused all-reduces off: micro-batches would share them."
        )
        return None
    supported = torch.tensor(int(is_supported()), dtype=torch.int32)
    dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=tp.cpu_group)
    if not supported.item():
        return None
    try:
        instance = MuseFusedAllReduce(tp.cpu_group, tp.device_group, device)
    except _InitFailed as e:
        logger.warning("Muse Spark fused all-reduces unavailable: %s", e)
        return None
    _instance, _instance_group = instance, tp
    tp.add_destroy_callback(functools.partial(_on_group_destroy, tp))
    logger.info(
        "Muse Spark fused all-reduces ready (TP%d, up to %d rows, %.0f MiB of %s "
        "inbox).",
        tp.world_size,
        MAX_ROWS,
        instance.nbytes / 2**20,
        instance.inbox_kind,
    )
    return instance
