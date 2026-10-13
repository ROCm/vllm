# SPDX-License-Identifier: Apache-2.0
"""Dispatch MoE expert weight_loader calls across a thread pool.

A py-spy profile of a DeepSeek-V4-Pro load shows 95.8% of the wall clock in
two lines::

    51.6%  _load_w13  (routed_experts.py:508)   expert_data.copy_(loaded_weight)
    44.2%  _load_w2   (routed_experts.py:541)   expert_data.copy_(loaded_weight)

with ``vmstat`` reporting ``wa=0`` -- the load is CPU-bound in the copies, not
waiting on the checkpoint. Those calls are issued one at a time from the
dispatch loop in ``DeepseekV4ForCausalLM.load_weights``. ``Tensor.copy_``
releases the GIL, so running them N-wide is close to an N-times speedup until
the host runs out of memory bandwidth. ATOM does the same thing with
``ATOM_LOADER_NUM_THREADS`` (default 16).

Why the dispatch loop and not ``weight_loader`` itself
------------------------------------------------------
``RoutedExperts`` hands its bound method to each parameter at construction
(``routed_experts.py:167``, ``"weight_loader": self.weight_loader``) and the
loop calls ``param.weight_loader``. Patching the class attribute later does
nothing at all -- the bound methods were captured long before. Wrapping the
caller sidesteps that entirely.

Thread safety
-------------
Each checkpoint tensor is one (expert, shard), so concurrent calls write
disjoint regions: distinct ``expert_id`` are distinct rows of the fused
parameter, and w1/w3 are distinct halves of one row. ``loaded_weight`` is a
read-only mmap view. The loader's own mutable state is a couple of ``set``
adds, which the GIL makes atomic.

The one thing that is NOT safe is expert parallelism: there
``weight_loader`` returns False for a non-local expert and the caller must
try the next mapping, so the return value steers control flow and cannot be
deferred to drain time. ``make_pool`` refuses to activate under EP.

Sized from ``OMP_NUM_THREADS`` (see ``_requested_threads``); inactive
when that is 1, and ``make_pool`` also declines under expert
parallelism.
"""

import atexit
import concurrent.futures
import os

from vllm.logger import init_logger

logger = init_logger(__name__)


# Cap on the count vLLM picks for us. The copies are memory-bandwidth bound,
# so threads stop paying well before they stop being allocated: a 255-CPU node
# at TP4 yields 63 per worker, four times the largest figure measured, with
# all four workers competing for the same bandwidth. 16 took a
# DeepSeek-V4-Pro load from ~1290 s to ~258 s; above that is untested, hence a
# conservative bound rather than a measured optimum. An OMP_NUM_THREADS the
# user set themselves is honoured as-is -- the cap exists to temper a generic
# default, not to overrule an explicit choice.
_MAX_AUTO_THREADS = 16


def _requested_threads() -> int:
    """Threads for the expert-weight copies, taken from OMP_NUM_THREADS.

    ``set_multiprocessing_worker_envs`` returns early when OMP_NUM_THREADS is
    already in the environment, so the variable holds the user's value when
    they set one and otherwise vLLM's own
    ``available_cpu_count() // local_world_size`` -- affinity- and
    cgroup-aware, and divided between the workers sharing the node.
    ``VLLM_OMP_NUM_THREADS_SET_BY_VLLM`` says which of the two it is, so an
    explicit setting is used verbatim while vLLM's generic default is capped
    for this bandwidth-bound workload.
    """
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw is None:
        return 1
    # Not a number means something upstream wrote nonsense into the variable
    # every OpenMP runtime in the process is also reading. Swallowing it would
    # silently drop the pool back to one thread and make a 5x slower load look
    # normal, so fail where the bad value is.
    if not raw.strip().lstrip("+-").isdigit():
        raise ValueError(
            f"OMP_NUM_THREADS={raw!r} is not an integer; the MoE loader pool "
            f"sizes itself from it"
        )
    n = int(raw)
    if n <= 1:
        return 1
    if os.environ.get("VLLM_OMP_NUM_THREADS_SET_BY_VLLM") == "1":
        return min(n, _MAX_AUTO_THREADS)
    return n


def _experts_are_sharded(model) -> bool | None:
    """Whether any MoE module holds only a subset of the global experts.

    Read off the model, NOT off ``get_current_vllm_config()``: that context is
    entered for ``initialize_model`` and has already exited by the time
    ``load_weights`` runs (``base_loader.py``), so asking there raises and the
    guard would refuse every time -- an inert patch that looks installed.

    Returns None when no MoE module was recognised, which the caller treats
    the same as "sharded": erring towards not threading costs load time, and
    erring the other way would silently drop expert weights.
    """
    seen = False
    for module in model.modules():
        local = getattr(module, "local_num_experts", None)
        glob = getattr(module, "global_num_experts", None)
        if local is None or glob is None:
            continue
        seen = True
        if local < glob or getattr(module, "expert_map", None) is not None:
            return True
    return False if seen else None


class _NullPool:
    active = False

    def submit(self, fn, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("submit() on an inactive pool")

    def drain(self) -> None:
        pass


class _Pool:
    """Bounded-in-flight thread pool for weight_loader calls.

    In-flight futures are capped because each one pins its ``loaded_weight``,
    and those are mmap views of the checkpoint: an unbounded queue would hold
    the whole 805 GiB of pages un-reclaimable.
    """

    active = True

    def __init__(self, num_threads: int):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="moe-load")
        self._pending: list[concurrent.futures.Future] = []
        self._max_inflight = max(4 * num_threads, 8)
        self._submitted = 0
        self._drained = False
        atexit.register(self._shutdown)
        logger.info("MoE loader threading enabled: %d threads, %d in flight.",
                    num_threads, self._max_inflight)

    def submit(self, fn, *args, **kwargs) -> None:
        self._pending.append(self._executor.submit(fn, *args, **kwargs))
        self._submitted += 1
        if len(self._pending) >= self._max_inflight:
            # Reap the oldest half rather than all of it, so the workers keep
            # a queue to pull from instead of going idle at every barrier.
            self._reap(len(self._pending) // 2)

    def _reap(self, count: int) -> None:
        for fut in self._pending[:count]:
            self._check(fut)
        del self._pending[:count]

    @staticmethod
    def _check(fut: concurrent.futures.Future) -> None:
        # .result() re-raises whatever the worker raised, on this thread.
        result = fut.result()
        if result is False:
            # Only reachable if a name-matched mapping failed, which is the
            # EP case make_pool refuses. Loud, because the serial loop would
            # have retried the next mapping and we did not.
            raise RuntimeError(
                "MoE loader threading: an expert weight_loader reported "
                "failure. The threaded path assumes a name-matched mapping "
                "always succeeds, which is why make_pool declines under "
                "expert parallelism -- reaching here means that check missed "
                "a sharded case. Set OMP_NUM_THREADS=1 to fall back to the "
                "serial loop.")

    def drain(self) -> None:
        if self._drained:
            return
        self._drained = True
        try:
            self._reap(len(self._pending))
        finally:
            self._executor.shutdown(wait=True)
        logger.info("MoE loader threading: %d expert weight loads dispatched.",
                    self._submitted)

    def _shutdown(self) -> None:
        # Only reached when load_weights raised before drain(); without it the
        # non-daemon worker threads keep the process from exiting.
        if not self._drained:
            self._drained = True
            self._executor.shutdown(wait=False, cancel_futures=True)


NULL_POOL = _NullPool()


def make_pool(model) -> "_Pool | _NullPool":
    """A pool for one load_weights call, or the inactive one."""
    num_threads = _requested_threads()
    if num_threads <= 1:
        return NULL_POOL
    sharded = _experts_are_sharded(model)
    if sharded is not False:
        logger.warning(
            "MoE loader threading requested (%d threads) but the experts are "
            "%s; staying single-threaded, because a non-local expert makes "
            "weight_loader's return value steer the dispatch loop.",
            num_threads,
            "sharded across ranks" if sharded else "not recognisable")
        return NULL_POOL
    return _Pool(num_threads)
