# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cut selection for the prefix-hit / microbatch-split hazard.

A request admitted on a prefix cache hit against blocks another request in the
same batch is only computing now must not land in an earlier microbatch than
that writer: ubatch 0's attention is enqueued before ubatch 1's KV write, so it
would read zeros. Rather than veto the step, move the cut.
"""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

BS = 16


def _runner(computed, tables, *, backend="deepep_high_throughput", num_ubatches=2):
    """Minimal stand-in exposing only what the hazard logic reads."""
    bt = SimpleNamespace(
        block_size=BS, get_numpy_array=lambda: np.asarray(tables, dtype=np.int32)
    )
    ns = SimpleNamespace(
        parallel_config=SimpleNamespace(
            use_ubatching=True, num_ubatches=num_ubatches, all2all_backend=backend
        ),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
            block_table=SimpleNamespace(block_tables=[bt]),
        ),
    )
    # `_microbatch_split_point` calls `self._microbatch_hazard_cuts`, so the
    # stand-in has to carry it bound.
    ns._microbatch_hazard_cuts = partial(GPUModelRunner._microbatch_hazard_cuts, ns)
    return ns


def _cuts(runner, n, q):
    return GPUModelRunner._microbatch_hazard_cuts(runner, n, np.asarray(q))


def _split(runner, n, q):
    return GPUModelRunner._microbatch_split_point(runner, n, np.asarray(q))


def test_no_sharing_no_hazard():
    """Disjoint blocks: nothing to avoid, and the default cut is kept."""
    r = _runner([0, 0], [[10, 11, 0], [20, 21, 0]])
    assert _cuts(r, 2, [32, 32]) is None
    assert _split(r, 2, [32, 32]) == (True, None)


def test_pure_decode_step_exits_early():
    """Every request one token: nothing writes a block anyone could read."""
    r = _runner([64, 64], [[10, 11, 12, 13], [10, 11, 12, 13]])
    assert _cuts(r, 2, [1, 1]) is None


def test_reader_before_writer_forbids_the_cut_between_them():
    """req0 has block 10 computed; req1 is writing block 10 now."""
    # req1: computed=0, writing blocks [0, ceil(32/16)) -> table cols 0..1 = 10, 11
    r = _runner([32, 0], [[10, 11, 0, 0], [10, 11, 0, 0]])
    cuts = _cuts(r, 2, [16, 32])
    assert cuts is not None, "reader/writer pair not detected"
    assert cuts[1], "the only interior cut should be forbidden"
    # Two requests, one interior cut, and it is unsafe -> veto.
    assert _split(r, 2, [16, 32]) == (False, None)


def test_writer_before_reader_is_safe():
    """Same sharing, opposite order: the writer already lands first."""
    r = _runner([0, 32], [[10, 11, 0, 0], [10, 11, 0, 0]])
    cuts = _cuts(r, 2, [32, 16])
    assert cuts is None, "a writer preceding its reader is not a hazard"


def test_cut_is_moved_rather_than_vetoed():
    """With a safe boundary available, take it instead of refusing to split."""
    # req0 reads block 10, req2 writes it -> cuts 1 and 2 forbidden; cut 3 is free.
    tables = [[10, 0, 0], [50, 0, 0], [10, 0, 0], [70, 0, 0]]
    r = _runner([16, 0, 0, 0], tables)
    q = [16, 16, 16, 16]
    cuts = _cuts(r, 4, q)
    assert cuts is not None and cuts[1] and cuts[2] and not cuts[3]
    allow, split = _split(r, 4, q)
    assert allow is True
    assert split == 48, f"expected the cut after req2 (48 tokens), got {split}"


def test_null_block_is_not_treated_as_shared():
    """SWA/hybrid pad block tables with id 0; it must not match everything."""
    r = _runner([32, 0], [[0, 0, 0, 0], [0, 0, 0, 0]])
    assert _cuts(r, 2, [16, 32]) is None


@pytest.mark.parametrize("backend", ["deepep_low_latency", "allgather_reducescatter"])
def test_non_ht_backends_veto_instead_of_moving(backend):
    """Only HT negotiates real per-rank counts, so only HT may cut per-rank."""
    tables = [[10, 0, 0], [50, 0, 0], [10, 0, 0], [70, 0, 0]]
    r = _runner([16, 0, 0, 0], tables, backend=backend)
    assert _split(r, 4, [16, 16, 16, 16]) == (False, None)
