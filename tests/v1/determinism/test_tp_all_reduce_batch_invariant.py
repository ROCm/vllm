# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel all-reduce must not depend on the number of tokens.

NCCL and RCCL pick their algorithm, channel count and chunk boundaries from the
message size, so the order in which a given element's contributions are summed
changes with the batch size. Under ``VLLM_BATCH_INVARIANT`` the communicator is
expected to route around that.

Requires at least 4 GPUs: a 2-rank sum is order independent, so TP=2 passes even
with a batch-variant collective.
"""

import os
from pathlib import Path

import pytest
import ray
import torch
from utils import skip_unsupported

from tests.utils import (
    init_test_distributed_environment,
    multi_process_parallel,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import set_custom_all_reduce

# Token counts spanning the small-message thresholds where the library
# all-reduce switches protocol and chunking.
TOKEN_COUNTS = [1, 2, 3, 4, 5, 8, 16, 17, 32, 64, 128, 256, 512]
HIDDEN_SIZE = 4096


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_batch_invariance_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)

    # ParallelConfig turns custom all-reduce off under batch invariance, which
    # leaves the library collective in charge. Match that here, otherwise this
    # exercises CustomAllreduce and passes no matter what the fallback does.
    set_custom_all_reduce(False)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    full = torch.randn(
        max(TOKEN_COUNTS),
        HIDDEN_SIZE,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    # The first row is present in every launch, so its all-reduced value is the
    # same mathematical sum each time and may only differ by reduction order.
    first_rows = {
        num_tokens: tensor_model_parallel_all_reduce(full[:num_tokens].clone())[0]
        for num_tokens in TOKEN_COUNTS
    }

    reference = first_rows[TOKEN_COUNTS[0]]
    variant = [n for n, row in first_rows.items() if not torch.equal(row, reference)]
    assert not variant, (
        f"all-reduce of row 0 changed with the token count at {variant} "
        f"(reduced over {tp_size} ranks, hidden size {HIDDEN_SIZE})"
    )


@skip_unsupported
@pytest.mark.skipif(
    torch.accelerator.device_count() < 4,
    reason="a 2-rank sum is order independent, so this needs at least 4 GPUs",
)
@pytest.mark.parametrize("tp_size", [4])
def test_tp_all_reduce_is_batch_invariant(
    tp_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    # multi_process_parallel forwards PYTHONPATH to the ray workers, which have
    # to import this module to unpickle the remote worker.
    test_dir = str(Path(__file__).parent)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [test_dir, os.environ.get("PYTHONPATH")])),
    )
    multi_process_parallel(monkeypatch, tp_size, 1, all_reduce_batch_invariance_worker)
