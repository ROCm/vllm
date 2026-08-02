# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel all-reduce must not depend on the number of tokens.

Library collectives pick their algorithm, channel count and chunk boundaries
from the message size, so the order in which a given element's contributions are
summed changes with the batch size. Under ``VLLM_BATCH_INVARIANT`` the
communicator is expected to route around that, whichever backend it lands on.

Requires at least 4 GPUs: a 2-rank sum is order independent, so TP=2 passes even
with a batch-variant collective.
"""

import os
from pathlib import Path

import pytest
import ray
import torch

from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_marks,
    multi_process_parallel,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import set_custom_all_reduce
from vllm.platforms import current_platform

# multi_gpu_test would also wrap the test in create_new_process_for_each_test,
# whose re-import breaks the ray workers below, so take its marks alone: the
# registered `distributed` selector keeps `-m distributed` picking this test up,
# and its skipif enforces the 4 GPUs the module docstring explains are needed.
pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(), reason="requires a CUDA-alike device"
    ),
    *multi_gpu_marks(num_gpus=4),
]

# Token counts spanning the small-message thresholds where the collectives
# switch protocol, chunking, or algorithm. At world size 4 the custom all-reduce
# switches from its one-shot to its two-shot kernel at 512KiB, i.e. between 32
# and 64 tokens for the 16-bit cases and between 17 and 32 for fp32.
TOKEN_COUNTS = [1, 2, 3, 4, 5, 8, 16, 17, 32, 64, 128, 256, 512]
HIDDEN_SIZE = 4096

# Row 0 always sits at offset 0, so it lands in the first chunk of every
# decomposition and stays invariant even when the rest of the tensor does not.
# Checking it alone hides real failures.
CHECK_ROWS = [0, 1, 2, 3, 7, 15, 31]

# (dtype, exponent_spread). The spread widens the operand range so that the fp32
# accumulator inside the reduction has to round: reduction order is only
# irrelevant while the accumulator has headroom over the input significand,
# which fp32 inputs never have and bf16 loses once activations have outliers.
CASES = [
    (torch.bfloat16, 0),
    (torch.bfloat16, 20),
    (torch.float16, 10),
    (torch.float32, 0),
]


def _check_all_reduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
    use_custom_all_reduce: bool,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)

    # Both of the paths batch invariance can take have to hold. With the custom
    # all-reduce enabled it serves everything under its size limit; with it
    # disabled -- and above its limit either way -- every message falls through
    # to all-gather plus a fixed rank-order sum. The library collective is never
    # reached under the mode.
    set_custom_all_reduce(use_custom_all_reduce)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    failures = []
    for dtype, spread in CASES:
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        full = torch.randn(
            max(TOKEN_COUNTS), HIDDEN_SIZE, generator=generator, device=device
        )
        if spread:
            exponents = torch.randint(
                -spread,
                spread,
                full.shape,
                generator=generator,
                device=device,
                dtype=torch.int32,
            )
            full = full * torch.exp2(exponents.float())
        full = full.to(dtype)

        reduced = {
            num_tokens: tensor_model_parallel_all_reduce(full[:num_tokens].clone())
            for num_tokens in TOKEN_COUNTS
        }
        for row in CHECK_ROWS:
            # A row is only comparable across launches that actually contain it.
            counts = [n for n in TOKEN_COUNTS if n > row]
            reference = reduced[counts[0]][row]
            variant = [n for n in counts if not torch.equal(reduced[n][row], reference)]
            if variant:
                failures.append(
                    f"{dtype} spread=+-{spread} row={row} changed at token "
                    f"counts {variant}"
                )

    assert not failures, (
        f"all-reduce depends on the token count over {tp_size} ranks "
        f"(custom_all_reduce={use_custom_all_reduce}):\n  " + "\n  ".join(failures)
    )


# multi_process_parallel does not forward extra arguments to the remote, so bind
# the two paths as separate workers.
@ray.remote(num_gpus=1, max_calls=1)
def custom_ar_worker(monkeypatch, tp_size, pp_size, rank, port):
    _check_all_reduce(monkeypatch, tp_size, pp_size, rank, port, True)


@ray.remote(num_gpus=1, max_calls=1)
def fallback_ar_worker(monkeypatch, tp_size, pp_size, rank, port):
    _check_all_reduce(monkeypatch, tp_size, pp_size, rank, port, False)


@pytest.mark.parametrize("tp_size", [4])
@pytest.mark.parametrize(
    "worker", [custom_ar_worker, fallback_ar_worker], ids=["custom_ar", "fallback_ar"]
)
def test_tp_all_reduce_is_batch_invariant(
    tp_size: int,
    worker,
    monkeypatch: pytest.MonkeyPatch,
):
    # multi_process_parallel forwards PYTHONPATH to the ray workers, which have
    # to import this module to unpickle the remote worker.
    test_dir = str(Path(__file__).parent)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [test_dir, os.environ.get("PYTHONPATH")])),
    )
    multi_process_parallel(monkeypatch, tp_size, 1, worker)
