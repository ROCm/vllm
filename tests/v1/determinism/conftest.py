# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

import vllm.envs as envs


@pytest.fixture(autouse=True)
def enable_batch_invariant_mode(monkeypatch: pytest.MonkeyPatch):
    """Automatically enable batch invariant kernel overrides for all tests."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")


@pytest.fixture(scope="module", autouse=True)
def settle_gpu_memory_between_modules():
    """Let ROCm actually release VRAM before the next module allocates.

    Most of this suite builds `LLM(...)` directly rather than going through the
    `vllm_runner` fixture, so it never picked up the settle that
    `tests/conftest.py` applies on that fixture's teardown. ROCm reclaims
    lazily, so a heavy end-to-end module can still be holding tens of GiB when
    the next one starts -- and modules that start a server ask for a fixed
    fraction of *free* memory, so they fail at startup rather than degrading.
    That is not a determinism failure but it reads like one in a suite report.

    Module scope, not function scope: the observed pollution is across files,
    and the amdsmi query behind this costs enough (~2s) that paying it per test
    would add more to a 564-item run than the model loads it protects.

    No-op off ROCm.
    """
    yield
    from tests.utils import wait_for_rocm_memory_to_settle

    wait_for_rocm_memory_to_settle()
