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

    Best effort, deliberately. `wait_for_rocm_memory_to_settle` raises when the
    devices are still busy at its timeout, and as a teardown fixture that turns
    somebody else's slow reclaim into an ERROR against a module whose own tests
    all passed. A full-suite run did exactly that: nine modules errored, every
    one of them at teardown, each having waited the full 240s first. The point
    here is to *give* reclaim a chance, not to assert that it happened -- if the
    memory is genuinely still held, the next module that needs it will say so,
    and with a message about the memory it actually wanted.

    No-op off ROCm.
    """
    yield
    import logging

    from tests.utils import wait_for_rocm_memory_to_settle

    try:
        wait_for_rocm_memory_to_settle()
    except Exception as exc:  # noqa: BLE001 - teardown must not fail the module
        logging.getLogger(__name__).warning(
            "GPU memory had not settled when this module finished, continuing "
            "anyway: %s",
            exc,
        )
