# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.workspace import reset_workspace_manager


def pytest_runtest_setup(item):
    # Reset the WorkspaceManager global before each test so that a previous
    # test's teardown does not leave _manager=None (causing AssertionError)
    # or leave stale GPU allocations across tests at high concurrency
    # (max-in-flight >= 32 on DPX cluster).
    reset_workspace_manager()
