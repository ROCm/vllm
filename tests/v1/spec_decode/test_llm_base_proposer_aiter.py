# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The spec-decode proposer only touches ROCM_AITER_FA when aiter is installed.

rocm_aiter_fa imports aiter at module load on ROCm. On builds without aiter
(gfx11), importing it while building the proposer's allowed attention types
crashed every V1 speculative-decoding engine at startup.
"""

import sys
import types

import pytest

import vllm._aiter_ops as aiter_ops
from vllm.v1.spec_decode.llm_base_proposer import _aiter_fa_metadata_types

AITER_FA_MODULE = "vllm.v1.attention.backends.rocm_aiter_fa"


def test_aiter_fa_metadata_skipped_without_aiter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(aiter_ops, "IS_AITER_FOUND", False)
    # Any import of the backend module now raises ImportError.
    monkeypatch.setitem(sys.modules, AITER_FA_MODULE, None)

    assert _aiter_fa_metadata_types() == []


def test_aiter_fa_metadata_included_with_aiter(monkeypatch: pytest.MonkeyPatch):
    class FakeMetadata:
        pass

    fake_module = types.ModuleType(AITER_FA_MODULE)
    fake_module.__dict__["AiterFlashAttentionMetadata"] = FakeMetadata
    monkeypatch.setattr(aiter_ops, "IS_AITER_FOUND", True)
    monkeypatch.setitem(sys.modules, AITER_FA_MODULE, fake_module)

    assert _aiter_fa_metadata_types() == [FakeMetadata]
