# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the --w4a16-prefill-dequant CLI arg and config plumbing."""

import argparse

import pytest

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs


class TestW4A16PrefillDequantArg:
    """Test that the CLI arg is parsed and flows to ModelConfig."""

    def test_default_is_off(self):
        assert ModelConfig.w4a16_prefill_dequant == "off"

    def test_engine_args_default(self):
        assert EngineArgs.w4a16_prefill_dequant == "off"

    @pytest.mark.parametrize("mode", ["off", "soft", "hard"])
    def test_valid_choices_accepted(self, mode: str):
        parser = argparse.ArgumentParser()
        parser = EngineArgs.add_cli_args(parser)
        args = parser.parse_args(
            ["--model", "dummy", f"--w4a16-prefill-dequant={mode}"]
        )
        assert args.w4a16_prefill_dequant == mode

    def test_invalid_choice_rejected(self):
        parser = argparse.ArgumentParser()
        parser = EngineArgs.add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "dummy", "--w4a16-prefill-dequant=always"])

    def test_env_var_removed(self):
        """VLLM_W4A16_PREFILL_DEQUANT env var should no longer be registered."""
        from vllm.envs import environment_variables

        assert "VLLM_W4A16_PREFILL_DEQUANT" not in environment_variables
