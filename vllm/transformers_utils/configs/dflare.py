# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared DFlare config helpers used by conversion, HF wrapping, and serving."""

from collections.abc import Mapping, MutableMapping
from typing import Any


def dflash_config_from_dflare(
    dflare_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the DFlash scheduling alias consumed by shared draft runtime.

    DFlare fusion reads ``dflare_config``. Parallel-query scheduling, mask
    tokens, and aux-hidden collection still look at ``dflash_config``. The
    alias copies DFlare metadata and forces ``use_aux_hidden_state=False``
    because DFlare concatenates target layers itself.
    """
    aliased = dict(dflare_config)
    aliased["use_aux_hidden_state"] = False
    return aliased


def apply_dflare_scheduling_alias(config: Any) -> Any:
    """Attach ``dflash_config`` from ``dflare_config`` when the alias is missing."""
    if isinstance(config, MutableMapping):
        dflare_config = config.get("dflare_config")
        if dflare_config is None:
            return config
        if config.get("dflash_config") is None:
            config["dflash_config"] = dflash_config_from_dflare(dflare_config)
        return config

    dflare_config = getattr(config, "dflare_config", None)
    if dflare_config is None:
        return config
    if getattr(config, "dflash_config", None) is None:
        config.dflash_config = dflash_config_from_dflare(dflare_config)
    return config
