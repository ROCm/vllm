# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm replacements for DSv4 ops whose shared version is platform-neutral.

Each module here exists so ``models/deepseek_v4/common/ops/`` can stay
byte-identical to upstream: the kernel lives in aiter, and a thin subclass
keeps vLLM's warmup contract around it.
"""

from .cache_utils import dequantize_and_gather_k_cache

__all__ = [
    "dequantize_and_gather_k_cache",
]
