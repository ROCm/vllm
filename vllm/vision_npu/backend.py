# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Abstract base class for vision NPU backends.
"""

from abc import ABC, abstractmethod

import numpy as np


class NPUVisionBackend(ABC):
    """Base class for vision processing NPU backends.

    This abstract class defines the interface that all NPU vision backends
    must implement. Different NPU implementations (FlexMLRT, ONNX Runtime, etc.)
    can subclass this to provide hardware-accelerated vision processing.
    """

    @abstractmethod
    def __init__(self, model_cache_path: str, device_name: str = "stx"):
        """Load vision model onto NPU.

        Args:
            model_cache_path: Path to pre-compiled NPU model cache
            device_name: NPU device identifier (e.g., "stx" for Strix)
        """
        pass

    @abstractmethod
    def forward(self, pixel_values, geometry=None) -> np.ndarray:
        """Run vision encoding on the NPU.

        Args:
            pixel_values: model's raw vision input (adapted by the preprocessor).
            geometry: model's per-item geometry (e.g. grid_thw for Qwen,
                tgt_sizes for MiniCPM); used or ignored by the preprocessor.

        Returns:
            embeddings: vision embeddings [tokens, hidden] (or [n, tokens, hidden]).
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output embedding dimension.

        Returns:
            Hidden dimension of output embeddings (e.g., 3584 for Qwen2.5-VL)
        """
        pass
