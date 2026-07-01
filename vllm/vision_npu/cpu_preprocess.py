# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU preprocessing operations for VitisAI-compiled vision models.

This module implements the CPU operations that VitisAI ExecutionProvider
normally handles automatically. When using FlexMLRT directly, we must
manually implement these operations.

For Qwen2.5-VL vision model:
- Input: pixel_values [4292, 1176] from HuggingFace processor
- Output: preprocessed [1073, 4, 1280] ready for NPU
- Postprocessing: Apply reverse_index Gather to NPU output
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessor registry — the single per-model extension point.
# Adding NPU support for a new model = write its preprocessor and register it
# here with @register_preprocessor("<model_type>"); no edits to the backend,
# the pybind, or the dispatch. Everything else (IO names/shapes/batch) is read
# generically from each cache's flexmlrt-hsi.json.
# ---------------------------------------------------------------------------
_PREPROCESSOR_REGISTRY: dict = {}


def register_preprocessor(*model_types: str):
    """Register a preprocessor builder (a class or a factory fn) for model_type(s)."""
    def deco(builder):
        for mt in model_types:
            _PREPROCESSOR_REGISTRY[mt.lower()] = builder
        return builder
    return deco


class Qwen2_5_VL_CPUPreprocessor:
    """CPU preprocessing for Qwen2.5-VL vision model before NPU execution."""

    def __init__(self, model_cache_dir: str):
        """
        Initialize CPU preprocessor with required parameters.

        Args:
            model_cache_dir: Path to NPU model cache directory containing ONNX model
        """
        import os

        import onnx

        # Load ONNX model to extract parameters
        # model_cache_dir is typically: .../qwen2_5_vl_vision_stitched_7b/vaiml_par_0
        # We need to go up two levels to find the .onnx file
        onnx_model_path = os.path.join(
            os.path.dirname(os.path.dirname(model_cache_dir)),
            "qwen2_5_vl_vision_stitched_7b.onnx",
        )

        if not os.path.exists(onnx_model_path):
            logger.warning(
                "[CPU Preprocess] ONNX not found at %s, trying alternative path",
                onnx_model_path,
            )
            # Alternative: look in parent directory
            alt_path = os.path.join(
                os.path.dirname(model_cache_dir), "qwen2_5_vl_vision_stitched_7b.onnx"
            )
            if os.path.exists(alt_path):
                onnx_model_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Cannot find ONNX model at {onnx_model_path} or {alt_path}"
                )

        logger.info("[CPU Preprocess] Loading ONNX model from %s", onnx_model_path)
        model = onnx.load(onnx_model_path)
        graph = model.graph

        # Extract parameters from ONNX model
        initializers = {init.name: init for init in graph.initializer}

        # Conv weights for patch embedding
        if "patch_embed.proj.weight" in initializers:
            weight_tensor = initializers["patch_embed.proj.weight"]
            self.conv_weight = onnx.numpy_helper.to_array(weight_tensor)
            logger.info(
                "[CPU Preprocess] Loaded conv weight: %s", self.conv_weight.shape
            )
        else:
            raise ValueError("patch_embed.proj.weight not found in ONNX model")

        # Gather indices for window reordering
        if "blocks.window_index" in initializers:
            indices_tensor = initializers["blocks.window_index"]
            self.window_index = onnx.numpy_helper.to_array(indices_tensor)
            logger.info(
                "[CPU Preprocess] Loaded window_index: %s", self.window_index.shape
            )
        else:
            raise ValueError("blocks.window_index not found in ONNX model")

        # Reverse index for final postprocessing
        if "merger.reverse_index" in initializers:
            reverse_tensor = initializers["merger.reverse_index"]
            self.reverse_index = onnx.numpy_helper.to_array(reverse_tensor)
            logger.info(
                "[CPU Preprocess] Loaded reverse_index: %s", self.reverse_index.shape
            )
        else:
            raise ValueError("merger.reverse_index not found in ONNX model")

        logger.info("[CPU Preprocess] Initialized successfully")

    def preprocess(self, pixel_values: torch.Tensor) -> np.ndarray:
        """
        Apply CPU preprocessing operations to pixel_values.

        Args:
            pixel_values: [seq_len, feature_dim] float32 tensor from HF processor
                         Expected shape: [4292, 1176]

        Returns:
            preprocessed: [1073, 4, 1280] float32 numpy array ready for NPU
        """
        # Convert to numpy
        if isinstance(pixel_values, torch.Tensor):
            pixel_values_np = pixel_values.cpu().float().numpy()
        else:
            pixel_values_np = pixel_values.astype(np.float32)

        logger.info("[CPU Preprocess] Input shape: %s", pixel_values_np.shape)

        # Operation 1: Reshape to [batch, 3, 2, 14, 14]
        # pixel_values [4292, 1176] → [4292, 3, 2, 14, 14]
        x = pixel_values_np.reshape(-1, 3, 2, 14, 14)

        # Operation 2: Conv3D for patch embedding
        # Input: [4292, 3, 2, 14, 14]
        # Weight: [1280, 3, 2, 14, 14]
        # Output: [4292, 1280, 1, 1, 1]
        out_channels = self.conv_weight.shape[0]
        batch_size = x.shape[0]
        conv_out = np.zeros((batch_size, out_channels, 1, 1, 1), dtype=np.float32)

        # Naive implementation - can be optimized with torch.nn.functional.conv3d
        for b in range(batch_size):
            for oc in range(out_channels):
                conv_out[b, oc, 0, 0, 0] = np.sum(x[b] * self.conv_weight[oc])

        # Operation 3: Reshape to [4292, 1280]
        x2 = conv_out.reshape(-1, 1280)

        # Operation 4: Reshape to [1073, 4, 1280] - merge patches 4x4
        x3 = x2.reshape(1073, 4, 1280)

        # Operation 5: Gather with window_index (reordering)
        # Note: This maintains shape [1073, 4, 1280]
        x4 = x3[self.window_index]

        logger.info("[CPU Preprocess] Output shape: %s", x4.shape)
        return x4

    def postprocess(self, npu_output: np.ndarray) -> np.ndarray:
        """
        Apply CPU postprocessing to NPU output.

        Args:
            npu_output: [1073, 3584] float32 array from NPU

        Returns:
            final_output: [1073, 3584] float32 array after reverse_index reordering
        """
        # Apply final Gather with reverse_index
        reordered = npu_output[self.reverse_index]
        logger.info(
            "[CPU Postprocess] Applied reverse_index, shape: %s", reordered.shape
        )
        return reordered


class Qwen2_5_VL_CPUPreprocessor_Optimized:
    """Optimized version using torch for Conv3D."""

    def __init__(self, model_cache_dir: str):
        """Initialize with torch-based Conv3D for faster preprocessing."""
        import os

        import onnx

        onnx_model_path = os.path.join(
            os.path.dirname(os.path.dirname(model_cache_dir)),
            "qwen2_5_vl_vision_stitched_7b.onnx",
        )

        if not os.path.exists(onnx_model_path):
            logger.warning(
                "[CPU Preprocess Optimized] ONNX not found at %s, trying alternative",
                onnx_model_path,
            )
            alt_path = os.path.join(
                os.path.dirname(model_cache_dir), "qwen2_5_vl_vision_stitched_7b.onnx"
            )
            if os.path.exists(alt_path):
                onnx_model_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Cannot find ONNX model at {onnx_model_path} or {alt_path}"
                )

        logger.info(
            "[CPU Preprocess Optimized] Loading ONNX model from %s", onnx_model_path
        )
        model = onnx.load(onnx_model_path)
        graph = model.graph
        initializers = {init.name: init for init in graph.initializer}

        # Load parameters and convert to torch
        weight_np = onnx.numpy_helper.to_array(initializers["patch_embed.proj.weight"])
        self.conv_weight = torch.from_numpy(weight_np).float()

        self.window_index = onnx.numpy_helper.to_array(
            initializers["blocks.window_index"]
        )
        self.reverse_index = onnx.numpy_helper.to_array(
            initializers["merger.reverse_index"]
        )

        # Release ONNX model from memory (saves ~600 MB CPU RAM)
        del model, graph, initializers, weight_np
        import gc

        gc.collect()
        logger.info(
            "[CPU Preprocess Optimized] Initialized with torch Conv3D "
            "(ONNX model released from memory)"
        )

    def preprocess(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Optimized preprocessing using torch.nn.functional.conv3d."""
        pixel_values = pixel_values.cpu().float()

        # Reshape to [batch, 3, 2, 14, 14]
        x = pixel_values.reshape(-1, 3, 2, 14, 14)

        # Conv3D using torch (much faster than numpy)
        import torch.nn.functional as F

        # Rearrange to [batch, channels, depth, height, width]
        conv_out = F.conv3d(
            x, self.conv_weight, bias=None, stride=(2, 14, 14), padding=(0, 0, 0)
        )  # Output: [4292, 1280, 1, 1, 1]

        # Reshape to [4292, 1280]
        x2 = conv_out.reshape(-1, 1280)

        # Reshape to [1073, 4, 1280]
        x3 = x2.reshape(1073, 4, 1280)

        # Gather with window_index
        x4_np = x3.numpy()[self.window_index]

        logger.info("[CPU Preprocess Optimized] Output shape: %s", x4_np.shape)
        return x4_np

    def postprocess(self, npu_output: np.ndarray) -> np.ndarray:
        """Apply reverse_index reordering."""
        return npu_output[self.reverse_index]


# ---------------------------------------------------------------------------
# Generic cache IO spec (read from the cache's own flexmlrt-hsi.json), so the
# runtime is batch- and shape-agnostic: it learns the NPU partition's input/
# output tensor names, shapes, and batch size B directly from the compiled cache.
# ---------------------------------------------------------------------------
def read_cache_io_spec(model_cache_dir: str) -> dict:
    """Read input/output tensor names + shapes from <cache>/0/flexmlrt-hsi.json.

    Returns {input_name, output_name, in_shape, out_shape}. ``in_shape[0]`` is
    the batch size B the cache was compiled for.
    """
    import json
    import os

    hsi = os.path.join(model_cache_dir, "0", "flexmlrt-hsi.json")
    with open(hsi) as f:
        spec = json.load(f)
    inp, out = spec["inputs"][0], spec["outputs"][0]
    return {
        "input_name": inp["tensor_name"],
        "output_name": out["tensor_name"],
        "in_shape": list(inp["cpu_shape"]),
        "out_shape": list(out["cpu_shape"]),
    }


@register_preprocessor("minicpmv", "minicpmo")
class MiniCPMV_Preprocessor:
    """Preprocessor for MiniCPM-V's fully-NPU vision cache (vpm + resampler).

    The cache runs a raw square 448x448 tile ([B,3,448,448] -> [B,64,3584]) with
    no CPU pre/post ops. vLLM hands tiles in HF ``reshape_by_patch`` layout
    ([3, 14, 14*N]); this inverts that with fold (col2im, exact) using tgt_sizes,
    resizes non-32x32 tiles to 448, then chunks the tiles into B-sized groups
    (padding the last) so the batch-B cache runs them. postprocess concatenates
    and trims back to the real tile count. Stateless (n derived from geometry).
    """

    PATCH = 14

    def __init__(self, model_cache_dir: str):
        spec = read_cache_io_spec(model_cache_dir)
        self.batch = int(spec["in_shape"][0])          # B the cache expects
        self.tile = int(spec["in_shape"][2])           # 448
        self.tokens = int(spec["out_shape"][-2])       # 64
        self.hidden = int(spec["out_shape"][-1])       # 3584

    def _to_square_tile(self, t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # invert reshape_by_patch: [C,14,14*N] -> [C,14h,14w]  (N = h*w)
        P = self.PATCH
        C = t.shape[0]
        N = int(h) * int(w)
        x = (
            t.float()
            .reshape(C, P, N, P)
            .permute(0, 1, 3, 2)
            .reshape(C * P * P, N)
        )
        img = torch.nn.functional.fold(
            x.unsqueeze(0),
            output_size=(int(h) * P, int(w) * P),
            kernel_size=(P, P),
            stride=(P, P),
        )[0]  # [C, 14h, 14w]
        if img.shape[-2:] != (self.tile, self.tile):
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(self.tile, self.tile),
                mode="bicubic", align_corners=False,
            )[0]
        return img

    def preprocess(self, pixel_values, geometry) -> list:
        """[per-tile reshape_by_patch tensors] + tgt_sizes -> list of [B,3,448,448]."""
        tgt = geometry.tolist() if hasattr(geometry, "tolist") else list(geometry)
        tiles = [
            self._to_square_tile(t, h, w).cpu().float().numpy()
            for t, (h, w) in zip(pixel_values, tgt)
        ]  # each [3,448,448]
        arr = np.stack(tiles, axis=0)  # [n,3,448,448]
        n, B = arr.shape[0], self.batch
        groups = []
        for i in range(0, n, B):
            g = arr[i:i + B]
            if g.shape[0] < B:  # pad last group up to B
                pad = np.zeros((B - g.shape[0], *g.shape[1:]), dtype=g.dtype)
                g = np.concatenate([g, pad], axis=0)
            groups.append(np.ascontiguousarray(g, dtype=np.float32))
        return groups

    def postprocess(self, outputs: list, geometry) -> np.ndarray:
        """list of [B,64,3584] -> [n,64,3584] (concat + trim padding)."""
        n = len(geometry)
        cat = np.concatenate([np.asarray(o) for o in outputs], axis=0)
        return cat[:n]


class _SingleGroupAdapter:
    """Adapt a single-input/single-output preprocessor (e.g. Qwen's) to the
    uniform list contract used by the backend:

        preprocess(pixel_values, geometry) -> [one NPU input]
        postprocess([one NPU output], geometry) -> array

    Also converts numpy -> torch for the wrapped preprocessor. Lets legacy
    single-shot preprocessors work unchanged behind the generic backend loop.
    """

    def __init__(self, inner):
        self._inner = inner

    def preprocess(self, pixel_values, geometry=None) -> list:
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values).float()
        return [self._inner.preprocess(pixel_values)]

    def postprocess(self, outputs: list, geometry=None) -> np.ndarray:
        return self._inner.postprocess(outputs[0])


@register_preprocessor("qwen2_5_vl", "qwen2_vl")
def _build_qwen_preprocessor(model_cache_dir: str):
    """Qwen2.5-VL: partial-NPU graph -> CPU-preprocess pipeline (optimized->numpy),
    wrapped in the uniform list contract (Qwen's preprocessor code is untouched)."""
    try:
        inner = Qwen2_5_VL_CPUPreprocessor_Optimized(model_cache_dir)
    except Exception as e:
        logger.warning(
            "Failed to load optimized Qwen preprocessor: %s, falling back to numpy",
            e,
        )
        inner = Qwen2_5_VL_CPUPreprocessor(model_cache_dir)
    return _SingleGroupAdapter(inner)


def get_preprocessor(model_cache_dir: str, model_type: str | None = None):
    """Look up and build the registered NPU vision preprocessor for model_type.

    Generic: adding a model needs only an @register_preprocessor entry, not an
    edit here. Raises with the known list if the model_type isn't registered.
    """
    builder = _PREPROCESSOR_REGISTRY.get((model_type or "").lower())
    if builder is None:
        raise ValueError(
            f"No NPU vision preprocessor registered for model_type={model_type!r}. "
            f"Registered: {sorted(_PREPROCESSOR_REGISTRY)}. "
            f"Add one with @register_preprocessor(\"{model_type}\")."
        )
    return builder(model_cache_dir)
