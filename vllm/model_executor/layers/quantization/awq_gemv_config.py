# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration loading for AWQ GEMV split-k selection on ROCm.

The AWQ GEMV HIP kernel uses split-k parallelism to improve occupancy
for small N dimensions. The optimal split-k value depends on both the
K and N dimensions and the target device. This module loads per-device
config files that specify (K, N) -> split_k mappings.

Config file format (JSON):
{
    "K,N": split_k,    // e.g. "2560,2560": 16
    ...
}

Config files are stored in awq_gemv_configs/ directory, named by device:
    device_name=Radeon_8060S_Graphics.json
"""

import functools
import json
import math
import os

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_config_file_name() -> str:
    """Construct the config file name for the current device."""
    from vllm.platforms import current_platform

    device_name = current_platform.get_device_name().replace(" ", "_")
    return f"device_name={device_name}.json"


def _validate_and_normalize_config(
    config: dict, config_file_path: str
) -> dict[tuple[int, int], int]:
    """Validate a loaded AWQ GEMV split-k config and return a normalized copy.

    Checks that every "K,N" key has positive integer components and that
    every value is a positive integer.  Raises ValueError on malformed
    entries.  Raises ValueError if the config contains no valid entries.

    Returns a new dict keyed by (K, N) tuples.
    """
    normalized: dict[tuple[int, int], int] = {}
    for key, value in config.items():
        parts = key.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid AWQ GEMV split-k config key: {key}")
        try:
            ck = int(parts[0])
            cn = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid AWQ GEMV split-k config key: {key}") from exc
        if ck <= 0 or cn <= 0:
            raise ValueError(f"Invalid AWQ GEMV split-k config key: {key}")

        if not isinstance(value, int):
            raise ValueError(
                f"Invalid AWQ GEMV split-k config value for {key}: {value}"
            )

        if value <= 0:
            raise ValueError(
                f"Invalid AWQ GEMV split-k config value for {key}: {value}"
            )

        normalized[(ck, cn)] = value

    if len(normalized) == 0:
        raise ValueError(
            f"AWQ GEMV split-k config has no valid entries: {config_file_path}"
        )

    return normalized


@functools.lru_cache
def get_awq_gemv_config() -> dict[tuple[int, int], int] | None:
    """Load the AWQ GEMV split-k config for the current device.

    Checks (in order):
    1. User-defined folder via VLLM_TUNED_CONFIG_FOLDER env var
    2. Built-in awq_gemv_configs/ directory

    Returns:
        Config dict mapping (K, N) tuples to split_k values, or None.
    """
    json_file_name = _get_config_file_name()
    config_file_paths: list[str] = []

    # User-defined config takes priority
    user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_defined_config_folder is not None:
        config_file_paths.append(
            os.path.join(user_defined_config_folder, json_file_name)
        )

    # Built-in configs
    default_config_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "awq_gemv_configs"
    )
    config_file_paths.append(os.path.join(default_config_dir, json_file_name))

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                logger.info(
                    "Using AWQ GEMV split-k config from %s",
                    config_file_path,
                )
                config = json.load(f)
                if not isinstance(config, dict):
                    raise ValueError(
                        "AWQ GEMV split-k config must be a JSON object: "
                        f"{config_file_path}"
                    )
                return _validate_and_normalize_config(config, config_file_path)

    logger.info(
        "No AWQ GEMV split-k config found for device. "
        "Using default heuristic. Config file not found at: %s",
        ", ".join(config_file_paths),
    )
    return None


def _default_split_k_heuristic(N: int) -> int:
    """Default split-k heuristic matching the original C++ logic.

    This is the fallback when no config file exists for the device.
    """
    if N <= 4096:
        return 16
    elif N <= 16384:
        return 8
    else:
        return 4


def _find_nearest_split_k(config: dict[tuple[int, int], int], K: int, N: int) -> int:
    """Find the split_k of the closest (K, N) entry in the config.

    Uses Euclidean distance in (K, N) space to find the nearest neighbor.
    """
    best_dist = math.inf
    best_sk = _default_split_k_heuristic(N)

    for (ck, cn), sk in config.items():
        dist = math.hypot(ck - K, cn - N)
        if dist < best_dist:
            best_dist = dist
            best_sk = sk

    return best_sk


def get_awq_gemv_split_k(K: int, N: int, group_size: int = 128) -> int:
    """Get the optimal split-k value for AWQ GEMV given dimensions K and N.

    Checks (in priority order):
    1. AWQ_GEMV_SPLIT_K environment variable (for manual tuning)
    2. Device-specific config file (exact (K, N) match, or nearest neighbor)
    3. Default heuristic (based on N) when no config file exists

    The JSON configs and nearest-neighbor heuristic were tuned for
    group_size=128. For other group sizes, return 0 (auto) to let the
    C++ kernel's internal heuristic choose. The env var override still
    applies regardless of group size (for manual tuning sweeps).
    TODO: expand JSON configs to cover other group sizes explicitly.

    Args:
        K: Input/reduction dimension (before any padding).
        N: Output dimension of the GEMV operation.
        group_size: Quantization group size (default 128).

    Returns:
        The split-k value to use (1, 2, 4, 8, or 16), or 0 for auto.
    """
    # Priority 1: Environment variable override (for tuning sweeps)
    env_split_k = os.environ.get("AWQ_GEMV_SPLIT_K")
    if env_split_k is not None:
        return int(env_split_k)

    if group_size != 128:
        return 0

    # Priority 2: Device config file
    config = get_awq_gemv_config()
    if config is not None:
        if (K, N) in config:
            return config[K, N]
        # No exact match - use nearest neighbor from config entries
        sk = _find_nearest_split_k(config, K, N)
        logger.debug(
            "No exact AWQ GEMV config for K=%d, N=%d; "
            "nearest neighbor gives split_k=%d",
            K,
            N,
            sk,
        )
        return sk

    # Priority 3: Default heuristic (no config file at all)
    return _default_split_k_heuristic(N)


def compute_awq_gemv_padding(
    num_groups: int,
    K: int,
    N: int,
) -> tuple[bool, int, int]:
    """Compute padding and split-k selection for AWQ GEMV.

    Priority order:
    1. AWQ_GEMV_SPLIT_K environment variable (manual override)
    2. Device config file (exact/nearest neighbor split-k)
    3. Default heuristic (based on N)

    Returns:
        Tuple of (should_pad, padded_groups, split_k).
    """
    group_size = K // num_groups if num_groups > 0 else K
    target_split_k = get_awq_gemv_split_k(K, N, group_size)

    if target_split_k == 0:
        return False, num_groups, 0

    if num_groups % target_split_k == 0:
        return False, num_groups, target_split_k

    MAX_PADDING_OVERHEAD = 0.15

    candidates = [target_split_k] + [sk for sk in [16, 8, 4, 2] if sk < target_split_k]

    for split_k in candidates:
        if num_groups % split_k == 0:
            return False, num_groups, split_k

        padded = ((num_groups + split_k - 1) // split_k) * split_k
        overhead = (padded - num_groups) / num_groups
        if overhead > MAX_PADDING_OVERHEAD:
            continue

        padded_K = padded * (K // num_groups)
        runtime_sk = get_awq_gemv_split_k(padded_K, N)
        if padded % runtime_sk == 0:
            return True, padded, runtime_sk

    return False, num_groups, target_split_k


def maybe_pad_awq_weights(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-pad AWQ weights along K so num_groups divides the target split-k.

    The awq_gemv_hip kernel parallelises over split-k, which requires
    num_groups to be divisible by the split-k factor; padding with zero groups
    buys a higher factor when the natural shape does not divide. Returns the
    tensors unchanged when padding is not applicable or not worth it.

    The matching input-side padding lives in ``_awq_gemm``, which detects
    ``qweight.shape[0] > K`` and pads the activation to match.
    """
    if group_size != 128:
        return qweight, qzeros, scales

    K = qweight.shape[0]
    N = qweight.shape[1] * 8  # unpack factor
    num_groups = qzeros.shape[0]

    should_pad, padded_groups, _split_k = compute_awq_gemv_padding(num_groups, K, N)
    if not should_pad or padded_groups <= num_groups:
        return qweight, qzeros, scales

    pad_groups = padded_groups - num_groups

    def _pad(t: torch.Tensor, rows: int) -> torch.Tensor:
        out = torch.zeros((rows, t.shape[1]), dtype=t.dtype, device=t.device)
        out[: t.shape[0]] = t
        return out

    return (
        _pad(qweight, K + pad_groups * group_size),
        _pad(qzeros, padded_groups),
        _pad(scales, padded_groups),
    )
