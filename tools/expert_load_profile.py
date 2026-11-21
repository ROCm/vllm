# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sum the routed experts returned by a vLLM server into an expert load profile.

Serve the model with `--enable-return-routed-experts`, send representative
requests to `/v1/completions` or `/v1/chat/completions`, and save the JSON
responses, one per line. Then:

    python tools/expert_load_profile.py --num-experts 256 \\
        -o profile.npy responses.jsonl [more.jsonl ...]

writes a float64 `[num_layers, num_experts]` array: how often each expert was
selected in each layer, over every returned token. A model can use it as the
expert load profile of its decode expert replicas (Muse Spark on ROCm:
`--additional-config '{"decode_expert_replicas": R,
"decode_expert_load_profile": "profile.npy"}'`).
"""

import argparse
import io
import json
from collections.abc import Iterable, Iterator

import numpy as np
import pybase64 as base64


def decode_routed_experts(encoded: str) -> np.ndarray:
    """A response's `routed_experts`: `[tokens, layers, top_k]` expert ids."""
    return np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)


def routed_experts_of(paths: Iterable[str]) -> Iterator[np.ndarray]:
    """Every `routed_experts` array in the JSON-lines response files."""
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                for choice in json.loads(line).get("choices", []):
                    encoded = choice.get("routed_experts")
                    if encoded:
                        yield decode_routed_experts(encoded)


def expert_load_profile(
    routed_experts: Iterable[np.ndarray], num_experts: int
) -> np.ndarray:
    """Per-layer expert selection counts `[num_layers, num_experts]` summed over
    `[tokens, layers, top_k]` routed-expert arrays (ids outside
    `[0, num_experts)` are ignored)."""
    profile: np.ndarray | None = None
    for ids in routed_experts:
        if ids.ndim != 3:
            raise ValueError(f"expected [tokens, layers, top_k], got {ids.shape}")
        num_layers = ids.shape[1]
        if profile is None:
            profile = np.zeros((num_layers, num_experts), dtype=np.float64)
        elif profile.shape[0] != num_layers:
            raise ValueError("routed_experts arrays disagree on the layer count")
        ids = ids.astype(np.int64)
        layer = np.broadcast_to(np.arange(num_layers)[None, :, None], ids.shape)
        keep = (ids >= 0) & (ids < num_experts)
        profile += np.bincount(
            (layer[keep] * num_experts + ids[keep]).ravel(),
            minlength=num_layers * num_experts,
        ).reshape(num_layers, num_experts)
    if profile is None:
        raise ValueError("no routed_experts found")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("responses", nargs="+", help="JSON-lines response files")
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("-o", "--output", required=True, help="output .npy path")
    args = parser.parse_args()
    profile = expert_load_profile(routed_experts_of(args.responses), args.num_experts)
    np.save(args.output, profile)
    print(
        f"{args.output}: {profile.shape[0]} layers x {profile.shape[1]} experts, "
        f"{int(profile.sum())} selections"
    )


if __name__ == "__main__":
    main()
