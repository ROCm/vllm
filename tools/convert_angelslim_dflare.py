#!/usr/bin/env python3
"""Convert an AngelSlim DFlare checkpoint to the vLLM draft contract."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-hidden-size", type=int, default=2560)
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        default=[1, 6, 11, 16, 21, 26, 31, 36, 39],
    )
    args = parser.parse_args()

    source_weights = args.checkpoint / "model.safetensors"
    if not source_weights.exists():
        raise FileNotFoundError(source_weights)
    args.output.mkdir(parents=True, exist_ok=True)

    tensors = {}
    with safe_open(source_weights, framework="pt", device="cpu") as source:
        for key in source.keys():
            tensors[key] = source.get_tensor(key)
    save_file(tensors, args.output / "model.safetensors")

    config = {
        "architectures": ["DFlareDraftModel"],
        "model_type": "qwen3",
        "vocab_size": 262144,
        "draft_vocab_size": 262144,
        "hidden_size": len(args.target_layer_ids) * args.target_hidden_size,
        "draft_hidden_size": 2560,
        "target_hidden_size": args.target_hidden_size,
        "intermediate_size": 10240,
        "num_hidden_layers": 7,
        "num_target_layers": len(args.target_layer_ids),
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 131072,
        "layer_types": ["full_attention"] * 7,
        "dflare_config": {
            "mask_token_id": 4,
            "target_layer_ids": list(range(len(args.target_layer_ids))),
            "causal": False,
        },
        "dflash_config": {
            "mask_token_id": 4,
            "target_layer_ids": list(range(len(args.target_layer_ids))),
            "use_aux_hidden_state": False,
            "causal": False,
        },
        "eagle_aux_hidden_state_layer_ids": args.target_layer_ids,
        "torch_dtype": "bfloat16",
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    for filename in ("generation_config.json", "tokenizer_config.json"):
        source = args.checkpoint / filename
        if source.exists():
            shutil.copy2(source, args.output / filename)
    print(json.dumps({"output": str(args.output), "config": config}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
