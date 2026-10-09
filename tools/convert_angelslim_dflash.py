#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert an AngelSlim Qwen3 DFlash checkpoint to the vLLM draft contract."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_source_config(checkpoint: Path) -> dict:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_vllm_config(source: dict) -> dict:
    dflash = source.get("dflash_config") or {}
    layer_ids = list(dflash.get("target_layer_ids") or [])
    if not layer_ids:
        raise ValueError("dflash_config.target_layer_ids is required")
    if dflash.get("mask_token_id") is None:
        raise ValueError("dflash_config.mask_token_id is required")

    hidden_size = int(source["hidden_size"])
    num_layers = int(source["num_hidden_layers"])
    block_size = int(source.get("block_size", 16))
    slot_ids = list(range(len(layer_ids)))
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": source.get("model_type", "qwen3"),
        "vocab_size": int(source["vocab_size"]),
        "draft_vocab_size": int(source["vocab_size"]),
        "hidden_size": hidden_size,
        "intermediate_size": int(source["intermediate_size"]),
        "num_hidden_layers": num_layers,
        "num_attention_heads": int(source["num_attention_heads"]),
        "num_key_value_heads": int(source["num_key_value_heads"]),
        "head_dim": int(
            source.get(
                "head_dim",
                hidden_size // int(source["num_attention_heads"]),
            )
        ),
        "hidden_act": source.get("hidden_act", "silu"),
        "attention_bias": bool(source.get("attention_bias", False)),
        "rms_norm_eps": float(source.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(
            source.get(
                "rope_theta",
                (source.get("rope_parameters") or {}).get(
                    "rope_theta",
                    1000000.0,
                ),
            )
        ),
        "max_position_embeddings": int(source.get("max_position_embeddings", 40960)),
        "block_size": block_size,
        "num_speculative_tokens": block_size - 1,
        "layer_types": ["full_attention"] * num_layers,
        "dflash_config": {
            "mask_token_id": int(dflash["mask_token_id"]),
            "target_layer_ids": slot_ids,
            "use_aux_hidden_state": True,
            "causal": False,
        },
        "eagle_aux_hidden_state_layer_ids": [layer_id + 1 for layer_id in layer_ids],
        "torch_dtype": "bfloat16",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source_weights = args.checkpoint / "model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(source_weights)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_weights, args.output / "model.safetensors")
    converted = build_vllm_config(load_source_config(args.checkpoint))
    (args.output / "config.json").write_text(
        json.dumps(converted, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "config": converted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
