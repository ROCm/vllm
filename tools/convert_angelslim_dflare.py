#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert an AngelSlim DFlare checkpoint to the vLLM draft contract."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from vllm.transformers_utils.configs.dflare import dflash_config_from_dflare


def load_source_config(checkpoint: Path) -> dict:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_vllm_config(
    source_config: dict,
    *,
    target_hidden_size: int | None = None,
    target_layer_ids: list[int] | None = None,
) -> dict:
    dflare_config = source_config.get("dflare_config") or {}
    layer_ids = (
        list(target_layer_ids)
        if target_layer_ids is not None
        else list(dflare_config.get("target_layer_ids") or [])
    )
    if not layer_ids:
        raise ValueError(
            "target layer IDs are required in source dflare_config or "
            "--target-layer-ids"
        )

    draft_hidden_size = int(source_config["hidden_size"])
    resolved_target_hidden_size = int(
        target_hidden_size
        if target_hidden_size is not None
        else source_config.get("target_hidden_size", draft_hidden_size)
    )
    num_hidden_layers = int(source_config["num_hidden_layers"])
    intermediate_size = int(source_config["intermediate_size"])
    block_size = int(source_config.get("block_size", 16))
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    raw_mask_token_id = dflare_config.get(
        "mask_token_id",
        source_config.get("mask_token_id"),
    )
    if raw_mask_token_id is None:
        raise ValueError(
            "mask_token_id is required in source dflare_config or as a top-level "
            "config field"
        )
    mask_token_id = int(raw_mask_token_id)
    model_type = source_config.get("model_type")
    if not model_type:
        raise ValueError(
            "model_type is required; AngelSlim DFlare drafts usually set this "
            "to the draft transformer family (typically qwen3), not the Gemma "
            "target architecture"
        )
    slot_ids = list(range(len(layer_ids)))
    dflare_out = {
        "mask_token_id": mask_token_id,
        "target_layer_ids": slot_ids,
        "causal": False,
        "rope_layout": dflare_config.get("rope_layout", "legacy"),
    }
    return {
        "architectures": ["DFlareDraftModel"],
        "model_type": model_type,
        "vocab_size": int(source_config.get("vocab_size", 262144)),
        "draft_vocab_size": int(source_config.get("vocab_size", 262144)),
        "hidden_size": len(layer_ids) * resolved_target_hidden_size,
        "draft_hidden_size": draft_hidden_size,
        "target_hidden_size": resolved_target_hidden_size,
        "target_embedding_size": int(
            source_config.get("target_embedding_size", resolved_target_hidden_size)
        ),
        "target_head_hidden_size": int(
            source_config.get(
                "target_head_hidden_size",
                source_config.get(
                    "target_embedding_size",
                    resolved_target_hidden_size,
                ),
            )
        ),
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_target_layers": len(layer_ids),
        "num_attention_heads": int(source_config["num_attention_heads"]),
        "num_key_value_heads": int(source_config["num_key_value_heads"]),
        "head_dim": int(
            source_config.get(
                "head_dim",
                draft_hidden_size // int(source_config["num_attention_heads"]),
            )
        ),
        "rms_norm_eps": float(source_config.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(source_config.get("rope_theta", 10000.0)),
        "max_position_embeddings": int(
            source_config.get("max_position_embeddings", 131072)
        ),
        "block_size": block_size,
        "num_speculative_tokens": block_size - 1,
        "layer_types": ["full_attention"] * num_hidden_layers,
        "dflare_config": dflare_out,
        "dflash_config": dflash_config_from_dflare(dflare_out),
        "eagle_aux_hidden_state_layer_ids": [layer_id + 1 for layer_id in layer_ids],
        "torch_dtype": source_config.get("torch_dtype", "bfloat16"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-hidden-size", type=int, default=None)
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        default=None,
    )
    args = parser.parse_args()

    source_weights = args.checkpoint / "model.safetensors"
    if not source_weights.exists():
        raise FileNotFoundError(source_weights)
    source_config = load_source_config(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)

    tensors = {}
    with safe_open(source_weights, framework="pt", device="cpu") as source:
        for key in source:
            tensors[key] = source.get_tensor(key)
    save_file(tensors, args.output / "model.safetensors")

    config = build_vllm_config(
        source_config,
        target_hidden_size=args.target_hidden_size,
        target_layer_ids=args.target_layer_ids,
    )
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
