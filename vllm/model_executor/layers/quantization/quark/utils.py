# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import regex as re
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant


def deep_compare(dict1: Any, dict2: Any) -> bool:
    if type(dict1) is not type(dict2):
        return False
    if isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            return False
        return all(deep_compare(dict1[k], dict2[k]) for k in dict1)
    elif isinstance(dict1, list):
        return set(dict1) == set(dict2)
    else:
        return dict1 == dict2


def should_ignore_layer(
    layer_name: str | None,
    ignore: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    if layer_name is None:
        return False

    # layer_name = model.layers.0.self_attn.qkv_proj
    # proj_name = qkv_proj
    proj_name = layer_name.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_proj_names = fused_mapping[proj_name]

        # Convert fused_name --> [shard_names]
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)
            for shard_proj_name in shard_proj_names
        ]

        # Layer should be ignored if shards are ignored.
        should_ignore_layer = None
        for shard_name in shard_names:
            should_ignore_shard = check_equal_or_regex_match(
                layer_name=shard_name, targets=ignore
            )

            # If shard_idx=0, set layer ignore to match shard.
            if should_ignore_layer is None:
                should_ignore_layer = should_ignore_shard

            # If shard_idx=1+ confirm scheme matches prior shards.
            elif should_ignore_shard != should_ignore_layer:
                raise ValueError(
                    f"Found a different quantization schemes for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme."
                )

    # Unfused layers like down_proj and o_proj will match
    # the safetensors checkpoint already.
    else:
        should_ignore_layer = check_equal_or_regex_match(
            layer_name=layer_name, targets=ignore
        )

    assert should_ignore_layer is not None
    return should_ignore_layer


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.
    """
    return any(_is_equal_or_regex_match(layer_name, target) for target in targets)


def _is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


def quant_to_mxfp4(x):
    """
    Quant the input tensor x to mxfp4 format
    """
    h, b, d = x.shape
    x, x_scales = dynamic_mxfp4_quant(x.reshape(-1, d))
    return x.view(h, b, d // 2), x_scales.view(h, b, d // 32)


def dequant_mxfp4_to_fp32(x, is_threed):
    """
    Dequant the input tensor x from mxfp4 format to fp32 format
    """
    # repeat interleave 2x because we pack mxfp4 in uint8
    x = x.repeat_interleave(2, dim=-1)
    if is_threed:
        x[..., ::2] = x[..., ::2] & 0xF
        x[..., 1::2] = x[..., 1::2] >> 4
    else:
        x[:, ::2] = x[:, ::2] & 0xF
        x[:, 1::2] = x[:, 1::2] >> 4

    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def convert_e8m0_to_fp32(x):
    """
    Convert the input tensor x from e8m0 format to fp32 format
    """
    # Convert the input tensor `x` (assumed to be in
    # e8m0 format) to float32. e8m0 is a custom 8-bit
    # floating point format with 8 bits for exponent, 0 for mantissa.
    # This means the value is essentially 2^(exponent - 127),
    #  similar to how IEEE-754 stores floats.

    # Convert x to float32 for computation, and
    # compute the power of 2 by subtracting the bias (127).
    x_f32 = 2 ** ((x.to(torch.float32)) - 127)

    # If the exponent value was 255 (i.e., 2^(128)), this
    # is a special case usually used to represent NaN or Inf.
    # Since this custom format has no mantissa, treat 2^128 as NaN.
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def quark_post_load_weights(
    qk_nope_head_dim: int,
    v_head_dim: int,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Post load weights for quark MXFP4 BMM
    """

    def _quant_and_split_weight(loaded_weight: torch.Tensor):
        W_UK, W_UV = loaded_weight.unflatten(
            0, (-1, (qk_nope_head_dim + v_head_dim))
        ).split([qk_nope_head_dim, v_head_dim], dim=1)
        W_UK, W_UK_scale = quant_to_mxfp4(W_UK.transpose(-2, -1))
        W_UV, W_UV_scale = quant_to_mxfp4(W_UV)
        W_UK_scale = W_UK_scale.contiguous()
        W_UV_scale = W_UV_scale.contiguous()
        return W_UK, W_UK_scale, W_UV, W_UV_scale

    # weight: [kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]
    # for the model with BF16 weight to use MXFP4 BMM,
    # quant the weight to U8 packed format(MXFP4*2)
    if weight.dtype == torch.bfloat16:
        W_UK, W_UK_scale, W_UV, W_UV_scale = _quant_and_split_weight(weight)
    elif weight.dtype == torch.uint8:
        assert weight_scale is not None, (
            "[Error][ROCm] weight_scale is required for U8 weight"
        )
        weight = dequant_mxfp4_to_fp32(weight, True).to(torch.bfloat16)
        weight_scale = weight_scale.repeat_interleave(32, dim=-1)
        weight_scale = convert_e8m0_to_fp32(weight_scale).to(torch.bfloat16)
        weight = weight * weight_scale
        W_UK, W_UK_scale, W_UV, W_UV_scale = _quant_and_split_weight(weight)
    else:
        raise ValueError("[Error][ROCm] Unsupported weight dtype: ", weight.dtype)

    return W_UK, W_UK_scale, W_UV, W_UV_scale
