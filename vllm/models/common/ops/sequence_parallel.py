# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
        x = torch.nn.functional.pad(x, pad)
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """This rank's slice of the padding mask, sharded exactly as `sp_shard`
    shards the tokens.

    `is_padding` must describe at least the rows of `hidden_states`; it may be
    longer, so that `ForwardContext.is_padding_full` -- the unsliced
    `max_num_tokens` buffer -- can be passed directly. It is *sliced* to the
    token count rather than checked against it: comparing a concrete length
    with a symbolic `hidden_states.shape[0]` specializes the dynamic batch
    dimension to whatever it held at trace time, and under vLLM's
    `@support_torch_compile` (which uses `mark_dynamic`, where specialization
    is a hard error rather than a silent recompile) the model then fails to
    compile at all. That is the same construct that had to be removed from
    `QuantFP8._bounded_per_tensor_scale`, and it is safe here today only
    because the two models that call this carry no `@support_torch_compile`.

    The rows `sp_shard` invents to reach a multiple of the TP size are padded
    with `True`, not `False`: they carry no token, so every consumer that
    bounds a reduction by this mask must exclude them.
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    is_padding = is_padding[:num_tokens]

    sp_pad = (-num_tokens) % tp_size
    if sp_pad > 0:
        is_padding = torch.nn.functional.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
