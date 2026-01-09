# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.attention.layer import MLAAttention
from vllm.config import CacheConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import aux_stream, direct_register_custom_op


def scp_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """All-gather for tensor model parallelism."""
    return get_tp_group().all_gatherv(tensor, dim=dim)


def scp_all_gather_fake(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Fake all-gather for tensor model parallelism (no-op)."""
    all_gather_size = tensor.shape[dim] * get_tensor_model_parallel_world_size()
    all_gather_shape = tensor.shape[:dim] + (all_gather_size,) + tensor.shape[dim + 1 :]
    out = torch.empty(all_gather_shape, dtype=tensor.dtype, device=tensor.device)
    return out


direct_register_custom_op(
    op_name="scp_all_gather",
    op_func=scp_all_gather,
    fake_impl=scp_all_gather_fake,
)


# def indexer_stream_prepare(layer_name):
#     forward_context: ForwardContext = get_forward_context()
#     self = forward_context.no_compile_layers[layer_name].mla_attn
#     self.sparse_indexer_stream.wait_stream(current_stream())
#     return

# def indexer_stream_prepare_fake(layer_name):
#     return

# direct_register_custom_op(
#     op_name="indexer_stream_prepare",
#     op_func=indexer_stream_prepare,
#     fake_impl=indexer_stream_prepare_fake,
# )

# def indexer_invoke(layer_name, hidden_states, q_c, positions, indexer_rope_emb):
#     forward_context: ForwardContext = get_forward_context()
#     self = forward_context.no_compile_layers[layer_name].mla_attn

#     with torch.cuda.stream(self.sparse_indexer_stream):
#         q_c = torch.ops.vllm.scp_all_gather(q_c, dim=0)
#         _topk_indices = self.indexer(
#             hidden_states, q_c, positions, self.indexer_rope_emb
#         )
#         self.indexer_event.record(self.sparse_indexer_stream)
#     return

# def indexer_invoke_fake(layer_name, hidden_states, q_c, positions, indexer_rope_emb):
#     return

# direct_register_custom_op(
#     op_name="indexer_invoke",
#     op_func=indexer_invoke,
#     fake_impl=indexer_invoke_fake,
# )


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None


@CustomOp.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(CustomOp):
    """MLA layer registered as CustomOp to allow OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently MLA ignores the enable/disable mechanism of CustomOp
    because there is only one in-tree implementation in forward_native.
    TODO: implement this with a new PluggableLayer mechanism.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.sparse_indexer_stream = aux_stream()
        self.indexer_event = torch.cuda.Event()

        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

    def forward_native(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        chunked_hidden_states = hidden_states
        chunked_positions = positions
        if envs.VLLM_SDP:
            chunked_hidden_states = torch.chunk(hidden_states, tp_size, dim=0)[tp_rank]
            chunked_positions = torch.chunk(positions, tp_size, dim=0)[tp_rank]
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )
            qkv_lora = self.fused_qkv_a_proj(chunked_hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            # if envs.VLLM_SDP:
            # self.sparse_indexer_stream.wait_stream(current_stream())
            q_c_sharded = q_c
            q = self.q_b_proj(q_c_sharded)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(chunked_hidden_states)[0]
            q = self.q_proj(chunked_hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                chunked_positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse:
            if envs.VLLM_SDP:
                # with torch.cuda.stream(self.sparse_indexer_stream):
                q_c = torch.ops.vllm.scp_all_gather(q_c, dim=0)
                _topk_indices = self.indexer(
                    hidden_states, q_c, positions, self.indexer_rope_emb
                )
            # self.indexer_event.record(self.sparse_indexer_stream)
            else:
                _topk_indices = self.indexer(
                    hidden_states, q_c, positions, self.indexer_rope_emb
                )

        if llama_4_scaling is not None:
            q *= llama_4_scaling
        if envs.VLLM_SDP:
            kv_lora = torch.cat([kv_c_normed, k_pe.squeeze(1)], dim=-1)
            kv_lora = torch.ops.vllm.scp_all_gather(kv_lora, dim=0)
            kv_c_normed, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            k_pe = k_pe.unsqueeze(1)
            # current_stream().wait_event(self.indexer_event)
        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(q.shape[0], self.num_heads * self.v_head_dim),
        )
        output = self.o_proj(attn_out)[0]
        if envs.VLLM_SDP:
            output = torch.ops.vllm.scp_all_gather(output, dim=0)
        return output

    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
