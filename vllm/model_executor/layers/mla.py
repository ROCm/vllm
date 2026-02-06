# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch
from typing import Union
from vllm.attention.layer import MLAAttention
from vllm.config import CacheConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm._aiter_ops import rocm_aiter_ops
from vllm.utils.torch_utils import direct_register_custom_op


if current_platform.is_rocm() and rocm_aiter_ops.is_enabled():
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.fused_mxfp4_quant import fused_rms_mxfp4_quant, fused_reduce_rms_mxfp4_quant

        def rocm_aiter_triton_qkv_a_proj_layernorm_impl(
            hidden_states_quant: torch.Tensor,
            hidden_states_quant_scale: torch.Tensor,
            weight_qkv_a_proj: torch.Tensor,
            weight_scale_qkv_a_proj: torch.Tensor,
            q_a_layernorm_weight: torch.Tensor,
            q_a_layernorm_variance_epsilon: float,
            kv_a_layernorm_weight: torch.Tensor,
            kv_a_layernorm_variance_epsilon: float,
            q_lora_rank: int,
            kv_lora_rank: int,
            qk_rope_head_dim: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            qkv_lora = gemm_afp4wfp4(hidden_states_quant, weight_qkv_a_proj, hidden_states_quant_scale, weight_scale_qkv_a_proj.T, skip_reduce=True)
            q_c, kv_c, k_pe = qkv_lora.split([q_lora_rank, kv_lora_rank, qk_rope_head_dim],
                                                dim=-1,
                                            )
            k_pe_reduced = None
            k_pe_reduced_out = None
            if k_pe.dim() == 3:
                M = hidden_states_quant.shape[0]
                device = hidden_states_quant.device
                k_pe_reduced = k_pe
                k_pe_reduced_out = torch.empty((M, q_lora_rank + kv_lora_rank + qk_rope_head_dim), dtype=torch.bfloat16, device=device)[..., :qk_rope_head_dim]
            (q_c, q_c_scale), _, kv_c_normed, _, k_pe_reduced_out = fused_reduce_rms_mxfp4_quant(q_c, q_a_layernorm_weight, q_a_layernorm_variance_epsilon, 
                                                    kv_c, kv_a_layernorm_weight, kv_a_layernorm_variance_epsilon, k_pe_reduced,
                                                    res1=None,
                                                    shuffle=False,
                                                    scale_shuffle_padding=False,
                                                    dtype=torch.bfloat16,
                                                    out3=k_pe_reduced_out)
            
            if k_pe_reduced_out is not None:
                k_pe = k_pe_reduced_out
            return q_c, q_c_scale, kv_c_normed, k_pe
        
        def rocm_aiter_triton_qkv_a_proj_layernorm_fake(
            hidden_states_quant: torch.Tensor,
            hidden_states_quant_scale: torch.Tensor,
            weight_qkv_a_proj: torch.Tensor,
            weight_scale_qkv_a_proj: torch.Tensor,
            q_a_layernorm_weight: torch.Tensor,
            q_a_layernorm_variance_epsilon: float,
            kv_a_layernorm_weight: torch.Tensor,
            kv_a_layernorm_variance_epsilon: float,
            q_lora_rank: int,
            kv_lora_rank: int,
            qk_rope_head_dim: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            M = hidden_states_quant.shape[0]
            device = hidden_states_quant.device
            rocm_aiter_fp4_dtype = torch.uint8
            rocm_aiter_fp4_quant_group_size = 32
            q_c = torch.empty((M, q_lora_rank // 2), dtype=rocm_aiter_fp4_dtype, device=device)
            q_c_scale = torch.empty((M, (q_lora_rank + rocm_aiter_fp4_quant_group_size - 1) // rocm_aiter_fp4_quant_group_size), dtype=torch.float32, device=device)
            kv_c_normed = torch.empty((M, kv_lora_rank), dtype=torch.bfloat16, device=device)
            k_pe = torch.empty((M, q_lora_rank + kv_lora_rank + qk_rope_head_dim), dtype=torch.bfloat16, device=device)[..., :qk_rope_head_dim]
            return q_c, q_c_scale, kv_c_normed, k_pe
        
        direct_register_custom_op(
            op_name="rocm_aiter_triton_qkv_a_proj_layernorm",
            op_func=rocm_aiter_triton_qkv_a_proj_layernorm_impl,
            mutates_args=[],
            fake_impl=rocm_aiter_triton_qkv_a_proj_layernorm_fake,
            dispatch_key=current_platform.dispatch_key,
        )


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


# --8<-- [start:multi_head_latent_attention]
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

    # --8<-- [end:multi_head_latent_attention]

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
        hidden_states: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        hidden_states_scales = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scales = hidden_states

        # If aiter is enabled and MXFP4 quantization is enabled, use the aiter implementation
        if hidden_states.dtype == torch.uint8 and rocm_aiter_ops.is_enabled():
            q_c, q_c_scale, kv_c_normed, k_pe = torch.ops.vllm.rocm_aiter_triton_qkv_a_proj_layernorm(
                                                    hidden_states_quant=hidden_states,
                                                    hidden_states_quant_scale=hidden_states_scales,
                                                    weight_qkv_a_proj=self.fused_qkv_a_proj.weight,
                                                    weight_scale_qkv_a_proj=self.fused_qkv_a_proj.weight_scale,
                                                    q_a_layernorm_weight=self.q_a_layernorm.weight,
                                                    q_a_layernorm_variance_epsilon=self.q_a_layernorm.variance_epsilon,
                                                    kv_a_layernorm_weight=self.kv_a_layernorm.weight,
                                                    kv_a_layernorm_variance_epsilon=self.kv_a_layernorm.variance_epsilon,
                                                    q_lora_rank=self.q_lora_rank,
                                                    kv_lora_rank=self.kv_lora_rank,
                                                    qk_rope_head_dim=self.qk_rope_head_dim)
            q = self.q_b_proj(q_c, x_quant_scales = q_c_scale)[0]
        elif self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states, x_quant_scales=hidden_states_scales)[0]
            #qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c)
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]

    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
