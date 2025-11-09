# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from vllm.attention import Attention
from vllm.config import CacheConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization import QuantizationConfig
from torch import nn
import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import cdiv, direct_register_custom_op
from vllm.logger import init_logger
logger = init_logger(__name__)

if current_platform.is_rocm() and envs.VLLM_ROCM_USE_AITER:
    VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP8_QUANT = envs.VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP8_QUANT
    VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT = envs.VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT
    VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE = envs.VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE #and envs.VLLM_ROCM_USE_AITER_MLA

    if VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP8_QUANT:
        from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
        import aiter as rocm_aiter
        rocm_aiter_fp8_dtype = rocm_aiter.dtypes.fp8
        rocm_aiter_fp8_quant_group_size = 128
    
    if VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT:
        from aiter.ops.triton.fused_mxfp4_quant import fused_rms_mxfp4_quant
        rocm_aiter_fp4_dtype = torch.uint8
        rocm_aiter_fp4_quant_group_size = 32
        def rocm_aiter_triton_fused_rms_quant_rms_fp4_impl(
            q_c: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            kv_c: torch.Tensor,
            weight2: torch.Tensor,
            eps2: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (q_c, q_c_scale), _, kv_c_normed, _ = fused_rms_mxfp4_quant(q_c, weight, eps, 
                                                    kv_c, weight2, eps2, 
                                                    shuffle=False,
                                                    scale_shuffle_padding=False,
                                                    res1=None)
            return q_c, q_c_scale, kv_c_normed
    
        def rocm_aiter_triton_fused_rms_quant_rms_fp4_fake(
            q_c: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            kv_c: torch.Tensor,
            weight2: torch.Tensor,
            eps2: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            M = q_c.shape[0]
            N1 = q_c.shape[1]
            N2 = kv_c.shape[1]
            device = q_c.device
            q_c = torch.empty((M, N1 // 2), dtype=rocm_aiter_fp4_dtype, device=device)
            q_c_scale = torch.empty((M, (N1 + rocm_aiter_fp4_quant_group_size - 1 ) // rocm_aiter_fp4_quant_group_size), dtype=torch.uint8, device=device)
            kv_c_normed = torch.empty((M, N2), dtype=kv_c.dtype, device=device)
            return q_c, q_c_scale, kv_c_normed
        
        direct_register_custom_op(
            op_name="rocm_aiter_triton_fused_rms_quant_rms_fp4",
            op_func=rocm_aiter_triton_fused_rms_quant_rms_fp4_impl,
            mutates_args=[],
            fake_impl=rocm_aiter_triton_fused_rms_quant_rms_fp4_fake,
            dispatch_key=current_platform.dispatch_key,
        )

else:
    VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP8_QUANT = False
    VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT = False
    VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE = False

VLLM_ROCM_USE_AITER_MLA = envs.VLLM_ROCM_USE_AITER_MLA
logger.info(f"[Aiter] {VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE=} {VLLM_ROCM_USE_AITER_MLA=}")
logger.info(f"[Aiter] {VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP8_QUANT=}")
logger.info(f"[Aiter] {VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT=}")

@dataclass
class MLAModules:
    """Modules used in MLA.
    """
    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: Optional[torch.nn.Module]
    kv_a_proj_with_mqa: Optional[torch.nn.Module]
    q_a_layernorm: Optional[torch.nn.Module]
    q_b_proj: Optional[torch.nn.Module]
    q_proj: Optional[torch.nn.Module]


@CustomOp.register("multi_head_latent_attention")
class MultiHeadLatentAttention(CustomOp):
    """MLA layer registered as CustomOp.
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
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
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

        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size
        self.mla_attn = Attention(
            num_heads=self.num_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=scale,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb if VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE else None,
        )
        self.use_triton_fused_rmsnorm_fp8_quant = VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP8_QUANT and quant_config.get_name() == 'fp8'
        self.use_triton_fused_rmsnorm_fp4_quant = VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT and quant_config.get_name() == 'quark'
        self.use_triton_fused_rmsnorm_quant = self.q_lora_rank is not None and (self.use_triton_fused_rmsnorm_fp8_quant or self.use_triton_fused_rmsnorm_fp4_quant)

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

    def forward_native(
        self,
        positions: torch.Tensor,
        hidden_states: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        hidden_states_scales = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scales = hidden_states

        if self.use_triton_fused_rmsnorm_quant:
            assert self.fused_qkv_a_proj is not None, \
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            assert self.q_a_layernorm is not None, \
                "q_a_layernorm is required when q_lora_rank is not None"
            assert self.q_b_proj is not None, \
                "q_b_proj is required when q_lora_rank is not None"
            qkv_lora = self.fused_qkv_a_proj(hidden_states, x_quant_scales=hidden_states_scales)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            weight = self.q_a_layernorm.weight
            eps = self.q_a_layernorm.variance_epsilon
            weight2 = self.kv_a_layernorm.weight
            eps2 = self.kv_a_layernorm.variance_epsilon
            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                    dim=-1)
            if self.use_triton_fused_rmsnorm_fp4_quant:
                q_c, q_c_scale, kv_c_normed = torch.ops.vllm.rocm_aiter_triton_fused_rms_quant_rms_fp4(q_c, weight, eps, 
                                                        kv_c, weight2, eps2)
            elif self.use_triton_fused_rmsnorm_fp8_quant:
                (q_c, q_c_scale), _, kv_c_normed, _ = fused_rms_fp8_group_quant(q_c, weight, eps, 
                                                        kv_c, weight2, eps2, 
                                                        group_size=rocm_aiter_fp8_quant_group_size,
                                                        dtype_quant=rocm_aiter_fp8_dtype, 
                                                        res1=None)
            q = self.q_b_proj(q_c, x_quant_scales = q_c_scale)[0]
        elif self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, \
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            assert self.q_a_layernorm is not None, \
                "q_a_layernorm is required when q_lora_rank is not None"
            assert self.q_b_proj is not None, \
                "q_b_proj is required when q_lora_rank is not None"
            qkv_lora = self.fused_qkv_a_proj(hidden_states, x_quant_scales=hidden_states_scales)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
                
            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                    dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c)
        else:
            assert self.kv_a_proj_with_mqa is not None, \
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            assert self.q_proj is not None, \
                "q_proj is required when q_lora_rank is None"
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]
            
            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                    dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE:
            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=(hidden_states.shape[0],
                            self.num_heads * self.v_head_dim),
                positions=positions)
        else:
            q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim:], k_pe)

            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=(hidden_states.shape[0],
                            self.num_heads * self.v_head_dim))
        return self.o_proj(attn_out)[0]

    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
