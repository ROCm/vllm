# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from copy import copy
from typing import List, Optional

import torch
import torch.nn as nn

from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata, AttentionType)
from vllm.attention.layer import Attention
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig
from vllm.forward_context import ForwardContext, get_forward_context

from vllm.logger import init_logger
logger = init_logger(__name__)

class ROCMGPTOSSMergedAttention(Attention):
    """
    Encoder attention is a special case that doesn't need a KV Cache.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        logits_soft_cap: Optional[float] = None,
        per_layer_sliding_window: Optional[int] = None,
        use_mla: bool = False,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        attn_backend: Optional[type[AttentionBackend]] = None,
        rotary_emb: Optional[nn.Module] = None,
        qkv_linear: Optional[nn.Module] = None,
        **extra_impl_args,
    ) -> None:

        super().__init__(num_heads=num_heads,
                         head_size=head_size,
                         scale=scale,
                         num_kv_heads=num_kv_heads,
                         alibi_slopes=alibi_slopes,
                         cache_config=cache_config,
                         quant_config=quant_config,
                         logits_soft_cap=logits_soft_cap,
                         per_layer_sliding_window=per_layer_sliding_window,
                         use_mla=use_mla,
                         prefix=prefix,
                         attn_type=attn_type,
                         kv_sharing_target_layer_name=kv_sharing_target_layer_name,
                         attn_backend=attn_backend,
                        #  rotary_emb=rotary_emb, # can remove
                         **extra_impl_args)
        
        self.impl.rotary_emb = rotary_emb
        self.impl.qkv_linear = qkv_linear

        assert self.use_output == True, f"{self.use_output=}"
        assert self.use_mla == False, f"{self.use_mla=}"
        assert self.use_direct_call == False, f"{self.use_direct_call=}"
        assert self.calculate_kv_scales == False, f"{self.calculate_kv_scales=}"

        logger.info(f"{self=} {self.impl.qkv_linear=} {self.impl.rotary_emb=}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: Optional[torch.Size] = None,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        assert positions is not None, f"{positions is not None}"
        output_shape = (query.shape[0], self.num_heads * self.head_size)
        output = torch.empty(output_shape,
                            dtype=query.dtype,
                            device=query.device)
        hidden_size = output_shape[-1]
        output = output.view(-1, self.num_heads, self.head_size)
        torch.ops.vllm.unified_attention_with_output(query, key, value, output, self.layer_name, None, positions=positions)
        return output.view(-1, hidden_size)
        