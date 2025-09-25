# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionLayer
from vllm.attention.ops.common import cp_lse_ag_out_rs
from vllm.attention.ops.rocm_aiter_mla import aiter_mla_decode_fwd
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group
from vllm.platforms import current_platform
from vllm.utils import cdiv
# yapf conflicts with isort for this docstring
# yapf: disable
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec

# yapf: enable


def is_aiter_mla_enabled() -> bool:
    return envs.VLLM_ROCM_USE_AITER \
        and envs.VLLM_ROCM_USE_AITER_MLA


def is_rocm_aiter_fp8bmm_enabled() -> bool:
    return current_platform.is_rocm() \
        and envs.VLLM_ROCM_USE_AITER_FP8BMM \
        and envs.VLLM_ROCM_USE_AITER


if is_aiter_mla_enabled():
    from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla

if is_rocm_aiter_fp8bmm_enabled():
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant
        as aiter_triton_fp8_bmm)
    from aiter.ops.triton.quant import dynamic_per_tensor_quant_fp8_i8

    def dynamic_per_batched_tensor_quant(
            x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
        B, M, N = x.shape
        x = x.contiguous().view(B * M, N)
        x_quant = torch.empty((B * M, N), dtype=dtype, device=x.device)
        x_quant_scale = torch.empty((1, ),
                                    dtype=torch.float32,
                                    device=x.device)
        dynamic_per_tensor_quant_fp8_i8(x_quant, x, x_quant_scale)
        x_quant = x_quant.view(B, M, N)
        return x_quant, x_quant_scale


class AiterMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_metadata_cls() -> type["AiterMLAMetadata"]:
        return AiterMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder


@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices: Optional[torch.Tensor] = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: Optional[torch.Tensor] = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: Optional[torch.Tensor] = None


class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         AiterMLAMetadata)
        assert self.kv_cache_spec.block_size == 1, "AITER MLA" \
            "only supports block size 1."

        self.compilation_config = vllm_config.compilation_config
        max_num_pages_per_req = cdiv(vllm_config.model_config.max_model_len,
                                     self.kv_cache_spec.block_size)
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device=device)
            self.paged_kv_indices = torch.zeros(max_num_pages,
                                                dtype=torch.int32,
                                                device=device)
            self.paged_kv_last_page_len = torch.zeros(max_num_reqs,
                                                      dtype=torch.int32,
                                                      device=device)

            self.qo_indptr = torch.arange(0,
                                          max_num_reqs + 1,
                                          dtype=torch.int32,
                                          device=device)

    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens_cpu: torch.Tensor,
                      seq_lens_device: torch.Tensor,
                      query_start_loc_cpu: torch.Tensor,
                      query_start_loc_device: torch.Tensor,
                      num_decode_tokens: int) -> AiterMLADecodeMetadata:
        page_size = self.kv_cache_spec.block_size
        block_table_bounds = (seq_lens_device + page_size - 1) // page_size
        device = self.device
        num_reqs = seq_lens_device.size(0)

        mask = (torch.arange(block_table_tensor.size(1),
                             dtype=block_table_tensor.dtype,
                             device=device).unsqueeze(0)
                < block_table_bounds.unsqueeze(1))
        paged_kv_indices = block_table_tensor[mask]

        paged_kv_last_page_len = seq_lens_device % page_size
        paged_kv_last_page_len = torch.where(paged_kv_last_page_len == 0,
                                             page_size, paged_kv_last_page_len)

        paged_kv_indptr = torch.cat([
            torch.zeros(1, dtype=block_table_bounds.dtype, device=device),
            block_table_bounds.cumsum(dim=0, dtype=torch.int32)
        ])

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():

            num_actual_pages = paged_kv_indices.size(0)

            self.paged_kv_indices[:num_actual_pages].copy_(paged_kv_indices,
                                                           non_blocking=True)
            self.paged_kv_indices[num_actual_pages:].fill_(-1)
            paged_kv_indices = self.paged_kv_indices[:num_actual_pages]

            self.paged_kv_indptr[:1 + num_reqs].copy_(paged_kv_indptr,
                                                      non_blocking=True)
            self.paged_kv_indptr[1 + num_reqs:].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[:1 + num_reqs]

            self.paged_kv_last_page_len[:num_reqs].copy_(
                paged_kv_last_page_len, non_blocking=True)
            self.paged_kv_last_page_len[num_reqs:].fill_(1)
            paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

            qo_indptr = self.qo_indptr[:1 + num_reqs]

        else:
            qo_indptr = torch.arange(0,
                                     num_reqs + 1,
                                     step=1,
                                     dtype=torch.int32,
                                     device=device)

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr)

        return attn_metadata


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)
        assert (num_heads == 16 or num_heads == 128), (
            f"Aiter MLA only supports 16 or 128 number of heads.\n"
            f"Provided {num_heads} number of heads.\n"
            "Try adjusting tensor_parallel_size value.")
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")
        self.fp8_dtype = current_platform.fp8_dtype()
        self.dcp_world_size: Optional[int] = None

        from aiter import flash_attn_varlen_func
        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _flash_attn_varlen_diff_headdims(self,
                                         q,
                                         k,
                                         v,
                                         return_softmax_lse=False,
                                         softmax_scale=None,
                                         **kwargs):
        output = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def _v_up_proj(self, x, out):
        if not is_rocm_aiter_fp8bmm_enabled():
            return super()._v_up_proj(x, out)
        x = x.view(-1, self.num_heads, self.kv_lora_rank)
        out = out.view(-1, self.num_heads, self.v_head_dim)
        out = aiter_triton_fp8_bmm(x,
                                   self.W_V,
                                   self.W_V_scale,
                                   group_size=128,
                                   YQ=out,
                                   transpose_bm=True,
                                   transpose_bm_in=True)
        x = out.view(-1, self.num_heads * self.v_head_dim)
        return x

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = self.get_and_maybe_dequant_weights(
            self.kv_b_proj, act_dtype).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        self.W_UK_T = W_UK.permute(1, 2, 0)
        if is_rocm_aiter_fp8bmm_enabled():
            W_K = W_UK.transpose(0, 1)
            W_V = W_UV.permute(1, 2, 0)
            self.W_K, self.W_K_scale = dynamic_per_batched_tensor_quant(
                W_K, dtype=self.fp8_dtype)
            self.W_V, self.W_V_scale = dynamic_per_batched_tensor_quant(
                W_V, dtype=self.fp8_dtype)
        else:
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1)
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0)

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
        mla_output_zeros: torch.Tensor = None,
        decode_q_out: torch.Tensor = None,
        output: torch.Tensor = None,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        B = q_nope.shape[0]

        if decode_q_out is not None:
            q = decode_q_out
        else:
            q = torch.cat([q_nope, q_pe], dim=-1)
        if mla_output_zeros is not None:
            o = mla_output_zeros
            assert o.shape[0] == B, f"{o.shape[0]=} {B=}"
            assert o.shape[
                1] == self.num_heads, f"{o.shape[1]=} {self.num_heads=}"
            assert o.shape[
                2] == self.kv_lora_rank, f"{o.shape[2]=} {self.kv_lora_rank=}"
        else:
            o = torch.zeros(B,
                            self.num_heads,
                            self.kv_lora_rank,
                            dtype=q.dtype,
                            device=q.device)

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        # max_seqlen_qo must be 1 except for MTP
        # TODO: Find the best value for MTP
        max_seqlen_qo = 1
        aiter_mla_decode_fwd(q, kv_buffer, o, self.scale,
                             attn_metadata.decode.qo_indptr, max_seqlen_qo,
                             attn_metadata.decode.paged_kv_indptr,
                             attn_metadata.decode.paged_kv_indices,
                             attn_metadata.decode.paged_kv_last_page_len)

        return self._v_up_proj(o, output)

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        positions: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if positions is None:
            return super().forward(layer=layer,
                                   q=q,
                                   k_c_normed=k_c_normed,
                                   k_pe=k_pe,
                                   kv_cache=kv_cache,
                                   attn_metadata=attn_metadata,
                                   output=output,
                                   output_scale=output_scale,
                                   output_block_scale=output_block_scale)
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for MLACommonImpl")

        if attn_metadata is None:
            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        if self.dcp_world_size is None:
            self.dcp_world_size = get_dcp_group().world_size

        fp8_attention = self.kv_cache_dtype.startswith("fp8")

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        assert attn_metadata.num_decodes is not None and \
            attn_metadata.num_prefills is not None and \
            attn_metadata.num_decode_tokens is not None

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]

        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            assert hasattr(self,
                           "rotary_emb"), f"rotary_emb not found in {self}"
            cos, sin = self.rotary_emb.cos_sin_cache.chunk(2, dim=-1)
            is_neox = self.rotary_emb.is_neox_style
            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            if is_rocm_aiter_fp8bmm_enabled():
                decode_q_out = torch.empty(
                    (num_decode_tokens, self.num_heads,
                     self.W_K.shape[1] + self.qk_rope_head_dim),
                    dtype=q.dtype,
                    device=q.device)
            if fp8_attention:
                kv_cache_og_dtype = kv_cache.dtype
                kv_cache = kv_cache.view(current_platform.fp8_dtype())
            fused_output = fused_qk_rope_cat_and_cache_mla(
                q_nope,
                q_pe,
                k_c_normed.unsqueeze(1),
                k_pe,
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                positions,
                cos,
                sin,
                layer._k_scale,
                is_neox,
                num_decode_toks_for_zeros=num_decode_tokens,
                apply_scale=(k_pe.dtype != kv_cache.dtype),
                q_out=None,
                decode_q_pe_out=decode_q_out[..., -self.qk_rope_head_dim:]
                if is_rocm_aiter_fp8bmm_enabled() else None,
                k_pe_out=k_pe,
            )
            if num_decode_tokens > 0:
                q, _, k_pe, kv_cache, mla_output_zeros = fused_output
            else:
                q, _, k_pe, kv_cache = fused_output
            if fp8_attention:
                kv_cache = kv_cache.view(kv_cache_og_dtype)

        if fp8_attention:
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        if has_prefill:
            output[num_decode_tokens:] = self._forward_prefill(
                prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache,
                attn_metadata, layer._k_scale)

        if has_decode:
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            if is_rocm_aiter_fp8bmm_enabled():
                decode_ql_nope = decode_q_out[..., :self.W_K.shape[1]] if (
                    kv_cache.numel() > 0 and positions is not None) else None
                decode_ql_nope = aiter_triton_fp8_bmm(decode_q_nope,
                                                      self.W_K,
                                                      self.W_K_scale,
                                                      group_size=128,
                                                      YQ=decode_ql_nope,
                                                      transpose_bm=True,
                                                      transpose_bm_in=True)
                return self._forward_decode(decode_ql_nope,
                                            decode_q_pe,
                                            kv_c_and_k_pe_cache=kv_cache,
                                            attn_metadata=attn_metadata,
                                            layer=layer,
                                            mla_output_zeros=mla_output_zeros,
                                            decode_q_out=decode_q_out,
                                            output=output[:num_decode_tokens])
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)

            # Pads the head_dim if necessary (for the underlying kernel)
            if self.q_pad_num_heads is not None:
                B, N, L = decode_q_pe.shape
                decode_pe_padded = decode_q_pe.new_empty(
                    (B, self.q_pad_num_heads, L))
                decode_pe_padded.resize_((B, N, L))
                decode_pe_padded.copy_(decode_q_pe)
                decode_q_pe = decode_pe_padded

            # Pads the head_dim if necessary (for the underlying kernel)
            N, B, P = decode_q_nope.shape
            _, _, L = self.W_UK_T.shape
            if self.q_pad_num_heads is not None:
                decode_ql_nope = decode_q_nope.new_empty(
                    (self.q_pad_num_heads, B, L))
                decode_ql_nope.resize_((N, B, L))

            else:
                decode_ql_nope = decode_q_nope.new_empty((N, B, L))

            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            torch.bmm(decode_q_nope, self.W_UK_T, out=decode_ql_nope)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

            if fp8_attention:
                ql_nope_shape = decode_ql_nope.shape
                decode_ql_nope, _ = ops.scaled_fp8_quant(
                    decode_ql_nope.reshape([
                        ql_nope_shape[0], ql_nope_shape[1] * ql_nope_shape[2]
                    ]), layer._q_scale)
                decode_ql_nope = decode_ql_nope.reshape(ql_nope_shape)
                q_pe_shape = decode_q_pe.shape
                decode_q_pe, _ = ops.scaled_fp8_quant(
                    decode_q_pe.reshape(
                        [q_pe_shape[0], q_pe_shape[1] * q_pe_shape[2]]),
                    layer._q_scale)
                decode_q_pe = decode_q_pe.reshape(q_pe_shape)

            decode_q = (decode_ql_nope, decode_q_pe)
            if self.dcp_world_size > 1:
                assert not fp8_attention, "DCP not support fp8 kvcache now."
                # concatenate decode_ql_nope and decode_q_pe -> (B, N, L + P)
                decode_q = torch.cat(decode_q, dim=-1)
                # decode_q do allgather in head dim.
                decode_q = get_dcp_group().all_gather(decode_q, dim=1)

            # call decode attn
            attn_out, lse = super()._forward_decode(decode_q, kv_cache,
                                                    attn_metadata, layer)

            # recorect dcp attn_out with lse.
            if self.dcp_world_size > 1:
                attn_out = cp_lse_ag_out_rs(attn_out, lse, get_dcp_group())

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])
        return output_padded
