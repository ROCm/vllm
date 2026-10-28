# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clean Multi-head Latent Attention for DeepSeek-V2/V3 and Kimi-K2 (AMD ROCm).

This is a self-contained MLA layer that owns the full attention path:

    hidden_states
      -> pre-attention ops (fused_qkv_a_proj -> fused q/kv RMSNorm -> q_b_proj;
         q_proj + kv_a_proj_with_mqa for checkpoints without q-LoRA)
      -> explicit prefill / decode split
           prefill: RoPE + cache insert (q_pe / k_pe in place)
                    -> kv_b_proj -> [k_nope | k_pe] -> prefill attention
                       (AITER fp8 asm prefill when the backend has scheduled it,
                        otherwise the selected MLA prefill backend)
                    (+ chunked-context merge, whose per-chunk gather -> kv_b_proj
                    -> K/V pack loop this layer owns)
           decode : W_UK absorb (BMM1) + RoPE + q-concat (+ fp8 quant) + cache
                    insert (AITER fused) -> impl.forward_mqa -> W_UV up-proj
      -> o_proj

The absorbed W_UK / W_UV BMMs run in the precision ``kv_b_proj`` was loaded in
(checkpoint or ``--quantization-config``): a per-tensor fp8 ``kv_b_proj`` uses
the AITER fp8 BMM, an MXFP4 one the AITER fp4 BMM, anything else a bf16 BMM
over the dequantized weight. Unquantized weights are never re-quantized.

Context parallelism: DCP shards the cache by token, so decode gathers the query
across the DCP group and LSE-combines the partial outputs. PCP keeps a full
cache replica on every rank and gathers each rank's prefill latents before the
insert; its rank-local prefill chunks then read their prefix as chunked context.
"""

from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    _init_kv_cache_quant,
    set_default_quant_scales,
    should_load_quant_weights,
)
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
    _get_kv_b_proj_input_dtype,
    accumulate_mla_context_chunk,
    init_mla_context_partial,
    split_kv_b_proj,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    resolve_quant_method,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kMxfp4Static,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import yarn_get_mscale
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend
from vllm.v1.attention.ops.dcp import MLADCPManager
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.pcp import maybe_gather_mla_latent_cache_inputs
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata

logger = init_logger(__name__)

AbsorbedBMM = Literal["bf16", "fp8", "fp4"]

# (bmm, W_K shape, W_V shape, kv cache dtype) already JIT-warmed in this process.
_WARMED_UP_BMMS: set[tuple] = set()


def _weight_quant_key(layer: nn.Module) -> QuantKey | None:
    """The weight quantization a linear was loaded with, or None if unquantized.

    Covers checkpoint methods (on the method or its per-layer scheme) and
    ``--quantization-config`` online methods.
    """
    method = getattr(layer, "quant_method", None)
    if method is None or isinstance(method, UnquantizedLinearMethod):
        return None
    for owner in (getattr(layer, "scheme", None), method):
        key = getattr(owner, "weight_quant_key", None)
        if key is not None:
            return key
    from vllm.model_executor.layers.quantization.online.mxfp4 import (
        Mxfp4OnlineLinearMethod,
    )

    if isinstance(method, Mxfp4OnlineLinearMethod):
        return kMxfp4Static
    return None


def select_absorbed_bmm(kv_b_proj: nn.Module) -> AbsorbedBMM:
    """Pick the absorbed-BMM precision from how ``kv_b_proj`` was quantized.

    The fp8 BMM takes one weight scale per tensor, so only a per-tensor fp8
    ``kv_b_proj`` maps onto it exactly. The fp4 BMM takes MXFP4 weights.
    Other formats keep a bf16 BMM over the dequantized weight.
    """
    if not rocm_aiter_ops.is_enabled():
        return "bf16"
    key = _weight_quant_key(kv_b_proj)
    if key == kFp8StaticTensorSym:
        scale = getattr(kv_b_proj, "weight_scale", None)
        if scale is not None and scale.numel() == 1:
            return "fp8"
    elif key == kMxfp4Static:
        from vllm.platforms.rocm import get_cdna_version

        if get_cdna_version() == 4:
            return "fp4"
    return "bf16"


class DeepseekV2MLAAttention(nn.Module, AttentionLayerBase):
    """DeepSeek-V2/V3 Multi-head Latent Attention for ROCm."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        # Latent "head" seen by the attention kernel / KV cache.
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_heads = num_heads
        self.num_local_heads = num_heads // tp_size

        # ---- Pre-attention projections ----
        self.fused_qkv_a_proj: MergedColumnParallelLinear | None = None
        self.q_a_layernorm: RMSNorm | None = None
        self.q_b_proj: ColumnParallelLinear | None = None
        self.q_proj: ColumnParallelLinear | None = None
        self.kv_a_proj_with_mqa: ReplicatedLinear | None = None
        if self.q_lora_rank is not None:
            # The latent Q and KV shards are replicated across TP ranks. Their
            # TP splitting happens in q_b_proj and kv_b_proj instead.
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            # DeepSeek-V2-Lite: full-rank Q, low-rank KV.
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # ---- Post-attention projection ----
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # ---- RoPE ----
        rope_parameters = dict(config.rope_parameters)
        if rope_parameters["rope_type"] != "default":
            rope_parameters["rope_type"] = (
                "deepseek_llama_scaling"
                if rope_parameters.get("attention_factor") == 1.0
                else "deepseek_yarn"
            )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )
        self.scale = self.qk_head_dim**-0.5
        if rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", False)
            mscale = yarn_get_mscale(rope_parameters["factor"], float(mscale_all_dim))
            self.scale *= mscale * mscale
        self.rms_norm_eps = config.rms_norm_eps

        # ---- Attention backend / impl / KV cache ----
        self.quant_config = quant_config
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        if self.kv_cache_dtype.endswith("_ds_mla"):
            raise ValueError(
                f"{self.kv_cache_dtype} is an NVIDIA-only MLA cache layout; use "
                "'auto' or 'fp8' on ROCm."
            )

        dtype = torch.get_default_dtype()
        self.attn_backend = get_attn_backend(
            self.head_size,
            dtype,
            self.kv_cache_dtype,
            use_mla=True,
            use_sparse=False,
            num_heads=self.num_local_heads,
        )
        _init_kv_cache_quant(self, quant_config, prefix)

        impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(  # type: ignore[assignment]
            num_heads=self.num_local_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=self.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=self.kv_b_proj,
            indexer=None,
        )
        # An fp8 cache takes an fp8 query when the backend consumes one; the
        # kernel then dequantizes it with _q_scale.
        self._quant_query = (
            is_quantized_kv_cache(self.kv_cache_dtype)
            and self.impl.supports_quant_query_input  # type: ignore[attr-defined]
        )

        # ---- Context parallelism ----
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        self.pcp_shard_decode_requests = parallel_config.pcp_shard_decode_requests
        if self.use_pcp and self.dcp_world_size > 1:
            raise NotImplementedError(
                "DeepSeek ROCm MLA supports DCP or PCP, not both together."
            )
        # Read by the MLA metadata builder for the chunked-context KV gather.
        self.dcp_manager: MLADCPManager | None = None
        if self.dcp_world_size > 1:
            if parallel_config.dcp_q_replicate:
                logger.warning_once(
                    "DeepSeek ROCm MLA ignores dcp_q_replicate and all-gathers "
                    "the decode query across the DCP group."
                )
            self.dcp_manager = MLADCPManager(
                vllm_config=vllm_config,
                device=next(self.kv_b_proj.parameters()).device,
                num_heads=self.num_local_heads,
                query_head_dim=self.head_size,
                output_head_dim=self.kv_lora_rank,
                query_dtype=current_platform.fp8_dtype()
                if self._quant_query
                else dtype,
                output_dtype=dtype,
                padded_num_heads=getattr(self.impl, "q_pad_num_heads", None),
                is_lse_base_on_e=self.impl.lse_base_on_e,
                use_pcp=False,
            )

        # Read by the MLA metadata builder (which clones it per builder) and by
        # profiling; the per-batch instance is ``attn_metadata.prefill``'s.
        self.prefill_backend = get_mla_prefill_backend(vllm_config)(
            num_heads=self.num_local_heads,
            scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            vllm_config=vllm_config,
        )
        self._chunked_prefill_workspace_size = (
            MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                vllm_config
            )
        )

        # Absorbed-BMM precision; fixed in process_weights_after_loading once
        # kv_b_proj's quantization is known.
        self._bmm: AbsorbedBMM = "bf16"
        spec_config = vllm_config.speculative_config
        num_spec = spec_config.num_speculative_tokens if spec_config else 0
        self._max_decode_tokens = min(
            1024, vllm_config.scheduler_config.max_num_seqs * (1 + (num_spec or 0))
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])
        # The cache as one row per slot, for kernels that index it by slot.
        self._kv_cache_rows: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # AttentionLayerBase interface
    # ------------------------------------------------------------------
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)
        try:
            self._kv_cache_rows = self.kv_cache.view(-1, 1, self.head_size)
        except RuntimeError:
            self._kv_cache_rows = None

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        return MLAAttentionSpec(  # type: ignore[call-arg]
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=self.kv_cache_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Absorb ``kv_b_proj`` into decode-time ``W_UK`` / ``W_UV`` and stage
        the RoPE tables the fused epilogues read.

        ``kv_b_proj`` produces ``[k_nope; v]`` per head from the ``kv_lora_rank``
        latent. For the MQA decode path we pre-split it so that queries are
        projected into latent space by ``W_UK`` and the attention output is
        projected back to ``v`` by ``W_UV`` -- avoiding materializing full K/V.
        """
        self.impl.process_weights_after_loading(act_dtype)

        # Dequantized (L, N, P) / (L, N, V).
        W_UK, W_UV = split_kv_b_proj(
            self.kv_b_proj,
            act_dtype,
            self.kv_lora_rank,
            self.num_local_heads,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )
        self._bmm = select_absorbed_bmm(self.kv_b_proj)
        # Copy in place on reload so captured graphs stay valid.
        if self._bmm == "fp8":
            # The dequantized weight sits on the per-tensor fp8 grid of this
            # scale, so dividing it back out recovers the loaded fp8 values.
            scale = self.kv_b_proj.weight_scale.detach().reshape(1).float()
            fp8_dtype = current_platform.fp8_dtype()
            for name, w in (
                ("W_K", W_UK.transpose(0, 1)),  # (N, L, P)
                ("W_V", W_UV.permute(1, 2, 0)),  # (N, V, L)
            ):
                replace_parameter(
                    self, name, (w.float() / scale).to(fp8_dtype).contiguous()
                )
                replace_parameter(self, f"{name}_scale", scale.clone())
        elif self._bmm == "fp4":
            from vllm.model_executor.layers.quantization.quark.utils import (
                quark_quantize_weight_to_mxfp4,
            )

            # W_UV contracts over L, the axis kv_b_proj's MXFP4 blocks run
            # along, so it keeps the loaded values. W_UK contracts over P and
            # has to be re-blocked along P.
            W_K, W_K_scale = quark_quantize_weight_to_mxfp4(W_UK)
            W_V, W_V_scale = quark_quantize_weight_to_mxfp4(W_UV.permute(1, 2, 0))
            for name, tensor in (
                ("W_K", W_K.transpose(0, 1)),  # (N, L, P // 2)
                ("W_K_scale", W_K_scale.transpose(0, 1)),  # (N, L, P // 32)
                ("W_V", W_V),  # (N, V, L // 2)
                ("W_V_scale", W_V_scale),  # (N, V, L // 32)
            ):
                replace_parameter(self, name, tensor.contiguous())
        else:
            # (L, N, V) -> (N, L, V)
            replace_parameter(self, "W_UV", W_UV.transpose(0, 1))
            # (L, N, P) -> (N, P, L)
            replace_parameter(self, "W_UK_T", W_UK.permute(1, 2, 0))

        # RoPE runs in the activation dtype, matching RotaryEmbedding, which
        # casts its table to the query dtype on every call. The vLLM kernels
        # read the packed [cos | sin] table; the AITER kernels take the halves
        # as separate dense [max_position, rot_dim // 2] tables.
        # The tables depend only on the config, so a weight reload keeps the
        # first ones (and the addresses captured graphs read).
        if getattr(self, "_cos_sin_cache", None) is None:
            cos_sin_cache = self.rotary_emb.cos_sin_cache.to(act_dtype)
            cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
            self.register_buffer("_cos_sin_cache", cos_sin_cache, persistent=False)
            self.register_buffer("_cos_cache", cos_cache.contiguous(), persistent=False)
            self.register_buffer("_sin_cache", sin_cache.contiguous(), persistent=False)

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state
        # dict] for more details.
        quant_method = (
            resolve_quant_method(self.quant_config, self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

        self._warmup_bmm(act_dtype)

    def _warmup_bmm(self, act_dtype: torch.dtype) -> None:
        """JIT-compile the Triton BMMs for every decode batch size up front.

        Mixed prefill/decode steps run outside CUDA graphs with any decode
        count, and each new count can select a different Triton config.
        """
        key = (
            self._bmm,
            tuple(self.W_K.shape) if self._bmm != "bf16" else None,
            tuple(self.W_V.shape) if self._bmm != "bf16" else None,
            self.kv_cache_dtype,
        )
        if self._bmm == "bf16" or key in _WARMED_UP_BMMS:
            return
        _WARMED_UP_BMMS.add(key)
        logger.info_once(
            "Warming up the AITER %s MLA BMMs for 1..%d decode tokens.",
            self._bmm,
            self._max_decode_tokens,
        )
        device = self.W_K.device
        max_tokens = self._max_decode_tokens
        N, L, R = self.num_local_heads, self.kv_lora_rank, self.qk_rope_head_dim
        q = torch.zeros(max_tokens, N, self.qk_head_dim, dtype=act_dtype, device=device)
        latent = torch.zeros(max_tokens, N, L, dtype=act_dtype, device=device)
        out = torch.empty(
            max_tokens, N * self.v_head_dim, dtype=act_dtype, device=device
        )
        fused = self._fused_absorb_supported()
        if fused:
            kv_c = torch.zeros(max_tokens, L, dtype=act_dtype, device=device)
            k_pe = torch.zeros(max_tokens, R, dtype=act_dtype, device=device)
            positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
            # No slot is written, so a one-row cache is enough.
            slots = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
            cache_rows = torch.zeros(
                1, 1, self.head_size, dtype=act_dtype, device=device
            )
        for m in range(1, max_tokens + 1):
            q_nope, q_pe = q[:m].split([self.qk_nope_head_dim, R], dim=-1)
            if fused:
                self._fused_absorb_rope_cache(
                    q_nope,
                    q_pe,
                    kv_c[:m],
                    k_pe[:m],
                    positions[:m],
                    slots[:m],
                    cache_rows,
                )
            else:
                self._absorb_q(q_nope)
            self._v_up_proj(latent[:m], out[:m])

    def _absorb_q(self, q_nope: torch.Tensor) -> torch.Tensor:
        """BMM1: ``(B, N, P) -> (B, N, L)`` with ``W_UK``, token-major."""
        x = q_nope.transpose(0, 1)  # (N, B, P)
        ql_nope = q_nope.new_empty(
            (q_nope.shape[0], self.num_local_heads, self.kv_lora_rank)
        )
        if self._bmm == "fp8":
            rocm_aiter_ops.triton_fp8_bmm(
                x,
                self.W_K,
                self.W_K_scale,
                group_size=128,
                transpose_bm=True,
                YQ=ql_nope,
            )
        elif self._bmm == "fp4":
            rocm_aiter_ops.batched_gemm_a16wfp4(
                x,
                self.W_K,
                self.W_K_scale,
                Y=ql_nope,
                transpose_bm=True,
                prequant=True,
            )
        else:
            # (N, B, P) x (N, P, L) -> (N, B, L), written transposed.
            torch.bmm(x, self.W_UK_T, out=ql_nope.transpose(0, 1))
        return ql_nope

    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """Project latent attention output back to ``v`` via ``W_UV`` (bmm)."""
        # (B, N, L) -> (N, B, L)
        x = x.reshape(-1, self.num_local_heads, self.kv_lora_rank).transpose(0, 1)
        out = out.view(-1, self.num_local_heads, self.v_head_dim)
        if self._bmm == "fp8":
            rocm_aiter_ops.triton_fp8_bmm(
                x,
                self.W_V,
                self.W_V_scale,
                group_size=128,
                transpose_bm=True,
                YQ=out,
            )
        elif self._bmm == "fp4":
            rocm_aiter_ops.batched_gemm_a16wfp4(
                x,
                self.W_V,
                self.W_V_scale,
                Y=out,
                transpose_bm=True,
                prequant=True,
            )
        else:
            # (N, B, L) x (N, L, V) -> (N, B, V) written transposed into (B, N, V)
            torch.bmm(x, self.W_UV, out=out.transpose(0, 1))

    def _attn_read_kv_cache(self) -> torch.Tensor:
        """Latent cache as seen by the attention read kernels.

        A per-tensor fp8 cache is allocated as ``uint8``; view it as fp8 so the
        backend reads it as E4M3 rather than as raw bytes.
        """
        if is_quantized_kv_cache(self.kv_cache_dtype):
            return self.kv_cache.view(current_platform.fp8_dtype())
        return self.kv_cache

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _normalize_q_kv(
        self, q_c: torch.Tensor, kv_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.q_a_layernorm is not None
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.fused_mla_dual_rms_norm(
                q_c,
                self.q_a_layernorm.weight,
                kv_c,
                self.kv_a_layernorm.weight,
                self.q_a_layernorm.variance_epsilon,
                self.kv_a_layernorm.variance_epsilon,
            )
        return self.q_a_layernorm(q_c), self.kv_a_layernorm(kv_c)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None
            assert self.q_b_proj is not None
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_c, k_pe = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            q_c, kv_c_normed = self._normalize_q_kv(q_c, kv_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.q_proj is not None
            assert self.kv_a_proj_with_mqa is not None
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed = self.kv_a_layernorm(kv_c)
            q = self.q_proj(hidden_states)[0]
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)

        attn_out = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._attention(positions, q, kv_c_normed, k_pe, attn_out)
        return self.o_proj(attn_out)[0]

    @eager_break_during_capture
    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> None:
        """Run MLA for one batch, writing ``attn_out`` in place.

        ``q`` is ``[T, N, P + R]`` (pre-RoPE), ``kv_c_normed`` is ``[T, L]`` and
        ``k_pe`` is ``[T, R]`` (pre-RoPE, possibly a strided view).
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            self._profile_run(kv_c_normed)
            attn_out.zero_()
            return
        if isinstance(attn_metadata_raw, list):
            # Speculative decoding: [0] is the base-model metadata dict.
            attn_metadata_raw = attn_metadata_raw[0]
        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = cast("MLACommonMetadata", attn_metadata_raw[self.layer_name])

        slot_mapping_by_layer = forward_context.slot_mapping
        assert isinstance(slot_mapping_by_layer, dict)
        slot_mapping = slot_mapping_by_layer[self.layer_name]

        num_actual_toks = attn_metadata.num_actual_tokens
        num_mqa_tokens = attn_metadata.num_decode_tokens

        cache_written = False
        if self.use_pcp:
            # Every PCP rank keeps a full cache replica but computes only its
            # own slice of the prefill tokens, so the latents of all ranks are
            # gathered before the insert. This runs over the padded batch: the
            # gathered slot mapping is laid out per padded rank-local token.
            ops.rotary_embedding(
                positions,
                q[..., self.qk_nope_head_dim :],
                k_pe,
                self.qk_rope_head_dim,
                self._cos_sin_cache,
                False,
            )
            cache_kv_c, cache_k_pe, cache_slots = maybe_gather_mla_latent_cache_inputs(
                kv_c_normed,
                k_pe,
                slot_mapping,
                num_mqa_tokens,
                use_pcp=True,
                pcp_shard_decode_requests=self.pcp_shard_decode_requests,
            )
            assert cache_slots is not None
            ops.concat_and_cache_mla(
                cache_kv_c,
                cache_k_pe,
                self.kv_cache,
                cache_slots,
                self.kv_cache_dtype,
                self._k_scale,
            )
            cache_written = True
            # Padded rows feed collectives that gather across PCP ranks.
            attn_out[num_actual_toks:].zero_()

        # Inputs and outputs may be padded for CUDA graphs.
        q = q[:num_actual_toks]
        kv_c_normed = kv_c_normed[:num_actual_toks]
        k_pe = k_pe[:num_actual_toks]
        positions = positions[:num_actual_toks]
        attn_out = attn_out[:num_actual_toks]

        # Decode tokens are laid out first, prefill tokens after. Unless PCP
        # already inserted the batch, each side writes its own cache slots.
        num_mha_tokens = num_actual_toks - num_mqa_tokens

        if num_mha_tokens > 0:
            self._forward_prefill(
                q[num_mqa_tokens:],
                kv_c_normed[num_mqa_tokens:],
                k_pe[num_mqa_tokens:],
                positions[num_mqa_tokens:],
                slot_mapping[num_mqa_tokens:num_actual_toks],
                attn_metadata,
                attn_out[num_mqa_tokens:],
                cache_written,
            )

        if num_mqa_tokens > 0:
            self._forward_decode(
                q[:num_mqa_tokens],
                kv_c_normed[:num_mqa_tokens],
                k_pe[:num_mqa_tokens],
                positions[:num_mqa_tokens],
                slot_mapping[:num_mqa_tokens],
                attn_metadata,
                attn_out[:num_mqa_tokens],
                cache_written,
            )

    def _profile_run(self, kv_c_normed: torch.Tensor) -> None:
        """Claim the worst-case transient memory of a real step.

        The largest transient is ``kv_b_proj``'s output for a full
        chunked-context workspace in ``_compute_prefill_context``.
        """
        _ = torch.empty(
            (
                self._chunked_prefill_workspace_size,
                self.num_local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ),
            device=kv_c_normed.device,
            dtype=kv_c_normed.dtype,
        )
        self.impl.reserve_workspace()

    def _rope_and_cache(
        self,
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_normed: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """RoPE ``q_pe`` / ``k_pe`` in place and insert ``[kv_c | k_pe]``."""
        if self.dcp_world_size == 1:
            ops.concat_and_cache_mla_rope_fused(
                positions,
                q_pe,
                k_pe,
                kv_c_normed,
                self._cos_sin_cache,
                False,
                slot_mapping,
                self.kv_cache,
                self.kv_cache_dtype,
                self._k_scale,
            )
            return
        # DCP leaves tokens this rank does not store without a slot; the fused
        # kernel skips their RoPE, but they are still attended to.
        ops.rotary_embedding(
            positions, q_pe, k_pe, self.qk_rope_head_dim, self._cos_sin_cache, False
        )
        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe,
            self.kv_cache,
            slot_mapping,
            self.kv_cache_dtype,
            self._k_scale,
        )

    # ---- Decode ----
    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        out: torch.Tensor,
        cache_written: bool,
    ) -> None:
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        mqa_q = self._decode_query(
            q_nope, q_pe, kv_c_normed, k_pe, positions, slot_mapping, cache_written
        )
        if self.dcp_manager is not None:
            assert self.dcp_manager.query_gather is not None
            mqa_q = self.dcp_manager.query_gather(mqa_q)

        latent_out, lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
            mqa_q, self._attn_read_kv_cache(), attn_metadata, self
        )

        if self.dcp_manager is not None:
            # Each rank attended over its shard of the cache; merge by LSE.
            assert lse is not None
            assert attn_metadata.decode is not None
            latent_out = self.dcp_manager.combine(
                latent_out,
                lse,
                seq_lens=attn_metadata.decode.seq_lens,
                query_start_loc=attn_metadata.query_start_loc[
                    : attn_metadata.num_decodes + 1
                ],
            )
        self._v_up_proj(latent_out, out=out)

    def _fused_absorb_supported(self) -> bool:
        """Whether one AITER launch can do BMM1 + RoPE + concat + cache insert.

        Those kernels write a bf16 query and an unscaled cache entry, so an fp8
        cache goes through the separate quantizing epilogue instead.
        """
        return self._bmm != "bf16" and not is_quantized_kv_cache(self.kv_cache_dtype)

    def _fused_absorb_rope_cache(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
            fused_fp4_bmm_rope_cat_and_cache_mla,
            fused_fp8_bmm_rope_cat_and_cache_mla,
        )

        args = (
            q_nope.transpose(0, 1),  # (N, B, P)
            self.W_K,
            self.W_K_scale,
            q_pe,
            kv_c_normed.unsqueeze(1),
            k_pe.unsqueeze(1),
            cache_rows,
            slot_mapping.flatten(),
            positions,
            self._cos_cache,
            self._sin_cache,
        )
        if self._bmm == "fp4":
            mqa_q, *_ = fused_fp4_bmm_rope_cat_and_cache_mla(
                *args,
                transpose_bm=True,
                prequant=True,
                is_neox=False,
                q_out_dtype=q_pe.dtype,
            )
        else:
            mqa_q, *_ = fused_fp8_bmm_rope_cat_and_cache_mla(
                *args,
                group_size=128,
                transpose_bm=True,
                is_neox=False,
                q_out_dtype=q_pe.dtype,
            )
        return mqa_q

    def _decode_query(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache_written: bool,
    ) -> torch.Tensor:
        """Build ``mqa_q = [W_UK(q_nope) | RoPE(q_pe)]`` in the dtype
        ``forward_mqa`` reads, inserting the decode latents unless the cache
        already holds them."""
        fuse = rocm_aiter_ops.is_enabled() and not cache_written
        if fuse and self._fused_absorb_supported() and self._kv_cache_rows is not None:
            return self._fused_absorb_rope_cache(
                q_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                positions,
                slot_mapping,
                self._kv_cache_rows,
            )

        num_tokens = q_nope.shape[0]
        ql_nope = self._absorb_q(q_nope)
        if fuse:
            from aiter.ops.cache import fused_qk_rope_concat_and_cache_mla

            mqa_q = torch.empty(
                (num_tokens, self.num_local_heads, self.head_size),
                dtype=current_platform.fp8_dtype()
                if self._quant_query
                else q_nope.dtype,
                device=q_nope.device,
            )
            # DCP gathers every token's query, including the tokens this rank
            # stores no slot for.
            dcp_kwargs = {"compute_all_q_rope": True} if self.dcp_world_size > 1 else {}
            # RoPE on q_pe / k_pe, mqa_q = [ql_nope | q_pe] (+ static fp8 quant),
            # and the latent cache insert, in one launch.
            fused_qk_rope_concat_and_cache_mla(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                self._attn_read_kv_cache(),
                mqa_q,
                slot_mapping.flatten(),
                self._k_scale,
                self._q_scale,
                positions,
                self._cos_cache,
                self._sin_cache,
                is_neox=False,
                is_nope_first=True,
                **dcp_kwargs,
            )
            return mqa_q

        if not cache_written:
            self._rope_and_cache(positions, q_pe, k_pe, kv_c_normed, slot_mapping)
        mqa_q = torch.cat([ql_nope, q_pe], dim=-1)
        if self._quant_query:
            mqa_q = ops.scaled_fp8_quant(mqa_q.view(num_tokens, -1), self._q_scale)[
                0
            ].view(mqa_q.shape)
        return mqa_q

    # ---- Prefill ----
    def _concat_k(self, k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
        """``[k_nope | k_pe]`` with the single-head ``k_pe`` broadcast to every
        head, written into one preallocated tensor (no expanded ``torch.cat``)."""
        k = k_nope.new_empty((*k_nope.shape[:-1], self.qk_head_dim))
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe.unsqueeze(1)
        return k

    def _use_fp8_asm_prefill(self, attn_metadata: "MLACommonMetadata") -> bool:
        """Whether the AITER backend scheduled its fp8 asm prefill for this batch.

        Mirrors ``AiterMLAImpl.forward_mha``: the kernel needs the fp8 PS
        metadata the builder produces, and it has no chunked-context merge.
        """
        prefill = attn_metadata.prefill
        assert prefill is not None
        return (
            getattr(self.impl, "_fp8_prefill_enabled", False)
            and getattr(attn_metadata, "fp8_prefill_qo_indptr", None) is not None
            and prefill.chunked_context is None
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        out: torch.Tensor,
        cache_written: bool,
    ) -> None:
        prefill = attn_metadata.prefill
        assert prefill is not None
        assert prefill.prefill_backend is not None

        if not cache_written:
            self._rope_and_cache(
                positions,
                q[..., self.qk_nope_head_dim :],
                k_pe,
                kv_c_normed,
                slot_mapping,
            )

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if self._use_fp8_asm_prefill(attn_metadata):
            impl = self.impl
            if impl._fused_prefill_qkv:  # type: ignore[attr-defined]
                from aiter.ops.triton.fusions.fused_mla_prefill_qkv_fp8 import (
                    fused_mla_prefill_qkv_fp8,
                )

                from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
                    AiterMLAHelper,
                )

                # The k_pe concat, the head padding and the fp8 cast in one launch.
                q, k, v = fused_mla_prefill_qkv_fp8(
                    q,
                    k_nope,
                    v,
                    k_pe,
                    AiterMLAHelper.get_fp8_prefill_num_heads(self.num_local_heads),
                    current_platform.fp8_dtype(),
                )
            else:
                k = self._concat_k(k_nope, k_pe)
            impl._mla_fp8_prefill_attn(q, k, v, attn_metadata, out)  # type: ignore[attr-defined]
            return

        k = self._concat_k(k_nope, k_pe)
        if prefill.q_data_type == current_platform.fp8_dtype():
            q = q.to(prefill.q_data_type)
            k = k.to(prefill.q_data_type)
            v = v.to(prefill.q_data_type)

        has_context = prefill.chunked_context is not None
        out_3d = out.view(-1, self.num_local_heads, self.v_head_dim)
        # When there is no chunked context, backends that honor `out` write the
        # attention result straight into it, avoiding a slice+flatten+copy.
        writes_out = not has_context and prefill.prefill_backend.supports_out()
        output_prefill = prefill.prefill_backend.run_prefill_new_tokens(
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
            out=out_3d if writes_out else None,
        )

        if has_context:
            if self.dcp_world_size > 1:
                # The context is sharded across the DCP group; the backend
                # all-gathers each chunk before the up-projection.
                context_output, context_lse = (
                    self.impl._context_parallel_compute_prefill_context(  # type: ignore[attr-defined]
                        q,
                        self._attn_read_kv_cache(),
                        attn_metadata,
                        k_scale=self._k_scale,
                        dcp_world_size=self.dcp_world_size,
                    )
                )
            else:
                context_output, context_lse = self._compute_prefill_context(
                    q, attn_metadata
                )
            suffix_output, suffix_lse = output_prefill
            merge_attn_states(
                output=out_3d,
                prefix_output=context_output[..., : self.v_head_dim],
                prefix_lse=context_lse,
                suffix_output=suffix_output[..., : self.v_head_dim],
                suffix_lse=suffix_lse,
            )
        elif not writes_out:
            assert isinstance(output_prefill, torch.Tensor)
            out.copy_(output_prefill[..., : self.v_head_dim].flatten(start_dim=-2))

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend the prefill queries over their cached context, chunk by chunk.

        Per chunk: gather the paged latent into the workspace, up-project it with
        ``kv_b_proj``, pack ``(k, v)`` and run the prefill backend's context
        kernel; partials are merged by LSE into one context partial.

        ``q`` already carries ``prefill.q_data_type`` (cast by the caller). An
        fp8 query reads the fp8 cache in its stored layout; anything else lands
        in the workspace dequantized to the model dtype.
        """
        prefill = attn_metadata.prefill
        assert prefill is not None
        prefill_backend = prefill.prefill_backend
        assert prefill_backend is not None
        chunked_context = prefill.chunked_context
        assert chunked_context is not None

        fp8_prefill = q.dtype == current_platform.fp8_dtype()
        workspace = chunked_context.workspace
        kv_cache = self._attn_read_kv_cache()
        kv_b_proj_input_dtype = _get_kv_b_proj_input_dtype(self.kv_b_proj, fp8_prefill)

        output: torch.Tensor | None = None
        output_lse: torch.Tensor | None = None
        for chunk in chunked_context.chunks:
            toks = chunk.num_context_tokens
            block_table = prefill.block_table[chunk.request_slice]
            if fp8_prefill:
                ops.cp_gather_cache(
                    src_cache=kv_cache,
                    dst=workspace[:toks],
                    block_table=block_table,
                    cu_seq_lens=chunk.cu_seq_lens,
                    batch_size=chunk.num_requests,
                    seq_starts=chunk.starts,
                )
            else:
                ops.gather_and_maybe_dequant_cache(
                    src_cache=kv_cache,
                    dst=workspace,
                    block_table=block_table,
                    cu_seq_lens=chunk.cu_seq_lens,
                    token_to_seq=chunk.token_to_seq,
                    num_tokens=toks,
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=self._k_scale,
                    seq_starts=chunk.starts,
                )

            gathered = workspace[:toks]
            kv_c_normed = gathered[..., : self.kv_lora_rank]
            if kv_b_proj_input_dtype is not None:
                kv_c_normed = kv_c_normed.to(kv_b_proj_input_dtype)
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            # The gathered k_pe was rotated on the way into the cache.
            k_pe = gathered[..., self.kv_lora_rank :]
            if fp8_prefill:
                kv_nope = kv_nope.to(q.dtype)
                k_pe = k_pe.to(q.dtype)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k(k_nope, k_pe)

            attn_output, attn_lse = prefill_backend.run_prefill_context_chunk(
                chunk=chunk, q=q[chunk.token_slice], k=k, v=v
            )

            if output is None:
                if (
                    len(chunked_context.chunks) == 1
                    and not chunked_context.empty_token_slices
                ):
                    # One chunk covering every prefill token: its partial *is*
                    # the context partial.
                    return attn_output, attn_lse
                output, output_lse = init_mla_context_partial(
                    chunked_context, attn_output, attn_lse, num_tokens=q.shape[0]
                )
            assert output_lse is not None
            accumulate_mla_context_chunk(
                chunk, attn_output, attn_lse, output, output_lse
            )

        assert output is not None and output_lse is not None
        return output, output_lse
