# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill.

On gfx1151 the backend serves decode and speculative decode with
the RDNA3.5 HIP kernel (csrc/rocm/rdna35_decode_attn.cu) on Triton's packed
HND KV cache, and everything else with TRITON_ATTN's kernels on that cache.
"""

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size, is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
    has_native_kv_cache_layout,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.ops.rdna35_hip_decode import (
    MAX_M,
    PREFILL_MIN_M,
    SUPPORTED_HEAD_SIZES,
    KernelVariant,
    expected_kv_cache_strides,
    load,
    load_prefill,
    make_prefill_scratch,
    make_scratch,
    prefill_variant_for,
    prefill_variants,
    scratch_bytes,
    variant_for,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout, KVQuantMode

logger = init_logger(__name__)


def _use_rdna35_kernel() -> bool:
    """gfx1151: the RDNA3.5 HIP decode kernel on the packed HND cache."""
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1151

    return on_gfx1151()


@dataclass
class RocmAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None

    # DFlash drafting sets this to False via CommonAttentionMetadata.
    causal: bool = True


class RocmAttentionMetadataBuilder(AttentionMetadataBuilder[RocmAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RocmAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)

        # Zero device query start locations to avoid invalid memory access in
        # the prefix prefill kernel during graph capture (#25985).
        common_attn_metadata.query_start_loc.zero_()

        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = RocmAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            causal=common_attn_metadata.causal,
        )
        return attn_metadata


class RocmAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # ROCM paged attention native C++ kernel only supports block sizes 16 and 32
        # due to shared memory (LDS) constraints on AMD GPUs.
        # See csrc/rocm/attention.cu CALL_CUSTOM_LAUNCHER_BLK macro.
        # However, vLLM allows support for any multiple of 16 via the Triton path.
        # As addressed in PR: https://github.com/vllm-project/vllm/pull/31380,
        # non-standard models (like qwen3-next with block_size 544, or qwen3_5
        # with 784 and 1056) are dynamically routed to our optimized Triton kernel
        # in `do_kv_cache_update`.
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        sizes = [32, 64, 80, 96, 128, 160, 192, 224, 256]
        if cls is RocmAttentionBackend and _use_rdna35_kernel():
            sizes.append(512)
        return sizes

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Not implemented
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        # ROCM custom attention kernel does not support sinks.
        # Callink this backend with sinks will cause it to fall back to the Triton
        # kernel, which is less efficient than the proper triton backends.
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        # ROCM_ATTN uses (2, num_blocks, ...) KV cache layout which is
        # incompatible with KV connectors that require blocks-first layout.
        return False

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_ATTN"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        if _use_rdna35_kernel():
            return RocmAttentionRdna35Impl
        return RocmAttentionImpl

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """ENCODER_DECODER is not supported because
        chunked_prefill_paged_decode's prefill kernel (context_attention_fwd)
        assumes self-attention semantics: it treats passed K/V as new tokens
        to mix with cached K/V. For cross-attention layers the encoder K/V
        are already fully cached, so mixing them again produces incorrect
        results when max_query_len > 1 (e.g. beam search).
        """
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """K and V as two head groups so the native HIP kernels address each side
        as one contiguous region (x-packed interior applied in split_kv_cache).
        On gfx1151, Triton's spec: K and V packed in the content of one head."""
        if cls is RocmAttentionBackend and _use_rdna35_kernel():
            return TritonAttentionBackend.customize_spec(spec)
        if spec.state_content_bytes is not None:
            return spec
        assert spec.head_size == spec.head_size_v, (
            "Separate K/V head groups require symmetric K/V head sizes."
        )
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=spec.num_kv_heads
            * spec.head_size
            * get_dtype_size(spec.dtype),
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The native HIP kernels hardcode group-contiguous block addressing, so
        # they need the K and V groups to span all blocks (H outermost within the layer)
        # The Triton fallbacks are stride-aware, so block-interior LBHNC is also
        # supported, just without the native kernels.
        if cls is RocmAttentionBackend and _use_rdna35_kernel():
            # The RDNA3.5 kernel addresses a page as [H, N, K|V]: keys of one
            # head contiguous in a page.  The Triton paths it falls back to run
            # faster on it too: 3-5 % prefill, 13-27 % batched decode.
            return (KVCacheLayout.LBHNC,)
        return (KVCacheLayout.LHBNC, KVCacheLayout.LBHNC)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        if _use_rdna35_kernel():
            return RocmAttentionRdna35MetadataBuilder
        return RocmAttentionMetadataBuilder


class RocmAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer

        """
        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
            sinks=self.sinks,
        )
        return output

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            layer: The attention layer, providing the q/k/v quantization scales.
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: logical [num_blocks, 2, block_size, num_kv_heads *
                head_size] under LHBNC (physically K/V-group-first)
            attn_metadata: Metadata for attention.
            output: Tensor that the attention result is written into.
            output_scale: Scale for fused output quantization.
            output_block_scale: Block scale for fused output quantization;
                not supported by this backend.

        Returns:
            shape = [num_tokens, num_heads * head_size]

        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for RocmAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # The bound view is logical [B, 2, N, H*hs]; split_kv_cache expects
        # the K/V groups first.
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            # q_scale only applies to an fp8 query; this path keeps the query
            # in full precision, so a non-1.0 q_scale is not applicable here.
            if query.dtype == self.fp8_dtype and layer._q_scale_float != 1.0:
                raise NotImplementedError(
                    "A non 1.0 q_scale with an fp8 query is not currently "
                    "supported by RocmAttentionImpl."
                )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        # Compute attention and update output up to `num_actual_tokens`.
        chunked_prefill_paged_decode(
            query=query[:num_actual_tokens],
            key=key[:num_actual_tokens] if key is not None else None,
            value=value[:num_actual_tokens] if value is not None else None,
            output=output[:num_actual_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            query_start_loc=cu_seqlens_q,
            seq_lens=seqused_k,
            max_seq_len=max_seqlen_k,
            max_query_len=max_seqlen_q,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            alibi_slopes=self.alibi_slopes,
            # self.sliding_window[0] is the FlashAttention-style left span (W - 1).
            # chunked_prefill_paged_decode expects the full window length W.
            sliding_window=1 + self.sliding_window[0],
            sm_scale=self.scale,
            output_scale=output_scale,
            sinks=self.sinks,
            causal=attn_metadata.causal,
        )

        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )

        # Reshape the input keys and values and store them in the cache.
        # Get the actual block_size from value_cache
        # value_cache shape: [num_blocks, num_heads, head_size, block_size]
        block_size = value_cache.shape[3]
        has_native_layout = has_native_kv_cache_layout(key_cache, value_cache)

        if block_size in (16, 32) and has_native_layout:
            # Normal 16, 32 with contiguous blocks: use vLLM native HIP C++ logic.
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
        else:
            # Non-standard blocks and hybrid attention/Mamba layouts need the
            # stride-aware Triton writer. The native reshape_and_cache kernel
            # assumes contiguous block storage and writes to the wrong hybrid
            # cache blocks.
            triton_reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

    def fused_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1),
            layer.num_kv_heads,  # type: ignore[attr-defined]
            layer.head_size,  # type: ignore[attr-defined]
        )
        flash_layout = False

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )


# The JIT-compiled module plus the scratch buffers sized for it.  The module is
# a pybind extension built at runtime, so it has no static type.
_Built = tuple[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

# Split-KV scratch, one set per variant and device, shared by every layer that
# runs it: layers launch one after another on one stream, and the counters are
# left clean by each launch.  Sized for the scheduler's batch, up to a budget;
# a larger batch falls back to Triton.
_SCRATCH: dict[tuple[KernelVariant, torch.device], tuple[int, Any]] = {}
_SCRATCH_BUDGET = 64 * 1024**2
# Prefill split-KV scratch, one set per shape and device, sized for all of
# the shape's rows and shared the same way.
_PREFILL_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
# Mixed batches are split only from this head size up.
_SPLIT_MIN_HEAD_SIZE = 256
_SPLIT_MIN_DECODES = 16
_SPLIT_MIN_PREFILL = 256
_SPLIT_NEEDS_LONG_PREFILL = {(2, 256)}
_SPLIT_SMALL_WINDOW = 512


class RocmAttentionRdna35MetadataBuilder(TritonAttentionMetadataBuilder):
    """Triton's metadata, with the batch ordered decodes first and the
    number of leading uniform decodes counted, so that a batch mixing
    prefills and decodes can send its decodes to the decode kernel and its
    prefills to the prefill kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        if self.reorder_batch_threshold is not None:
            self.reorder_batch_threshold = min(self.reorder_batch_threshold, MAX_M)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_decodes, num_decode_tokens = 0, 0
        if self.reorder_batch_threshold is not None:
            num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        md.num_decodes = num_decodes  # type: ignore[attr-defined]
        md.num_decode_tokens = num_decode_tokens  # type: ignore[attr-defined]
        # Host copies for the prefill kernel, one launch per sequence: query
        # starts, and sequence lengths (exact for prefill rows).
        md.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu  # type: ignore[attr-defined]
        md.seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound  # type: ignore[attr-defined]
        return md


class RocmAttentionRdna35Impl(TritonAttentionImpl):
    """ROCM_ATTN on gfx1151: the RDNA3.5 HIP kernels where they serve the
    call (decode, and prefills of PREFILL_MIN_M tokens or more),
    TRITON_ATTN's forward for everything else, on the same packed cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._variant: KernelVariant | None = None
        self._built: _Built | None = None
        self._rejected: str | None = None
        # Counters, so a test can assert the kernel really ran. A benchmark
        # that silently falls back measures Triton and reports it as this
        # backend, which is worse than an error.
        self.kernel_calls = 0
        self.fallback_calls = 0
        # Mixed batches served as decodes on the kernel plus the rest on Triton.
        self.split_calls = 0
        # Batches whose prefills ran on the prefill kernel.
        self.prefill_calls = 0
        try:
            from vllm.config import get_current_vllm_config

            max_seqs = get_current_vllm_config().scheduler_config.max_num_seqs
        except Exception:
            max_seqs = 1
        self._max_seqs = max(1, max_seqs)
        self._cap = 0

    def _reject(self, reason: str) -> None:
        """Record why this shape falls back.  Info, not a warning: as gfx1151's
        default backend it falls back in normal operation (prefills, head
        sizes the kernel does not build)."""
        if self._rejected != reason:
            self._rejected = reason
            logger.info_once(
                "ROCM_ATTN (gfx1151) falling back to Triton: %s", reason, scope="local"
            )

    def _served_window(self, kwargs: dict) -> int | None:
        """Features both kernels implement.  Returns the window in keys (0:
        full causal), or None after recording why not."""
        if kwargs["alibi_slopes"] is not None or kwargs["sinks"] is not None:
            self._reject("alibi/sinks unsupported")
            return None
        if kwargs["softcap"] or not kwargs["causal"]:
            self._reject("softcap or non-causal unsupported")
            return None
        # vLLM passes a causal window of w keys as (w - 1, 0).
        window = kwargs["window_size"]
        win = 0
        if window is not None and window[0] >= 0:
            if window[1] != 0:
                self._reject(f"only causal sliding windows, got {window}")
                return None
            win = window[0] + 1
        # Features the kernel does not implement must not reach it silently.
        if kwargs.get("mm_prefix_range") is not None:
            self._reject("multimodal bidirectional prefix unsupported")
            return None
        if kwargs.get("rswa_prefix_lens") is not None:
            self._reject("rswa unsupported")
            return None
        if kwargs.get("chunk_lookback", -1) >= 0:
            self._reject("chunk lookback unsupported")
            return None
        # Not the descale tensors: on the unquantized path k_descale is still a
        # broadcast of a 1.0 scale, so testing it for None never fires.
        if kwargs["kv_quant_mode"] != KVQuantMode.NONE:
            self._reject(f"KV quant mode {kwargs['kv_quant_mode']!r} unsupported")
            return None
        return win

    def _served_tensors(
        self, kv_cache: torch.Tensor, kwargs: dict, head_sizes: tuple[int, ...] | None
    ) -> bool:
        """fp16 or bf16 throughout, GQA, whole 16-key pages, and (decode) a
        built head size; else False after recording why."""
        dtype = kwargs["q"].dtype
        if dtype not in (torch.float16, torch.bfloat16):
            self._reject(f"kernel is fp16 or bf16 only, got {dtype}")
            return False
        if kv_cache.dtype != dtype:
            self._reject(f"KV cache is {kv_cache.dtype}, query is {dtype}")
            return False
        if head_sizes is not None and self.head_size not in head_sizes:
            self._reject(f"head_size {self.head_size} not built")
            return False
        if self.num_heads % self.num_kv_heads:
            self._reject("q heads must divide evenly over kv heads")
            return False
        if kv_cache.shape[2] % 16:
            self._reject(f"block size {kv_cache.shape[2]} is not a multiple of 16")
            return False
        return True

    def _prepare(self, kv_cache: torch.Tensor, **kwargs) -> _Built | None:
        """Decide whether the kernel can serve this call, and build it if so.

        Every condition is checked rather than assumed. The kernel walks the
        paged KV cache with its own address arithmetic instead of reading the
        tensor's strides, so a layout it did not expect would not fault — it
        would read the wrong addresses and return finite, wrong numbers.

        Returns the compiled module and its scratch buffers, or None to fall
        back to Triton.
        """
        win = self._served_window(kwargs)
        if win is None:
            return None

        # Every sequence with the same number of query tokens: decode, or
        # speculative decode.  Batches with prefills go to _serve_prefills or
        # Triton.
        nseq = kwargs["seqused_k"].shape[0]
        max_query_len = kwargs["max_seqlen_q"]
        if kwargs["q"].shape[0] != nseq * max_query_len:
            self._reject("sequences of unequal query length")
            return None
        # A decode kernel: every distinct M is a build of its own, so a
        # prompt served here would compile a variant per prompt length.
        if max_query_len > MAX_M:
            self._reject(
                f"{max_query_len} query tokens per sequence, more than {MAX_M}"
            )
            return None
        if not self._served_tensors(kv_cache, kwargs, SUPPORTED_HEAD_SIZES):
            return None
        dtype = kwargs["q"].dtype

        # Logical KV cache order is (num_blocks, num_kv_heads, block_size, 2*hs).
        q = kwargs["q"]
        block_size = kv_cache.shape[2]
        variant = variant_for(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            max_query_len,
            block_size,
            0 if kv_cache.stride(1) < kv_cache.stride(2) else 1,
            win,
            dtype,
            nseq > 1,
        )
        if variant is None:
            self._reject("no build in _rocm_C for this shape and layout")
            return None
        expected = expected_kv_cache_strides(variant)
        actual = (kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2))
        if actual != expected:
            self._reject(f"KV strides {actual} != {expected} for this layout")
            return None

        if self._variant != variant:
            module = load(variant)
            if module is None:
                self._reject(f"{variant.name} is not in this _rocm_C")
                return None
            key = (variant, q.device)
            if key not in _SCRATCH:
                if variant.batched:
                    per_seq = scratch_bytes(variant)
                    cap = min(self._max_seqs, max(1, _SCRATCH_BUDGET // per_seq))
                    _SCRATCH[key] = (cap, make_scratch(variant, q.device, cap))
                else:
                    _SCRATCH[key] = (1, make_scratch(variant, q.device))
            self._cap, scratch = _SCRATCH[key]
            self._built = (module, scratch)
            self._variant = variant
        if nseq > self._cap:
            self._reject(f"{nseq} sequences, scratch holds {self._cap}")
            return None
        return self._built

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The kernel when it can serve the call, else Triton's forward."""
        if (
            attn_metadata is None
            or self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER)
            or output_scale is not None
            or output_block_scale is not None
        ):
            # Profiling, encoder attention and quantized output: Triton's.
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        n = attn_metadata.num_actual_tokens
        args = {
            "q": query[:n],
            "out": output[:n],
            "seqused_k": attn_metadata.seq_lens,
            "block_table": attn_metadata.block_table,
            "max_seqlen_q": attn_metadata.max_query_len,
            "softmax_scale": self.scale,
            "causal": attn_metadata.causal,
            "window_size": self.sliding_window,
            "alibi_slopes": self.alibi_slopes,
            "sinks": self.sinks,
            "softcap": self.logits_soft_cap,
            "mm_prefix_range": attn_metadata.mm_prefix_range_tensor,
            "rswa_prefix_lens": attn_metadata.rswa_prefix_lens,
            "chunk_lookback": self.chunk_lookback,
            "kv_quant_mode": self._kv_quant_mode,
        }
        if self._serve_prefills(
            layer, query, key, value, kv_cache, attn_metadata, args
        ):
            self.kernel_calls += 1
            self.prefill_calls += 1
            return output
        if self._split_mixed(layer, query, key, value, kv_cache, attn_metadata, args):
            self.kernel_calls += 1
            self.split_calls += 1
            return output
        built = self._prepare(kv_cache, **args)
        if built is None:
            self.fallback_calls += 1
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output
            )
        self.kernel_calls += 1
        self._launch(built, kv_cache, args)
        return output

    def _serve_prefills(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        md: TritonAttentionMetadata,
        kwargs: dict,
    ) -> bool:
        """Serve a batch of decodes followed by prefills of PREFILL_MIN_M
        tokens or more: the decodes on the decode kernel (Triton if it cannot
        take them), each prefill a launch of the prefill kernel.  Returns
        False, having done nothing, when the batch is not like that.

        Never under CUDA-graph capture: the launches follow this step's
        lengths on the host, and prefills run piecewise anyway.
        """
        nd = getattr(md, "num_decodes", 0)
        ndt = getattr(md, "num_decode_tokens", 0)
        nreq = kwargs["seqused_k"].shape[0]
        starts = getattr(md, "query_start_loc_cpu", None)
        if nd >= nreq or starts is None or torch.cuda.is_current_stream_capturing():
            return False
        if kwargs["max_seqlen_q"] < PREFILL_MIN_M:
            return False
        win = self._served_window(kwargs)
        if win is None or not self._served_tensors(kv_cache, kwargs, None):
            return False
        block_size = kv_cache.shape[2]
        kv_row = 2 * self.head_size
        hnd = (block_size * self.num_kv_heads * kv_row, block_size * kv_row, kv_row)
        if (kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2)) != hnd:
            self._reject("prefill: KV cache is not the packed HND layout")
            return False
        dtype = kwargs["q"].dtype
        starts = starts[: nreq + 1].tolist()
        seq_lens = getattr(md, "seq_lens_cpu", None)
        seq_lens = seq_lens[:nreq].tolist() if seq_lens is not None else None
        launches: list[tuple[Any, int, int, int]] = []
        for i in range(nd, nreq):
            m = starts[i + 1] - starts[i]
            if m < PREFILL_MIN_M:
                self._reject(f"prefill of {m} query tokens, fewer than {PREFILL_MIN_M}")
                return False
            variant = prefill_variant_for(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                block_size,
                1,
                win,
                dtype,
                m,
            )
            module = load_prefill(variant) if variant is not None else None
            if module is None:
                self._reject(
                    f"prefill of {m} query tokens: no build in _rocm_C for "
                    f"{self.num_heads}/{self.num_kv_heads}/{self.head_size} "
                    f"window {win} page {block_size} {dtype}"
                )
                return False
            host_len = seq_lens[i] if seq_lens is not None else md.max_seq_len
            launches.append((module, i, m, max(host_len, m)))

        if nd:
            dec = dict(kwargs)
            for k in ("q", "out"):
                dec[k] = kwargs[k][:ndt]
            for k in ("seqused_k", "block_table"):
                dec[k] = kwargs[k][:nd]
            dec["max_seqlen_q"] = ndt // nd
            built = self._prepare(kv_cache, **dec)
            if built is not None:
                self._launch(built, kv_cache, dec)
            else:
                decodes = replace(
                    md,
                    num_actual_tokens=ndt,
                    max_query_len=ndt // nd,
                    query_start_loc=md.query_start_loc[: nd + 1],
                    seq_lens=md.seq_lens[:nd],
                    block_table=md.block_table[:nd],
                )
                super().forward(
                    layer,
                    query[:ndt],
                    key,
                    value,
                    kv_cache,
                    decodes,
                    kwargs["out"][:ndt],
                )

        shape = (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            block_size,
            win,
            dtype,
        )
        scratch_key = (shape, kwargs["q"].device)
        if scratch_key not in _PREFILL_SCRATCH:
            _PREFILL_SCRATCH[scratch_key] = make_prefill_scratch(
                prefill_variants(*shape), kwargs["q"].device
            )
        scratch = _PREFILL_SCRATCH[scratch_key]
        for module, i, m, host_len in launches:
            q0 = starts[i]
            module.prefill_attn(
                kwargs["q"][q0 : q0 + m],
                kv_cache,
                kwargs["block_table"][i],
                kwargs["out"][q0 : q0 + m],
                *scratch,
                kwargs["seqused_k"][i : i + 1],
                host_len,
                kwargs["softmax_scale"],
            )
        return True

    def _split_mixed(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        md: TritonAttentionMetadata,
        kwargs: dict,
    ) -> bool:
        """Serve a batch of decodes followed by prefills in two launches: the
        decodes on the kernel, the prefills on Triton.  Returns False, having
        done nothing, when the batch is not like that or the kernel cannot
        take its decodes.

        Not under CUDA-graph capture: the split is host arithmetic on this
        step's batch, and a captured graph would replay one step's split.
        """
        nd = getattr(md, "num_decodes", 0)
        ndt = getattr(md, "num_decode_tokens", 0)
        nreq = kwargs["seqused_k"].shape[0]
        if not 0 < nd < nreq or torch.cuda.is_current_stream_capturing():
            return False
        # The prefills leave the launch they shared with the decodes, and a
        # short one alone is latency-bound in Triton.  Below D=256 the
        # kernel's decode gain over Triton (1.0-1.1x) does not pay for that.
        if self.head_size < _SPLIT_MIN_HEAD_SIZE:
            return False
        # A few decodes beside a short extend: the extend, alone, costs more
        # than the kernel saves on the decodes (0.79-0.85x measured).
        if nd < _SPLIT_MIN_DECODES and kwargs["q"].shape[0] - ndt < _SPLIT_MIN_PREFILL:
            return False
        # Two kv heads at D=256: Triton's short extend alone costs more than
        # the kernel saves even on 16-32 decodes (0.73-0.93x measured).
        prefill_tokens = kwargs["q"].shape[0] - ndt
        pair = (self.num_kv_heads, self.head_size)
        if pair in _SPLIT_NEEDS_LONG_PREFILL and prefill_tokens < _SPLIT_MIN_PREFILL:
            return False
        # A small window leaves a decode little KV to save on: a few of them
        # do not pay for the prefill's own launch (0.95-0.98x at w512).
        window = kwargs["window_size"]
        if (
            window is not None
            and 0 <= window[0] < _SPLIT_SMALL_WINDOW
            and nd < _SPLIT_MIN_DECODES
        ):
            return False
        dec = dict(kwargs)
        for k in ("q", "out"):
            dec[k] = kwargs[k][:ndt]
        for k in ("seqused_k", "block_table"):
            dec[k] = kwargs[k][:nd]
        dec["max_seqlen_q"] = ndt // nd
        built = self._prepare(kv_cache, **dec)
        if built is None:
            return False
        self._launch(built, kv_cache, dec)
        prefills = replace(
            md,
            num_actual_tokens=md.num_actual_tokens - ndt,
            query_start_loc=md.query_start_loc[nd:] - ndt,
            seq_lens=md.seq_lens[nd:],
            block_table=md.block_table[nd:],
        )
        super().forward(
            layer, query[ndt:], key, value, kv_cache, prefills, kwargs["out"][ndt:]
        )
        return True

    def _launch(self, built: _Built, kv_cache: torch.Tensor, kwargs: dict) -> None:
        module, (acc, softmax_max, softmax_sum, arrivals) = built
        # Called directly rather than through a registered custom op: this path
        # is exercised under CUDA-graph capture, not torch.compile, so the op
        # wrapper would only add indirection inside the region being measured.
        module.decode_attn(
            kwargs["q"],
            kv_cache,
            kwargs["block_table"],
            kwargs["out"],
            acc,
            softmax_max,
            softmax_sum,
            arrivals,
            # The device tensor, not max_seqlen_k: this runs under full
            # CUDA-graph capture, where a host int would be frozen at its
            # capture-time value for every replay.
            kwargs["seqused_k"],
            kwargs["softmax_scale"],
        )
