# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill.

On gfx1151 (Strix Halo) the backend serves decode and speculative decode with
the RDNA3.5 HIP kernel (csrc/rocm/rdna35_decode_attn.cu) on Triton's packed
HND KV cache, and everything else with TRITON_ATTN's kernels on that cache.
"""

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.ir import ops as ir_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
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
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
    has_native_kv_cache_layout,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.ops.rdna35_hip_decode import (
    MAX_M,
    SUPPORTED_HEAD_SIZES,
    KernelVariant,
    VariantBuildError,
    expected_kv_cache_strides,
    load,
    make_scratch,
    scratch_bytes,
    variant_for,
)
from vllm.v1.attention.ops.rdna35_rope_cache import (
    rdna35_rope_cache,
    rdna35_rope_cache_available,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode
from vllm.v1.utils import create_attention_profiler_scope

logger = init_logger(__name__)


def _on_rdna35() -> bool:
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
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
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
        if cls is RocmAttentionBackend and _on_rdna35():
            sizes.append(512)
        return sizes

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

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
        if _on_rdna35():
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

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if _on_rdna35():
            return TritonAttentionBackend.get_kv_cache_shape(
                num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
            )
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def get_kv_cache_stride_order(
        cls,
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if cls is RocmAttentionBackend and _on_rdna35():
            return TritonAttentionBackend.get_kv_cache_stride_order(
                include_num_layers_dimension
            )
        raise NotImplementedError

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        # Keys of one head contiguous in a page.  The decode kernel is tuned
        # on it, and the Triton paths it falls back to run faster on it too:
        # 3-5 % prefill, 13-27 % batched decode.
        if cls is RocmAttentionBackend and _on_rdna35():
            return "HND"
        return None

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        if _on_rdna35():
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

        with create_attention_profiler_scope(
            backend_name="ROCM_ATTN",
            batch_size=seq_lens.shape[0],
            max_query_len=max_query_len,
            max_seq_len=max_query_len,  # For encoder, Q and S are the same
            num_heads=self.num_heads,
            head_size=self.head_size,
            num_kv_heads=self.num_kv_heads,
            dtype=query.dtype,
            is_causal=False,
        ):
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
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
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

        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size
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
        with create_attention_profiler_scope(
            backend_name="ROCM_ATTN",
            batch_size=seqused_k.shape[0],
            max_query_len=max_seqlen_q,
            max_seq_len=max_seqlen_k,
            num_heads=self.num_heads,
            head_size=self.head_size,
            num_kv_heads=self.num_kv_heads,
            dtype=query.dtype,
            is_causal=attn_metadata.causal,
        ):
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
                sliding_window=self.sliding_window[0],
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
            kv_cache, self.num_kv_heads, self.head_size
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
            kv_cache,
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
# Mixed batches are split only from this head size up.
_SPLIT_MIN_HEAD_SIZE = 256
_SPLIT_MIN_DECODES = 16
_SPLIT_MIN_PREFILL = 256
_SPLIT_NEEDS_LONG_PREFILL = {(2, 256)}
_SPLIT_SMALL_WINDOW = 512


class RocmAttentionRdna35MetadataBuilder(TritonAttentionMetadataBuilder):
    """Triton's metadata, with the batch ordered decodes first and the
    number of leading uniform decodes counted, so that a batch mixing
    prefills and decodes can send its decodes to the kernel."""

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
        return md


class RocmAttentionRdna35Impl(TritonAttentionImpl):
    """ROCM_ATTN on gfx1151: the RDNA3.5 HIP kernel where it serves the call,
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

    def _prepare(self, kv_cache: torch.Tensor, **kwargs) -> _Built | None:
        """Decide whether the kernel can serve this call, and build it if so.

        Every condition is checked rather than assumed. The kernel walks the
        paged KV cache with its own address arithmetic instead of reading the
        tensor's strides, so a layout it did not expect would not fault — it
        would read the wrong addresses and return finite, wrong numbers.

        Returns the compiled module and its scratch buffers, or None to fall
        back to Triton.
        """
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

        # Every sequence with the same number of query tokens: decode, or
        # speculative decode.  Mixed batches (prefill in them) go to Triton.
        nseq = kwargs["seqused_k"].shape[0]
        max_m = kwargs["max_seqlen_q"]
        if kwargs["q"].shape[0] != nseq * max_m:
            self._reject("sequences of unequal query length")
            return None
        # A decode kernel: every distinct M is a build of its own, so a
        # prompt served here would compile a variant per prompt length.
        if max_m > MAX_M:
            self._reject(f"{max_m} query tokens per sequence, more than {MAX_M}")
            return None
        dtype = kwargs["q"].dtype
        if dtype not in (torch.float16, torch.bfloat16):
            self._reject(f"kernel is fp16 or bf16 only, got {dtype}")
            return None
        if kv_cache.dtype != dtype:
            self._reject(f"KV cache is {kv_cache.dtype}, query is {dtype}")
            return None
        if self.head_size not in SUPPORTED_HEAD_SIZES:
            self._reject(f"head_size {self.head_size} not built")
            return None
        if self.num_heads % self.num_kv_heads:
            self._reject("q heads must divide evenly over kv heads")
            return None
        if kv_cache.shape[2] % 16:
            self._reject(f"block size {kv_cache.shape[2]} is not a multiple of 16")
            return None

        # Logical KV cache order is (num_blocks, num_kv_heads, block_size, 2*hs).
        q = kwargs["q"]
        block_size = kv_cache.shape[2]
        variant = variant_for(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            max_m,
            block_size,
            0 if kv_cache.stride(1) < kv_cache.stride(2) else 1,
            win,
            dtype,
            nseq > 1,
        )
        expected = expected_kv_cache_strides(variant)
        actual = (kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2))
        if actual != expected:
            self._reject(f"KV strides {actual} != {expected} for this layout")
            return None

        if self._variant != variant:
            try:
                module = load(variant)
            except VariantBuildError as exc:
                self._reject(str(exc).splitlines()[0])
                return None
            key = (variant, q.device)
            if key not in _SCRATCH:
                if variant.batch:
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

    # The fusion passes (fuse_rope_kvcache, fuse_qk_norm_rope_kvcache) hand
    # RoPE, the q/k RMSNorm and the cache write to rdna35_rope_cache, one
    # launch instead of Inductor's norm/RoPE kernels plus the Triton writer.
    # Head sizes the kernel is built for, read by the qk-norm pass.
    fused_qk_norm_rope_kvcache_head_sizes: ClassVar[tuple[int, ...]] = (
        64,
        96,
        128,
        256,
        512,
    )
    # The kernel applies Gemma 4's weightless V norm before the write, and
    # writes the normalised V to the v_out the pass hands attention.
    fused_qk_norm_rope_kvcache_v_norm: ClassVar[bool] = True
    # The decode kernel reads a contiguous query: RoPE writes it to a buffer of
    # its own rather than back into the strided qkv view.
    fused_rope_kvcache_q_out: ClassVar[bool] = True

    def _fused_rope_cache_supported(self) -> bool:
        return (
            rdna35_rope_cache_available()
            and self.attn_type == AttentionType.DECODER
            and self._kv_quant_mode == KVQuantMode.NONE
            and self.head_size in self.fused_qk_norm_rope_kvcache_head_sizes
        )

    def fused_rope_kvcache_supported(self) -> bool:
        return self._fused_rope_cache_supported()

    def fused_qk_norm_rope_kvcache_supported(self) -> bool:
        return self._fused_rope_cache_supported()

    @staticmethod
    def _fused_rejection(
        x: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_weight: torch.Tensor | None = None,
        k_weight: torch.Tensor | None = None,
    ) -> str | None:
        """Why rdna35_rope_cache cannot take this call, or None.  It runs NeoX
        RoPE and reads one element type for activations, cache and cos/sin,
        with norm weights in that type (RMSNorm) or fp32 (GemmaRMSNorm's
        1 + w), both the same."""
        if not is_neox:
            return "GPT-J (non-NeoX) RoPE"
        if kv_cache.dtype != x.dtype or cos_sin_cache.dtype != x.dtype:
            return (
                f"a {kv_cache.dtype} cache or {cos_sin_cache.dtype} cos/sin "
                f"with {x.dtype} activations"
            )
        if (
            q_weight is not None
            and k_weight is not None
            and (
                q_weight.dtype not in (x.dtype, torch.float32)
                or k_weight.dtype != q_weight.dtype
            )
        ):
            return f"{q_weight.dtype}/{k_weight.dtype} norm weights"
        return None

    @staticmethod
    def _warn_unfused(reason: str) -> None:
        logger.warning_once(
            "ROCM_ATTN (gfx1151): rdna35_rope_cache does not take %s; running the "
            "unfused RoPE and KV cache write.",
            reason,
        )

    def do_rope_and_kv_cache_update(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
        q_out: torch.Tensor | None = None,
        k_out: torch.Tensor | None = None,
    ) -> None:
        t, d = query.shape[0], self.head_size
        q, k, v = query.view(t, -1, d), key.view(t, -1, d), value.view(t, -1, d)
        if q_out is not None:
            q_out = q_out.view(t, -1, d)
        if k_out is not None:
            k_out = k_out.view(t, -1, d)
        reason = self._fused_rejection(query, is_neox, kv_cache, cos_sin_cache)
        if reason is not None:
            self._warn_unfused(reason)
            ops.rotary_embedding(positions, query, key, d, cos_sin_cache, is_neox)
            self.do_kv_cache_update(layer, key, value, kv_cache, layer_slot_mapping)
            if q_out is not None:
                q_out.copy_(q)
            if k_out is not None:
                k_out.copy_(k)
            return
        rdna35_rope_cache(
            positions,
            q,
            k,
            v,
            kv_cache,
            layer_slot_mapping,
            cos_sin_cache,
            q_out=q_out,
            k_out=k_out,
        )

    def do_qk_norm_rope_kvcache_update(
        self,
        layer: torch.nn.Module,
        qkv: torch.Tensor,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
        v_norm: bool = False,
        v_out: torch.Tensor | None = None,
    ) -> None:
        t, d = qkv.shape[0], self.head_size
        hq, hkv = self.num_heads, self.num_kv_heads
        q, k, v = (
            x.view(t, -1, d) for x in qkv.split([hq * d, hkv * d, hkv * d], dim=-1)
        )
        q_out, k_out = q_out.view(t, hq, d), k_out.view(t, hkv, d)
        reason = self._fused_rejection(
            qkv, is_neox, kv_cache, cos_sin_cache, q_weight, k_weight
        )
        if reason is not None:
            self._warn_unfused(reason)
            # The unfused ops, in the order the model applies them.
            q_out.copy_(ir_ops.rms_norm(q, q_weight, rms_norm_eps))
            k_out.copy_(ir_ops.rms_norm(k, k_weight, rms_norm_eps))
            ops.rotary_embedding(positions, q_out, k_out, d, cos_sin_cache, is_neox)
            if v_norm:
                v = ir_ops.rms_norm(v, None, rms_norm_eps)
            self.do_kv_cache_update(layer, k_out, v, kv_cache, layer_slot_mapping)
            if v_out is not None:
                v_out.copy_(v)
            return
        rdna35_rope_cache(
            positions,
            q,
            k,
            v,
            kv_cache,
            layer_slot_mapping,
            cos_sin_cache,
            q_weight=q_weight,
            k_weight=k_weight,
            v_norm=v_norm,
            eps=rms_norm_eps,
            q_out=q_out,
            k_out=k_out,
            v_out=None if v_out is None else v_out.view(t, hkv, d),
        )

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
        # The kernel reads a contiguous query.  The rotary custom op rotates
        # the qkv view in place, so without the RoPE+cache fusion (or past its
        # token range) the query arrives strided.
        q = kwargs["q"]
        if not q.is_contiguous():
            q = q.contiguous()
        # Called directly rather than through a registered custom op: this path
        # is exercised under CUDA-graph capture, not torch.compile, so the op
        # wrapper would only add indirection inside the region being measured.
        module.decode_attn(
            q,
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
