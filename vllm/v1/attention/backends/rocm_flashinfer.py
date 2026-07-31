# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer on ROCm.

Backed by AMD's ROCm port of FlashInfer (the ``amd-flashinfer`` wheel, imported
as ``flashinfer``): https://github.com/AMD-Ecosystem/flashinfer

This is deliberately a separate backend from
``vllm.v1.attention.backends.flashinfer``. The CUDA backend is built around
TRT-LLM/cuDNN/FP4 code paths and ``fast_decode_plan``, none of which exist in
the HIP build, so sharing it would mean threading platform branches through
every method.

Two hard-won constraints shape this file; both are silent-wrongness traps
rather than crashes, so they are enforced rather than documented:

1. The prefill wrapper is pinned to ``backend="aiter"``. On ROCm the ``auto``
   route resolves to the HIP ``fa2`` kernels, and as of flashinfer
   0.5.3+amd.1 the fa2 *prefill* kernel returns numerically wrong results
   (verified against a torch SDPA reference on gfx942 across paged/ragged/
   single entry points, fp16 and bf16, page sizes 1/16/32/64). Decode is
   correct on every route.
2. The AITER route silently *ignores* ALiBi slopes, RoPE scaling, attention
   sinks and fp8 dequant scales rather than erroring. Anything in that list is
   rejected up front, in ``validate_configuration`` or in ``__init__``.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)

# Matches the CUDA backend's default. Not yet tuned for the HIP kernels.
FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

# See module docstring: "auto"/"fa2" prefill is numerically broken on ROCm.
_PREFILL_BACKEND = "aiter"
# Decode is correct on every route; "auto" lets flashinfer pick per shape.
_DECODE_BACKEND = "auto"

# The HIP decode kernel dispatches the GQA group size through a fixed switch
# (DISPATCH_GQA_GROUP_SIZE in flashinfer/include/flashinfer/utils.cuh). Any
# other ratio aborts inside plan() with "Unsupported group_size: N" rather than
# failing selection, so it has to be screened here. Notably this rules out
# 14/2=7 (Qwen2.5-0.5B) and 16.
SUPPORTED_GQA_GROUP_SIZES = (1, 2, 3, 4, 8)


class RocmFlashInferBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # The AITER prefill route requires q_dtype == kv_dtype, so quantized KV
    # caches are out of scope here.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # KV cache writes go through do_kv_cache_update(), not forward().
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHINFER"

    @staticmethod
    def get_impl_cls() -> type["RocmFlashInferImpl"]:
        return RocmFlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["RocmFlashInferMetadataBuilder"]:
        return RocmFlashInferMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Verified against an SDPA reference on gfx942. Page sizes >= 128 are
        # a trtllm-gen feature and do not apply here.
        return [16, 32, 64]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # K and V packed in the content dim, matching the CUDA FlashInfer
        # backend: (B, H, N, 2*hs). The NHD permutation lives in
        # get_kv_cache_stride_order().
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # NHD only. The AITER route rejects HND outright, and this backend
        # cannot run prefill without AITER, so there is no HND branch to offer.
        if include_num_layers_dimension:
            # (num_blocks, num_layers, block_size, num_kv_heads, 2*head_size)
            return (1, 0, 3, 2, 4)
        # (num_blocks, block_size, num_kv_heads, 2*head_size)
        return (0, 2, 1, 3)

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        return "NHD"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        from vllm.platforms.rocm import get_cdna_version

        # amd-flashinfer supports gfx942 (CDNA3) and gfx950 (CDNA4) only.
        # get_cdna_version() reads amd-smi, which is more reliable on ROCm than
        # torch.cuda.get_device_capability().
        return get_cdna_version() >= 3

    @classmethod
    def supports_sink(cls) -> bool:
        # The AITER route accepts a `sinks` kwarg and ignores it.
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        # window_left is plumbed through plan(), but it is not yet verified to
        # be honored on the AITER prefill route. A silently-ignored window only
        # shows up as wrong output at long context, so stay off until proven.
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Head counts are not part of the selector config, so read them off the
        # current config the way FlashInferBackend does for block sizes.
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or vllm_config.model_config is None:
            return None

        num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        if num_kv_heads <= 0:
            return None
        if num_qo_heads % num_kv_heads != 0:
            return (
                f"query heads ({num_qo_heads}) must be a multiple of KV heads "
                f"({num_kv_heads})"
            )
        group_size = num_qo_heads // num_kv_heads
        if group_size not in SUPPORTED_GQA_GROUP_SIZES:
            return (
                f"GQA group size {group_size} ({num_qo_heads} query heads / "
                f"{num_kv_heads} KV heads) is not supported; the HIP decode "
                f"kernel only dispatches {list(SUPPORTED_GQA_GROUP_SIZES)}"
            )
        return None


@dataclass
class RocmFlashInferMetadata(AttentionMetadata):
    num_actual_tokens: int

    # Not read by forward() -- the runner hands slot_mapping to
    # do_kv_cache_update() directly -- but carried like the other backends do,
    # since test harnesses and profiling tools read it off the metadata.
    slot_mapping: torch.Tensor

    # Batch split. Decodes are reordered to the front of the batch.
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # Planned wrappers; None when that half of the batch is empty.
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Flatten the per-request rows of block_table into a ragged index array."""
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )


class RocmFlashInferMetadataBuilder(AttentionMetadataBuilder[RocmFlashInferMetadata]):
    # CUDA graph support is deliberately deferred: get correctness first, then
    # raise this to UNIFORM_SINGLE_TOKEN_DECODE together with the persistent
    # decode-wrapper cache that it requires.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config

        self.num_qo_heads = model_config.get_num_attention_heads(parallel_config)
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_dim = kv_cache_spec.head_size
        self.page_size = kv_cache_spec.block_size

        self.q_data_type = model_config.dtype
        self.kv_data_type = self.q_data_type
        self.cache_dtype = cache_config.cache_dtype

        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages_per_req = cdiv(model_config.max_model_len, self.page_size)

        self.paged_kv_indptr = self._make_buffer(max_num_reqs + 1)
        self.paged_kv_indices = self._make_buffer(max_num_reqs * max_num_pages_per_req)
        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)

        params = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, RocmFlashInferImpl)
        )
        self.global_hyperparameters = params
        self.window_left = params.window_left
        self.logits_soft_cap = params.logits_soft_cap
        self.sm_scale = params.sm_scale

        self._workspace_buffer: torch.Tensor | None = None
        self._prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None
        self._decode_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None

    def _make_buffer(self, size: int) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            size, dtype=torch.int32, device=self.device, pin_memory=PIN_MEMORY
        )

    def _get_workspace_buffer(self) -> torch.Tensor:
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.device,
            )
        return self._workspace_buffer

    def _get_prefill_wrapper(self) -> BatchPrefillWithPagedKVCacheWrapper:
        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                "NHD",
                backend=_PREFILL_BACKEND,
            )
        return self._prefill_wrapper

    def _get_decode_wrapper(self) -> BatchDecodeWithPagedKVCacheWrapper:
        if self._decode_wrapper is None:
            self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                "NHD",
                # use_tensor_cores=True disqualifies the AITER route and is not
                # needed by the HIP decode kernels.
                use_tensor_cores=False,
                backend=_DECODE_BACKEND,
            )
        return self._decode_wrapper

    def _build_paged_kv_indices(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Fill the indptr / indices / last_page_len buffers for the batch."""
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1 : num_reqs + 1],
        )
        self.paged_kv_indptr.copy_to_gpu(num_reqs + 1)

        num_actual_pages = int(self.paged_kv_indptr.np[num_reqs])
        _copy_page_indices_kernel[(num_reqs,)](
            self.paged_kv_indices.gpu[:num_actual_pages],
            block_table_tensor,
            block_table_tensor.stride(0),
            self.paged_kv_indptr.gpu[: num_reqs + 1],
            BLOCK_SIZE=1024,
        )

        # A sequence whose length is an exact multiple of page_size has a full
        # last page, not a zero-length one.
        last_page_len_np = seq_lens_np % self.page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (last_page_len_np == 0) & (seq_lens_np != 0),
            self.page_size,
            last_page_len_np,
        )
        self.paged_kv_last_page_len.copy_to_gpu(num_reqs)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmFlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
            )
        )

        block_table_tensor = common_attn_metadata.block_table_tensor
        seq_lens_np = common_attn_metadata.seq_lens.cpu().numpy()
        num_blocks_np = (seq_lens_np + self.page_size - 1) // self.page_size

        self._build_paged_kv_indices(
            num_blocks_np, seq_lens_np, block_table_tensor, num_reqs
        )

        indptr = self.paged_kv_indptr.gpu[: num_reqs + 1]
        indices = self.paged_kv_indices.gpu
        last_page_len = self.paged_kv_last_page_len.gpu[:num_reqs]

        attn_metadata = RocmFlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
        )

        # Decodes occupy [0, num_decodes); prefills occupy [num_decodes, num_reqs).
        if num_decodes > 0:
            decode_wrapper = self._get_decode_wrapper()
            decode_wrapper.plan(
                indptr[: num_decodes + 1],
                indices,
                last_page_len[:num_decodes],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                sm_scale=self.sm_scale,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_data_type,
            )
            attn_metadata.decode_wrapper = decode_wrapper

        if num_prefills > 0:
            qo_indptr = common_attn_metadata.query_start_loc
            # Re-base both index arrays onto the prefill sub-batch.
            qo_indptr_p = qo_indptr[num_decodes:] - qo_indptr[num_decodes]
            indptr_p = indptr[num_decodes:] - indptr[num_decodes]
            indices_p = indices[int(self.paged_kv_indptr.np[num_decodes]) :]

            prefill_wrapper = self._get_prefill_wrapper()
            prefill_wrapper.plan(
                qo_indptr_p,
                indptr_p,
                indices_p,
                last_page_len[num_decodes:num_reqs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=True,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                sm_scale=self.sm_scale,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_data_type,
            )
            attn_metadata.prefill_wrapper = prefill_wrapper

        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # MultiLevelCascadeAttentionWrapper is absent from the HIP build.
        return False


class RocmFlashInferImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.logits_soft_cap = logits_soft_cap

        # The AITER route accepts these and silently ignores them, which would
        # produce plausible-but-wrong output. Fail loudly instead.
        if alibi_slopes is not None:
            raise NotImplementedError(
                "ROCM_FLASHINFER does not support ALiBi slopes: the AITER "
                "route ignores them silently. Use ROCM_AITER_FA or TRITON_ATTN."
            )
        if sinks is not None:
            raise NotImplementedError(
                "ROCM_FLASHINFER does not support attention sinks: the AITER "
                "route ignores them silently."
            )
        self.sinks = None
        self.alibi_slopes = None

        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            raise NotImplementedError(
                "ROCM_FLASHINFER does not yet support sliding window; "
                "window_left is not verified against the AITER route."
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "ROCM_FLASHINFER only supports decoder-self attention, got "
                f"{attn_type}."
            )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.kv_sharing_target_layer_name is not None:
            # Sharing an earlier layer's cache; that layer already wrote it.
            return
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmFlashInferMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "ROCM_FLASHINFER requires an output buffer"
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "ROCM_FLASHINFER does not support fused output quantization."
            )

        if attn_metadata is None:
            # Profiling run: KV cache is not populated yet.
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens
        # The wrappers take the KV cache as a (k, v) tuple of NHD views.
        kv_cache_tuple = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.decode_wrapper is not None:
            attn_metadata.decode_wrapper.run(
                query[:num_decode_tokens],
                kv_cache_tuple,
                out=output[:num_decode_tokens],
            )

        if attn_metadata.prefill_wrapper is not None:
            attn_metadata.prefill_wrapper.run(
                query[num_decode_tokens:num_actual_tokens],
                kv_cache_tuple,
                out=output[num_decode_tokens:num_actual_tokens],
            )

        return output
