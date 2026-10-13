# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm's DeepSeek-V4 paged-K reader, sourced from aiter.

The shared reader in ``common/ops/cache_utils.py`` knows one record layout:
fp8 [0,448) | bf16 [448,576), with the UE8M0 scales grouped in a per-block
region placed after the block's token data. ``VLLM_DSV4_AITER_SPARSE_MLA``
writes a second, 640-byte aligned record --
fp8 [0,448) | scales [448,462) | pad | bf16 [512,640) -- because the aiter
decode kernel's scaled MMA wants each token's scales inside its own record.

Teaching the shared kernel both layouts meant editing a file every platform
reads. aiter owns that kernel instead, and this subclass points at it. The
warmup contract is inherited untouched: ``dispatch``, ``get_warmup_keys`` and
``warmup_inputs`` describe the cache's paging, not its record layout, so none
of them need to know which record is in use. Only ``__call__`` differs, by the
four geometry constexprs the aiter kernel takes.
"""

import torch

from aiter.ops.triton.quant.fused_mxfp8_quant import _rec_geometry
from aiter.ops.triton._triton_kernels.quant.fused_mxfp8_quant import (
    _fused_deepseek_v4_dequant_gather_k_cache_kernel,
)

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    kernel_launcher,
)
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    DequantizeAndGatherKCacheKernel as _SharedDequantizeAndGatherKCacheKernel,
)


class DequantizeAndGatherKCacheKernel(_SharedDequantizeAndGatherKCacheKernel):
    """The shared reader with aiter's record-layout-aware kernel."""

    kernel = _fused_deepseek_v4_dequant_gather_k_cache_kernel

    @kernel_launcher
    def __call__(  # type: ignore[override]
        self,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor | None,
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
        *,
        use_fnuz: bool = False,
    ) -> LaunchSpec:
        num_reqs = seq_lens.shape[0]
        rec_bytes, sc_in_rec, rope_in_rec, sc_step = _rec_geometry(k_cache)
        return (num_reqs, self.NUM_WORKERS), dict(
            out_stride0=out.stride(0),
            out_stride1=out.stride(1),
            max_blocks_per_seq=block_table.shape[-1],
            fp8_dim=448,
            bf16_dim=64,
            scale_dim=8,
            quant_block=64,
            cache_block_size=block_size,
            token_data_size=576,
            rec_bytes=rec_bytes,
            sc_in_rec=sc_in_rec,
            rope_in_rec=rope_in_rec,
            sc_step=sc_step,
            block_stride=k_cache.stride(0),
            output_dim=512,
            fp8_max=448.0,
            n_quant_blocks=7,
        )


_DEQUANTIZE_AND_GATHER_K_CACHE_KERNEL = DequantizeAndGatherKCacheKernel()


def dequantize_and_gather_k_cache(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
) -> None:
    """Dequantize and gather a paged DSv4 K cache, on either record layout.

    ``use_fnuz`` MUST match the encoder of the specific cache being read:
    ``False`` for ``compressed_k_cache`` (Triton encoder is OCP everywhere),
    ``current_platform.is_fp8_fnuz()`` for ``swa_k_cache`` (C++ encoder
    writes FNUZ on gfx942 and OCP on gfx950).

    The shared wrapper's cutedsl branch is dropped: it is a CUDA kernel and
    ``has_cutedsl()`` is never true on this platform.
    """
    _DEQUANTIZE_AND_GATHER_K_CACHE_KERNEL(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz=use_fnuz,
    )
