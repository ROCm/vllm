# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


def _is_flashmla_available() -> tuple[bool, str | None]:
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )

    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


if _is_flashmla_available()[0]:
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
        FlashMLASchedMeta,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
else:
    # ------------------------------------------------------------------ #
    # PyTorch reference implementations for non-CUDA platforms (e.g. ROCm)
    # Based on SGLang's reference: sglang/srt/flashmla_tests/ref.py
    # ------------------------------------------------------------------ #
    logger.info("FlashMLA CUDA kernels unavailable — using PyTorch reference "
                "implementations for MLA attention (functional but slower).")

    class FlashMLASchedMeta:  # type: ignore[no-redef]
        """Placeholder scheduling metadata (ignored by the reference path)."""
        def __init__(self):
            self.tile_scheduler_metadata = None
            self.num_splits = None
            self.have_initialized = False

    def get_mla_metadata(*_args, **_kwargs):  # type: ignore[assignment]
        return FlashMLASchedMeta(), None

    # ---- helpers ---------------------------------------------------- #

    # DeepSeek V4 (MODEL1) FP8 cache layout constants
    _D = 512       # total KV head dim
    _D_NOPE = 448  # FP8 NoPE portion
    _D_ROPE = 64   # bf16 RoPE portion
    _TILE = 64     # quantization block (tile) size
    _NUM_TILES = 7  # _D_NOPE // _TILE

    def _dequant_slot(
        cache_2d: torch.Tensor,
        slot_idx: int,
        block_size: int,
        fp8_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize a single token from the paged FP8 KV cache.

        cache_2d: (num_blocks, block_bytes) uint8 — flat 2D view of cache.
        The C kernel writes with block-structured layout:
          [0,            bs*576):  token data (each 576B = 448 fp8 + 128 bf16)
          [bs*576, bs*576+bs*8):  UE8M0 scales (each 8B = 7 real + 1 pad)
        """
        token_data_size = _D_NOPE + _D_ROPE * 2  # 576
        blk_idx = slot_idx // block_size
        pos = slot_idx % block_size

        block_row = cache_2d[blk_idx]  # (block_bytes,) — raw bytes for this block
        token_base = pos * token_data_size
        scale_base = block_size * token_data_size + pos * 8

        # NoPE: dequantize FP8 with UE8M0 scales
        fp8_raw = block_row[token_base:token_base + _D_NOPE]
        fp8_vals = fp8_raw.view(fp8_dtype).to(torch.float32)

        scale_raw = block_row[scale_base:scale_base + _NUM_TILES]
        # UE8M0 → float32: place exponent bits in IEEE754 exponent field
        scales_f32 = (scale_raw.to(torch.int32) << 23).view(torch.float32)
        # Expand: each scale covers _TILE=64 elements
        scales_expanded = scales_f32.repeat_interleave(_TILE)

        nope_dequant = (fp8_vals * scales_expanded).to(torch.bfloat16)

        # RoPE: direct bf16 copy
        rope_base = token_base + _D_NOPE
        rope_vals = block_row[rope_base:rope_base + _D_ROPE * 2].view(
            torch.bfloat16
        )

        return torch.cat([nope_dequant, rope_vals])  # [_D]

    def _gather_and_dequant_kv(
        k_cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather and dequantize sparse KV tokens from paged FP8 cache.

        Args:
            k_cache: (num_blocks, block_size, 1, head_bytes) uint8
            indices: (B, 1, topk) int32 — global slot IDs; -1 = invalid
            lengths: (B,) int32 or None — valid count per batch element

        Returns:
            gathered_kv: (B, 1, topk, _D) bf16
            invalid_mask: (B, topk) bool
        """
        block_size = k_cache.shape[1]
        fp8_dtype = current_platform.fp8_dtype()
        B, _, topk = indices.shape

        # Create a 2D view matching the C kernel's view:
        #   swa_kv_cache_2d = swa_kv_cache.view(shape[0], -1)
        # This gives (num_blocks, total_bytes_per_block) with correct
        # byte addressing that matches the block-structured layout.
        cache_2d = k_cache.view(k_cache.shape[0], -1)

        kv = torch.zeros(B, 1, topk, _D, dtype=torch.bfloat16,
                         device=k_cache.device)

        # Build invalid mask
        invalid = indices.squeeze(1) < 0  # (B, topk)
        if lengths is not None:
            topk_range = torch.arange(topk, device=lengths.device).view(1, topk)
            invalid = invalid | (topk_range >= lengths.unsqueeze(1))

        for b in range(B):
            for t in range(topk):
                if invalid[b, t]:
                    continue
                slot = int(indices[b, 0, t].item())
                kv[b, 0, t] = _dequant_slot(
                    cache_2d, slot, block_size, fp8_dtype
                )

        return kv, invalid

    # ---- flash_mla_with_kvcache (decode) reference ------------------- #

    def flash_mla_with_kvcache(  # type: ignore[assignment]
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table=None,
        head_dim_v: int = 512,
        tile_scheduler_metadata=None,
        cache_seqlens=None,
        is_fp8_kvcache: bool = False,
        indices=None,
        attn_sink=None,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        topk_length=None,
        extra_topk_length=None,
        softmax_scale=None,
        causal: bool = False,
        num_splits=None,
        out=None,
    ):
        """PyTorch reference for sparse FP8 MLA decode attention.

        Based on SGLang's ref_sparse_attn_decode.
        """
        B, S_q, H_q, D_qk = q.shape
        D_v = head_dim_v
        if softmax_scale is None:
            softmax_scale = D_qk ** -0.5

        assert indices is not None

        # 1. Gather & dequantize SWA KV (sparse — only indexed tokens)
        swa_kv, swa_invalid = _gather_and_dequant_kv(
            k_cache, indices, topk_length
        )
        # swa_kv: (B, 1, topk, _D) bf16

        # 2. Optionally gather extra (compressed) KV
        if extra_k_cache is not None and extra_indices_in_kvcache is not None:
            extra_kv, extra_invalid = _gather_and_dequant_kv(
                extra_k_cache, extra_indices_in_kvcache, extra_topk_length
            )
            all_kv = torch.cat([swa_kv, extra_kv], dim=2)
            all_invalid = torch.cat([swa_invalid, extra_invalid], dim=1)
        else:
            all_kv = swa_kv
            all_invalid = swa_invalid

        # Q and K should both be _D=512 dims (head_dim).
        # If D_qk > _D somehow, pad KV to match.
        if D_qk > _D:
            all_kv = torch.nn.functional.pad(all_kv, (0, D_qk - _D))

        # 4. Compute attention
        # q: (B, S_q, H_q, D_qk), all_kv: (B, S_q, total_topk, D_qk)
        q_f = q.float().view(B * S_q, H_q, D_qk)
        kv_f = all_kv.float().view(B * S_q, -1, D_qk)
        total_topk = kv_f.shape[1]

        # scores: (B*S_q, H_q, total_topk)
        attn_scores = torch.bmm(q_f, kv_f.transpose(1, 2)) * softmax_scale

        # Mask invalid
        inv_mask = all_invalid.view(B * S_q, 1, total_topk).expand_as(attn_scores)
        attn_scores[inv_mask] = float("-inf")

        # LSE and softmax
        lse = torch.logsumexp(attn_scores, dim=-1)  # (B*S_q, H_q)
        attn_weights = torch.exp(attn_scores - lse.unsqueeze(-1))

        # Weighted sum over V (first D_v dims)
        v_f = kv_f[..., :D_v]
        output = torch.bmm(attn_weights, v_f)  # (B*S_q, H_q, D_v)
        output = output.view(B, S_q, H_q, D_v)
        lse = lse.view(B, S_q, H_q)

        # 5. Apply attn_sink: output *= 1/(1+exp(attn_sink - lse))
        if attn_sink is not None:
            sink_ratio = 1.0 / (
                1.0 + torch.exp(attn_sink.view(1, 1, H_q) - lse)
            )
            output = output * sink_ratio.unsqueeze(-1)

        # Handle lonely q (no valid k) → zero output
        lonely = lse == float("-inf")
        output[lonely.unsqueeze(-1).expand_as(output)] = 0.0
        lse[lonely] = float("+inf")

        output = output.to(torch.bfloat16)
        lse = lse.transpose(1, 2)  # → (B, H_q, S_q) to match FlashMLA convention

        if out is not None:
            out.copy_(output)

        return output, lse

    # ---- flash_mla_sparse_fwd (prefill) reference -------------------- #

    def flash_mla_sparse_fwd(  # type: ignore[assignment]
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        d_v: int = 512,
        attn_sink=None,
        topk_length=None,
        out=None,
    ):
        """PyTorch reference for sparse MLA prefill attention.

        Based on SGLang's ref_sparse_attn_fwd.
        """
        S_q, H_q, D_qk = q.shape
        S_kv = kv.shape[0]
        topk = indices.shape[2]

        # indices: (S_q, 1, topk) → squeeze to (S_q, topk)
        idx = indices.squeeze(1).clone()

        # Mask out invalid indices based on topk_length
        if topk_length is not None:
            topk_range = torch.arange(topk, device=idx.device).unsqueeze(0)
            idx[topk_range >= topk_length.unsqueeze(1)] = -1

        invalid_mask = (idx < 0) | (idx >= S_kv)  # (S_q, topk)
        idx[invalid_mask] = 0  # safe index for gather

        # Gather KV
        kv_2d = kv.squeeze(1)  # (S_kv, D_qk)
        gathered_kv = kv_2d.index_select(0, idx.reshape(-1)).reshape(
            S_q, topk, D_qk
        ).float()

        q_f = q.float()

        # Attention scores: (S_q, H_q, topk)
        P = torch.bmm(q_f, gathered_kv.transpose(1, 2)) * sm_scale
        P[invalid_mask.unsqueeze(1).expand_as(P)] = float("-inf")

        orig_lse = torch.logsumexp(P, dim=-1)  # (S_q, H_q)
        max_logits = P.max(dim=-1).values  # (S_q, H_q)

        # Merge LSE with attn_sink for output scaling
        if attn_sink is not None:
            lse_for_o = torch.logsumexp(
                torch.stack([
                    orig_lse,
                    attn_sink.unsqueeze(0).expand(S_q, H_q),
                ], dim=0),
                dim=0,
            )
        else:
            lse_for_o = orig_lse.clone()

        # Make inf LSE → +inf so exp(-inf - (+inf)) = 0
        lse_for_o[lse_for_o == float("-inf")] = float("+inf")

        s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
        output = torch.bmm(s_for_o, gathered_kv[..., :d_v])  # (S_q, H_q, d_v)

        # Handle lonely q
        lonely = orig_lse == float("-inf")
        orig_lse[lonely] = float("+inf")

        output = output.to(torch.bfloat16)
        if out is not None:
            out.copy_(output)

        return output, max_logits, orig_lse

    # varlen functions are not used by V4 — keep stubs
    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse
