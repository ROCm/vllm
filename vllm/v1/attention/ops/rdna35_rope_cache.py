# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode Q/K norm + RoPE + K/V cache write for RDNA3.5 (gfx1151).

One launch replaces what a decode step runs between the qkv projection and
attention (ROCM_ATTN on gfx1151): the per-head RMSNorm of Q and K (Qwen3,
Gemma 3/4, Qwen3.5), NeoX RoPE (full or partial), Gemma 4's weightless V norm,
and the write of K and V into the packed cache ``(blocks, Hkv, block, 2*D)``.
The kernel is in ``csrc/rocm/rdna35_rope_cache.cu``.

Decode only: ``T = nseq * M`` tokens with ``M <= 8``.  The kernel itself works
per token; callers route only decode batches to it.
"""

import torch


def rdna35_rope_cache_available() -> bool:
    """Whether the op is in this build (it is built for gfx1151 only)."""
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "rdna35_rope_cache"
    )


def rdna35_rope_cache(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    kv_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    *,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    v_norm: bool = False,
    eps: float = 1e-6,
    pos_offset: int = 0,
    q_scale_beta: float = 0.0,
    q_scale_orig_max: int = 0,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
) -> None:
    """Normalise and rotate Q and K, and write K and V into the paged cache.

    Args:
        positions: int64 ``[T]``, or ``[3, T]`` for mrope/imrope.  Only row
            0 is read, which is right only when the three rows are equal
            (text tokens, so every decode token); nothing checks it, so do not
            pass multimodal prefill positions.  The fusion passes never do:
            MRotaryEmbedding does not go through the rotary op they match.
        q: ``[T, Hq, D]`` with a unit last stride; rotated in place unless
            ``q_out`` is given.  The head stride is free, so Qwen3.5's
            ``[q | gate]`` heads are passed as a view of their q halves.
        k: ``[T, Hkv, D]``, or None for a layer that only rotates Q (Gemma 4
            KV-shared layers); rotated in place unless ``k_out`` is given.
        v: ``[T, Hkv, D]``; only written to the cache.
        kv_cache: ``(blocks, Hkv, block, 2*D)`` in logical order, any strides
            but a unit last one, or None to skip the write.
        slot_mapping: int64 ``[<= T]``; negative slots are not written.
        cos_sin_cache: ``[max_pos, rot]``, cos then sin, ``rot <= D``; None for
            a layer without RoPE.
        q_weight: per-head RMSNorm weight of Q, in the activation dtype
            (``RMSNorm``) or fp32 (``GemmaRMSNorm``'s ``1 + w``); None for no
            Q/K norm.
        k_weight: the same for K.
        v_norm: weightless RMSNorm of V before the write (Gemma 4).
        eps: RMSNorm epsilon.
        pos_offset: added to positions to index the cache (Phi long RoPE).
        q_scale_beta: Mistral's ``llama_4_scaling`` beta, 0 to disable.
        q_scale_orig_max: its original max position embeddings.
        q_out: optional ``[T, Hq, D]`` destination for Q.
        k_out: optional ``[T, Hkv, D]`` destination for K.
        v_out: optional ``[T, Hkv, D]`` destination for V as written to the
            cache (normalised with ``v_norm``).
    """
    torch.ops._rocm_C.rdna35_rope_cache(
        positions,
        q,
        k,
        v,
        cos_sin_cache,
        q_weight,
        k_weight,
        v_norm,
        eps,
        pos_offset,
        q_scale_beta,
        q_scale_orig_max,
        kv_cache,
        slot_mapping,
        q_out,
        k_out,
        v_out,
    )


if rdna35_rope_cache_available():
    from torch.library import register_fake

    @register_fake("_rocm_C::rdna35_rope_cache")
    def _rdna35_rope_cache_fake(
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        cos_sin_cache: torch.Tensor | None,
        q_weight: torch.Tensor | None,
        k_weight: torch.Tensor | None,
        v_norm: bool,
        eps: float,
        pos_offset: int,
        q_scale_beta: float,
        q_scale_orig_max: int,
        kv_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        q_out: torch.Tensor | None,
        k_out: torch.Tensor | None,
        v_out: torch.Tensor | None = None,
    ) -> None:
        return
