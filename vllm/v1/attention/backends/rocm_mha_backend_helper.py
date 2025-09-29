# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Helper functions for ROCm attention backend selection logic.
Centralized logic for choosing between different attention backends and implementations.
"""

from typing import Optional, Tuple, Callable
from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def get_rocm_mha_backend_selection() -> Tuple[str, Optional[str]]:
    """
    Centralized logic for ROCm attention backend selection.
    
    Returns:
        tuple: (backend_class_path, unified_attention_impl_path)
            - backend_class_path: Full class path for the selected backend
            - unified_attention_impl_path: Path to unified attention implementation (if applicable)
    
    Priority order:
    1. AITER MHA: If VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MHA=1
    2. AITER Unified: If VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0 and VLLM_USE_AITER_UNIFIED_ATTENTION=1
    3. vLLM Unified: Default unified attention implementation
    4. Dynamic: Fallback to dynamic backend
    """
    if not current_platform.is_rocm():
        return None, None
    
    # Check AITER availability
    aiter_available = False
    try:
        import aiter  # noqa: F401
        aiter_available = True
    except Exception:
        aiter_available = False
    
    # Priority 1: AITER MHA if both flags are on and available
    if (envs.VLLM_ROCM_USE_AITER and 
        envs.VLLM_ROCM_USE_AITER_MHA and 
        aiter_available):
        logger.info("ROCm Backend Selection: Using AITER FlashAttention backend")
        return ("vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend", None)
    
    # Priority 2: AITER unified attention for Triton
    if (envs.VLLM_USE_AITER_UNIFIED_ATTENTION and 
        not envs.VLLM_V1_USE_PREFILL_DECODE_ATTENTION and 
        aiter_available):
        logger.info("ROCm Backend Selection: Using Triton backend with AITER unified attention")
        return ("vllm.v1.attention.backends.triton_attn.TritonAttentionBackend", 
                "aiter.ops.triton.unified_attention.unified_attention")
    
    # Priority 3: vLLM unified attention for Triton (default)
    logger.info("ROCm Backend Selection: Using Triton backend with vLLM unified attention")
    return ("vllm.v1.attention.backends.triton_attn.TritonAttentionBackend", 
            "vllm.attention.ops.triton_unified_attention.unified_attention")
    


def get_unified_attention_impl() -> Optional[Callable]:
    """
    Get the appropriate unified attention implementation based on environment variables.
    
    Returns:
        Callable or None: The unified attention function to use, or None for split path
    """
    if not current_platform.is_rocm():
        return None
    
    # Check AITER availability
    aiter_available = False
    try:
        import aiter  # noqa: F401
        aiter_available = True
    except Exception:
        aiter_available = False
    
    # Priority 1: AITER unified attention
    if (envs.VLLM_USE_AITER_UNIFIED_ATTENTION and 
        not envs.VLLM_V1_USE_PREFILL_DECODE_ATTENTION and 
        aiter_available):
        try:
            from aiter.ops.triton.unified_attention import unified_attention
            logger.info("ROCm Unified Attention: Using AITER implementation")
            return unified_attention
        except Exception:
            pass
    
    # Priority 2: vLLM unified attention
    else:
        try:
            from vllm.attention.ops.triton_unified_attention import unified_attention
            logger.info("ROCm Unified Attention: Using vLLM implementation")
            return unified_attention
        except Exception:
            pass
    
    # Default: use split path
    logger.info("ROCm Unified Attention: Using split prefill/decode attention")
    return None


# def should_use_aiter_mha() -> bool:
#     """
#     Check if AITER MHA backend should be used.
    
#     Returns:
#         bool: True if AITER MHA should be used
#     """
#     if not current_platform.is_rocm():
#         return False
    
#     # Check AITER availability
#     aiter_available = False
#     try:
#         import aiter  # noqa: F401
#         aiter_available = True
#     except Exception:
#         aiter_available = False
    
#     return (envs.VLLM_ROCM_USE_AITER and 
#             envs.VLLM_ROCM_USE_AITER_MHA and 
#             aiter_available)
