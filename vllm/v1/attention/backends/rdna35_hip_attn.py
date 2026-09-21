# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA3.5 (gfx1151) HIP decode attention.

A hand-written HIP kernel for the decode and speculative-decode regime, which
on Strix Halo reaches a higher fraction of the memory roofline than the Triton
unified kernel.

It deliberately subclasses the Triton backend rather than standing alone: the
KV cache shape, the stride order and the metadata builder are inherited, so the
two are identical by construction rather than by maintenance. Only the kernel
launch differs, and anything the kernel does not cover falls back to Triton.
"""

from typing import Any, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.attention.ops.rdna35_hip_decode import (
    KernelVariant,
    expected_kv_cache_strides,
    load,
    make_scratch,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)

# The kernel's wave split assumes 8 fp16 per lane, and its GQA indexing assumes
# the q heads divide evenly over the kv heads.
_SUPPORTED_HEAD_SIZES = (256,)

# The JIT-compiled module plus the scratch buffers sized for it.  The module is
# a pybind extension built at runtime, so it has no static type.
_Built = tuple[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class Rdna35HipAttentionBackend(TritonAttentionBackend):
    # bfloat16 stays listed so a bf16 model can still select this backend and
    # be served by the Triton fallback; the kernel itself takes fp16 only,
    # because its inner product is __builtin_amdgcn_fdot2.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "RDNA35_HIP_ATTN"

    @staticmethod
    def get_impl_cls() -> type["Rdna35HipAttentionImpl"]:
        return Rdna35HipAttentionImpl

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False


class Rdna35HipAttentionImpl(TritonAttentionImpl):
    """Triton's impl with the kernel launch swapped when the shape fits."""

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

    def _reject(self, reason: str) -> None:
        """Record why this shape falls back.

        Warning rather than info: callers select this backend to exercise the
        HIP kernel, so falling back is something they need to see.
        """
        if self._rejected != reason:
            self._rejected = reason
            logger.warning_once(
                "RDNA35_HIP_ATTN falling back to Triton: %s", reason, scope="local"
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
        window = kwargs["window_size"]
        if window is not None and window[0] >= 0:
            self._reject("sliding window unsupported")
            return None
        # Not the descale tensors: on the unquantized path k_descale is still a
        # broadcast of a 1.0 scale, so testing it for None never fires.
        if kwargs["kv_quant_mode"] != KVQuantMode.NONE:
            self._reject(f"KV quant mode {kwargs['kv_quant_mode']!r} unsupported")
            return None

        seqused_k = kwargs["seqused_k"]
        if seqused_k.shape[0] != 1:
            self._reject(f"kernel handles one sequence, got {seqused_k.shape[0]}")
            return None
        if kwargs["q"].dtype is not torch.float16:
            self._reject(f"kernel is fp16 only, got {kwargs['q'].dtype}")
            return None
        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            self._reject(f"head_size {self.head_size} not built")
            return None
        if self.num_heads % self.num_kv_heads:
            self._reject("q heads must divide evenly over kv heads")
            return None

        # Logical KV cache order is (num_blocks, num_kv_heads, block_size, 2*hs).
        q = kwargs["q"]
        block_size = kv_cache.shape[2]
        variant = KernelVariant(
            head_size=self.head_size,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_m=q.shape[0],
            block_size=block_size,
            layout=0 if kv_cache.stride(1) < kv_cache.stride(2) else 1,
        )
        expected = expected_kv_cache_strides(variant)
        actual = (kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2))
        if actual != expected:
            self._reject(f"KV strides {actual} != {expected} for this layout")
            return None

        if self._variant != variant:
            self._built = (load(variant), make_scratch(variant, q.device))
            self._variant = variant
        return self._built

    def _run_attention(self, *, kv_cache: torch.Tensor, **kwargs) -> None:
        built = self._prepare(kv_cache, **kwargs)
        if built is None:
            self.fallback_calls += 1
            super()._run_attention(kv_cache=kv_cache, **kwargs)
            return
        self.kernel_calls += 1

        module, (acc, softmax_max, softmax_sum) = built
        # Called directly rather than through a registered custom op: this path
        # is exercised under CUDA-graph capture, not torch.compile, so the op
        # wrapper would only add indirection inside the region being measured.
        module.decode_attn(
            kwargs["q"],
            kv_cache,
            kwargs["block_table"][0],
            kwargs["out"],
            acc,
            softmax_max,
            softmax_sum,
            # max_seqlen_k, not seqused_k[0]: reading the tensor would be a
            # device-to-host copy, which invalidates a CUDA-graph capture. With
            # the single sequence _prepare insists on, the two are equal.
            kwargs["max_seqlen_k"],
            kwargs["softmax_scale"],
        )
