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

from typing import Any, ClassVar, TypedDict

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.attention.ops.rdna35_hip_decode import (
    KernelVariant,
    VariantBuildError,
    expected_kv_cache_strides,
    load,
    make_scratch,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)

# The kernel's V and K slices are one b64 or b128 per lane, which the head
# dims below allow; its GQA packing assumes the q heads divide evenly over the
# kv heads.
_SUPPORTED_HEAD_SIZES = (64, 128, 256, 512)

# Workgroups the long-context split aims for.  Fewer, longer streams keep
# LPDDR5X closer to its peak than many short ones: a tile-structured stream
# with this kernel's access order measured 97.9 % of peak at 16 workgroups,
# 96.4 % at 40 and 90-94 % at 64 and up.
_TARGET_WORKGROUPS = 16


def _segments_for(num_kv_heads: int, rg: int) -> int:
    """Most KV segments per (kv head, row group), for _TARGET_WORKGROUPS."""
    return max(1, _TARGET_WORKGROUPS // (num_kv_heads * rg))


def _rows_split(gqa: int, max_m: int, head_size: int) -> tuple[int, int]:
    """Row groups, and a d split if one is needed, for one kv head's rows.

    A wave's accumulator is its rows times the d it owns, and it has to stay
    near 64 VGPRs: 16 rows at 128 d, 32 at 64.  Row groups are the cheaper way
    down -- they only re-read the kv head's KV from L2 -- but they split whole
    q heads, so a GQA with no fitting divisor (7, 5) halves each wave's d
    instead, which measured 63-73 % of roof against 48-69 % for rg = GQA.  At
    D=512 it is LDS rather than VGPRs: Q and the K tile leave room for one row
    tile.

    Returns:
        `(rg, dspl)`, dspl 0 meaning the kernel's own rule.
    """
    rows = gqa * max_m
    cap = 32 if head_size == 64 else 16
    for rg in range(1, gqa + 1):
        if gqa % rg == 0 and rows // rg <= cap:
            break
    if rows // rg <= cap and (rg == 1 or rows // rg >= 8):
        return rg, 0
    if head_size == 128 and rows <= 2 * cap:
        return 1, 2
    return rg, 0


# Best knobs measured per configuration, keyed on (Hq, Hkv, D, M).
#
# The heuristics in _knobs_for are rules fitted to the whole shape table; this
# is the exceptions list, and it wins where both apply.  Rows carry only the
# knobs actually measured -- a partial row is normal, and a configuration
# absent here behaves exactly as the heuristics say, so landing a row can only
# affect the configuration it names.
#
# Provenance is OPTIMIZATIONS.md (per knob) and golden/ (per shape).  Do not
# add a row without a measurement behind it.
class _Knobs(TypedDict, total=False):
    """Launch knobs a configuration may pin.  Total=False: a row sets only what
    was measured."""

    nseg: int
    rg: int
    minb: int
    nw: int
    dspl: int


# Every shipped configuration, measured: coordinate descent over (nw, dspl,
# rg, workgroup target, minb) scored by geomean %roof over the seven contexts
# matrix.py reports (tools/tune.py).  dspl 0 is the kernel's own rule.
_TUNED: dict[tuple[int, int, int, int], _Knobs] = {
    # D=64
    (14, 2, 64, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (14, 2, 64, 4): {"nseg": 8, "rg": 1, "minb": 2, "nw": 4, "dspl": 0},
    (16, 2, 64, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (16, 2, 64, 4): {"nseg": 8, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (32, 8, 64, 1): {"nseg": 2, "rg": 1, "minb": 2, "nw": 4, "dspl": 0},
    (32, 8, 64, 4): {"nseg": 2, "rg": 1, "minb": 2, "nw": 4, "dspl": 0},
    (32, 32, 64, 1): {"nseg": 1, "rg": 1, "minb": 2, "nw": 2, "dspl": 0},
    (32, 32, 64, 4): {"nseg": 1, "rg": 1, "minb": 4, "nw": 2, "dspl": 0},
    # D=128
    (16, 2, 128, 1): {"nseg": 4, "rg": 2, "minb": 1, "nw": 4, "dspl": 0},
    (16, 2, 128, 4): {"nseg": 4, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (32, 2, 128, 1): {"nseg": 4, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (32, 2, 128, 4): {"nseg": 4, "rg": 4, "minb": 2, "nw": 8, "dspl": 0},
    (28, 4, 128, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 4, "dspl": 2},
    (28, 4, 128, 4): {"nseg": 4, "rg": 1, "minb": 1, "nw": 8, "dspl": 2},
    (32, 4, 128, 1): {"nseg": 4, "rg": 2, "minb": 1, "nw": 4, "dspl": 0},
    (32, 4, 128, 4): {"nseg": 2, "rg": 2, "minb": 4, "nw": 4, "dspl": 0},
    (16, 8, 128, 1): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (16, 8, 128, 4): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (24, 8, 128, 1): {"nseg": 1, "rg": 1, "minb": 4, "nw": 4, "dspl": 0},
    (24, 8, 128, 4): {"nseg": 1, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (32, 8, 128, 1): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (32, 8, 128, 4): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (40, 8, 128, 1): {"nseg": 1, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (40, 8, 128, 4): {"nseg": 2, "rg": 1, "minb": 1, "nw": 8, "dspl": 2},
    (10, 10, 128, 1): {"nseg": 1, "rg": 1, "minb": 2, "nw": 8, "dspl": 0},
    (10, 10, 128, 4): {"nseg": 1, "rg": 1, "minb": 2, "nw": 8, "dspl": 0},
    (32, 32, 128, 1): {"nseg": 1, "rg": 1, "minb": 2, "nw": 2, "dspl": 0},
    (32, 32, 128, 4): {"nseg": 1, "rg": 1, "minb": 1, "nw": 2, "dspl": 0},
    # D=256
    (8, 1, 256, 1): {"nseg": 8, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (8, 1, 256, 4): {"nseg": 8, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (8, 2, 256, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 4},
    (8, 2, 256, 4): {"nseg": 4, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (16, 2, 256, 1): {"nseg": 8, "rg": 2, "minb": 2, "nw": 8, "dspl": 0},
    (16, 2, 256, 4): {"nseg": 4, "rg": 2, "minb": 1, "nw": 4, "dspl": 0},
    (8, 4, 256, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 2, "dspl": 0},
    (8, 4, 256, 4): {"nseg": 4, "rg": 1, "minb": 2, "nw": 4, "dspl": 4},
    (16, 4, 256, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 2, "dspl": 0},
    (16, 4, 256, 4): {"nseg": 4, "rg": 2, "minb": 2, "nw": 8, "dspl": 0},
    (24, 4, 256, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 8, "dspl": 0},
    (24, 4, 256, 4): {"nseg": 4, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (16, 8, 256, 1): {"nseg": 2, "rg": 1, "minb": 2, "nw": 2, "dspl": 0},
    (16, 8, 256, 4): {"nseg": 2, "rg": 1, "minb": 4, "nw": 4, "dspl": 4},
    # D=512
    (8, 1, 512, 1): {"nseg": 16, "rg": 1, "minb": 1, "nw": 8, "dspl": 8},
    (8, 1, 512, 4): {"nseg": 8, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (16, 1, 512, 1): {"nseg": 16, "rg": 1, "minb": 1, "nw": 8, "dspl": 8},
    (16, 1, 512, 4): {"nseg": 8, "rg": 4, "minb": 1, "nw": 8, "dspl": 0},
    (8, 2, 512, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (8, 2, 512, 4): {"nseg": 8, "rg": 1, "minb": 2, "nw": 8, "dspl": 8},
    (16, 2, 512, 1): {"nseg": 8, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (16, 2, 512, 4): {"nseg": 4, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (32, 4, 512, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (32, 4, 512, 4): {"nseg": 2, "rg": 2, "minb": 2, "nw": 8, "dspl": 0},
}


def _knobs_for(
    num_q_heads: int, num_kv_heads: int, head_size: int, max_m: int
) -> _Knobs:
    """Launch knobs for one configuration: the heuristics, then the measured
    overrides on top.

    Args:
        num_q_heads: Query heads.
        num_kv_heads: KV heads.
        head_size: Head dimension.
        max_m: Query tokens per sequence.

    Returns:
        Keyword arguments for `KernelVariant`.
    """
    tuned = _TUNED.get((num_q_heads, num_kv_heads, head_size, max_m), {})
    rg, dspl = _rows_split(num_q_heads // num_kv_heads, max_m, head_size)
    rg = tuned.get("rg", rg)
    knobs: _Knobs = {
        "rg": rg,
        "nseg": _segments_for(num_kv_heads, rg),
        # Four waves where a wave carries the whole head dim: the fewer tiles
        # a workgroup holds, the more of them are in flight at short context.
        "nw": 4 if head_size <= 128 else 8,
        # Short contexts want one segment per block to fill the machine; long
        # ones a few blocks per segment so a workgroup overlaps its own tiles.
        # Only kv-head-rich configurations have the parallelism to spare.
        "minb": 2 if num_kv_heads * rg >= 8 else 1,
    }
    if dspl:
        knobs["dspl"] = dspl
    knobs.update(tuned)
    return knobs


# The JIT-compiled module plus the scratch buffers sized for it.  The module is
# a pybind extension built at runtime, so it has no static type.
_Built = tuple[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class Rdna35HipAttentionBackend(TritonAttentionBackend):
    # bfloat16 stays listed so a bf16 model can still select this backend and
    # be served by the Triton fallback; the kernel itself takes fp16 only,
    # because its products are fp16 WMMAs.
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
        if kv_cache.shape[2] % 16:
            self._reject(f"block size {kv_cache.shape[2]} is not a multiple of 16")
            return None

        # Logical KV cache order is (num_blocks, num_kv_heads, block_size, 2*hs).
        q = kwargs["q"]
        block_size = kv_cache.shape[2]
        variant = KernelVariant(
            **_knobs_for(self.num_heads, self.num_kv_heads, self.head_size, q.shape[0]),
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
            try:
                module = load(variant)
            except VariantBuildError as exc:
                self._reject(str(exc).splitlines()[0])
                return None
            self._built = (module, make_scratch(variant, q.device))
            self._variant = variant
        return self._built

    def _run_attention(self, *, kv_cache: torch.Tensor, **kwargs) -> None:
        built = self._prepare(kv_cache, **kwargs)
        if built is None:
            self.fallback_calls += 1
            super()._run_attention(kv_cache=kv_cache, **kwargs)
            return
        self.kernel_calls += 1

        module, (acc, softmax_max, softmax_sum, arrivals) = built
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
            arrivals,
            # max_seqlen_k, not seqused_k[0]: reading the tensor would be a
            # device-to-host copy, which invalidates a CUDA-graph capture. With
            # the single sequence _prepare insists on, the two are equal.
            kwargs["max_seqlen_k"],
            kwargs["softmax_scale"],
        )
