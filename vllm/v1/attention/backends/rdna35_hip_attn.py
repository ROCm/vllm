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
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.rdna35_hip_decode import (
    KernelVariant,
    VariantBuildError,
    expected_kv_cache_strides,
    load,
    make_scratch,
    scratch_bytes,
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
    rspl: int
    pf: int
    cpub: int
    vinlds: int
    dot: int
    bfly: int
    gt: int


# Every shipped configuration, measured: coordinate descent over (nw, dspl,
# rg, workgroup target, minb) scored by geomean %roof over the seven contexts
# matrix.py reports (tools/tune.py).  dspl 0 is the kernel's own rule.
_TUNED: dict[tuple[int, int, int, int], _Knobs] = {
    # D=64
    (14, 2, 64, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (14, 2, 64, 4): {"nseg": 4, "rg": 1, "minb": 2, "nw": 4, "vinlds": 1},
    (16, 2, 64, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (16, 2, 64, 4): {"nseg": 4, "rg": 2, "minb": 2, "nw": 4, "vinlds": 1},
    (32, 8, 64, 1): {"nw": 4, "rg": 1, "minb": 2, "nseg": 1, "pf": 1},
    (32, 8, 64, 4): {"nw": 4, "rg": 1, "minb": 1, "nseg": 1, "pf": 1},
    (32, 32, 64, 1): {"nseg": 1, "rg": 1, "minb": 2, "nw": 2, "dspl": 0},
    (32, 32, 64, 4): {"nseg": 1, "rg": 1, "minb": 4, "nw": 2, "dspl": 0},
    # D=128
    (16, 2, 128, 1): {"nseg": 4, "rg": 2, "minb": 2, "nw": 4, "vinlds": 1},
    (16, 2, 128, 4): {"nseg": 4, "rg": 2, "minb": 2, "nw": 4, "vinlds": 1},
    (32, 2, 128, 1): {"nseg": 4, "rg": 2, "minb": 2, "nw": 4, "vinlds": 1},
    (32, 2, 128, 4): {"nseg": 4, "rg": 4, "minb": 2, "nw": 4, "vinlds": 1},
    (28, 4, 128, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 4, "dspl": 2},
    (28, 4, 128, 4): {"nseg": 4, "rg": 1, "minb": 1, "nw": 8, "dspl": 2},
    (32, 4, 128, 1): {"nw": 4, "rg": 2, "minb": 4, "nseg": 2, "vinlds": 1},
    (32, 4, 128, 4): {"nseg": 2, "rg": 2, "minb": 4, "nw": 4, "vinlds": 1},
    (16, 8, 128, 1): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (16, 8, 128, 4): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (24, 8, 128, 1): {"nseg": 1, "rg": 1, "minb": 4, "nw": 4, "dspl": 0},
    (24, 8, 128, 4): {"nseg": 1, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (32, 8, 128, 1): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (32, 8, 128, 4): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (40, 8, 128, 1): {"nseg": 1, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (40, 8, 128, 4): {"nseg": 2, "rg": 1, "minb": 1, "nw": 8, "dspl": 2},
    (10, 10, 128, 1): {"nseg": 1, "rg": 1, "minb": 2, "nw": 4, "vinlds": 1},
    (10, 10, 128, 4): {"nseg": 1, "rg": 1, "minb": 2, "nw": 4, "vinlds": 1},
    (32, 32, 128, 1): {"nseg": 1, "rg": 1, "minb": 2, "nw": 2, "dspl": 0},
    (32, 32, 128, 4): {"nseg": 1, "rg": 1, "minb": 1, "nw": 2, "dspl": 0},
    # D=256
    (8, 1, 256, 1): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4},
    (8, 1, 256, 4): {"nseg": 16, "rg": 2, "minb": 1, "nw": 4, "dspl": 2, "vinlds": 1},
    (8, 2, 256, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4},
    (8, 2, 256, 4): {"nseg": 4, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (16, 2, 256, 1): {"nseg": 8, "rg": 2, "minb": 2, "nw": 8, "dspl": 0},
    (16, 2, 256, 4): {"nseg": 8, "rg": 2, "minb": 1, "nw": 4, "dspl": 2},
    (8, 4, 256, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4, "gt": 1},
    (8, 4, 256, 4): {"nw": 4, "rg": 1, "minb": 2, "nseg": 2, "dspl": 4, "pf": 1},
    (16, 4, 256, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 2, "gt": 1},
    (16, 4, 256, 4): {"nw": 4, "rg": 2, "minb": 1, "nseg": 2},
    (24, 4, 256, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 2},
    (24, 4, 256, 4): {"nw": 4, "rg": 1, "minb": 1, "nseg": 4, "rspl": 2, "pf": 1},
    (16, 8, 256, 1): {"nseg": 2, "rg": 1, "minb": 2, "nw": 2, "dspl": 0},
    (16, 8, 256, 4): {"nseg": 2, "rg": 1, "minb": 4, "nw": 4, "dspl": 4},
    # D=512
    (8, 1, 512, 1): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4},
    (8, 1, 512, 4): {"nseg": 8, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (16, 1, 512, 1): {"nw": 8, "rg": 1, "minb": 1, "nseg": 16, "dspl": 8},
    (16, 1, 512, 4): {"nseg": 8, "rg": 4, "minb": 1, "nw": 8, "dspl": 0},
    (8, 2, 512, 1): {"dot": 1, "nw": 8, "nseg": 2},
    (8, 2, 512, 4): {"nw": 8, "rg": 1, "minb": 2, "nseg": 4, "dspl": 8, "pf": 1},
    (16, 2, 512, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4},
    (16, 2, 512, 4): {"nseg": 4, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (32, 4, 512, 1): {"dot": 1, "nw": 4, "nseg": 2, "bfly": 4},
    (32, 4, 512, 4): {"nseg": 2, "rg": 2, "minb": 2, "nw": 8, "dspl": 0},
}

# bf16 rows that differ from _TUNED.  The dot decomposition's bf16 products
# (v_dot2_f32_bf16) fall behind at long context -- 0.73-0.97x of the WMMA
# rows at 32k -- so in bf16 these shapes keep their WMMA configuration.
_TUNED_BF16: dict[tuple[int, int, int, int], _Knobs] = {
    (8, 1, 256, 1): {"nseg": 8, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (8, 2, 256, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 4},
    (8, 4, 256, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 2, "dspl": 0},
    (16, 4, 256, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 2, "dspl": 0},
    (24, 4, 256, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 8, "dspl": 0},
    (8, 1, 512, 1): {"nseg": 16, "rg": 1, "minb": 1, "nw": 8, "dspl": 8},
    (8, 2, 512, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (16, 2, 512, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (32, 4, 512, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
}

# Sliding-window layers, keyed on (Hq, Hkv, D, M, window).  They read only the
# window's keys whatever the sequence length, so they are tuned apart from the
# full-attention layers that share their shape (gemma interleaves the two).
_TUNED_SWA: dict[tuple[int, int, int, int, int], _Knobs] = {
    (8, 1, 256, 1, 512): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4},
    (8, 1, 256, 4, 512): {
        "nw": 4,
        "rg": 1,
        "minb": 1,
        "nseg": 16,
        "rspl": 2,
        "pf": 1,
        "cpub": 1,
    },
    (8, 2, 256, 1, 512): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4, "gt": 1},
    (8, 2, 256, 4, 512): {
        "nw": 4,
        "rg": 1,
        "minb": 1,
        "nseg": 8,
        "dspl": 4,
        "pf": 1,
    },
    (8, 4, 256, 1, 1024): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4, "gt": 1},
    (8, 4, 256, 4, 1024): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4, "gt": 1},
    (8, 4, 256, 1, 4096): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 2},
    (8, 4, 256, 4, 4096): {
        "nw": 4,
        "rg": 1,
        "minb": 2,
        "nseg": 4,
        "dspl": 2,
        "vinlds": 1,
    },
    (16, 8, 256, 1, 1024): {"dot": 1, "nw": 4, "nseg": 1, "bfly": 4, "gt": 1},
    (16, 8, 256, 4, 1024): {"nw": 4, "rg": 2, "minb": 2, "nseg": 1},
    (32, 16, 256, 1, 1024): {"dot": 1, "nw": 4, "nseg": 1, "bfly": 4},
    (32, 16, 256, 4, 1024): {"nw": 2, "rg": 1, "minb": 2, "nseg": 1},
}


def _knobs_for(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_m: int,
    window: int = 0,
    dtype: torch.dtype = torch.float16,
    batch: bool = False,
) -> _Knobs:
    """Launch knobs for one configuration: the heuristics, then the measured
    overrides on top.

    Args:
        num_q_heads: Query heads.
        num_kv_heads: KV heads.
        head_size: Head dimension.
        max_m: Query tokens per sequence.
        window: Sliding window in keys, 0 for full attention.
        dtype: Element type; bf16 rows may differ (`_TUNED_BF16`).
        batch: More than one sequence.  The dot decomposition buys latency for
            one sequence and loses to WMMA once a batch fills the machine, so
            a dot row gives way to its WMMA row (`_TUNED_BF16`) or the
            heuristics.

    Returns:
        Keyword arguments for `KernelVariant`.
    """
    if window:
        tuned = _TUNED_SWA.get(
            (num_q_heads, num_kv_heads, head_size, max_m, window), {}
        )
    else:
        key = (num_q_heads, num_kv_heads, head_size, max_m)
        tuned = _TUNED.get(key, {})
        if dtype == torch.bfloat16 or (batch and tuned.get("dot")):
            tuned = _TUNED_BF16.get(key, tuned)
    if batch and tuned.get("dot"):
        tuned = {}
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

# Split-KV scratch, one set per variant and device, shared by every layer that
# runs it: layers launch one after another on one stream, and the counters are
# left clean by each launch.  Sized for the scheduler's batch, up to a budget;
# a larger batch falls back to Triton.
_SCRATCH: dict[tuple[KernelVariant, torch.device], tuple[int, Any]] = {}
_SCRATCH_BUDGET = 64 * 1024**2
# Mixed batches are split only from this head size up (OPTIMIZATIONS 028).
_SPLIT_MIN_HEAD_SIZE = 256
_SPLIT_MIN_DECODES = 16
_SPLIT_MIN_PREFILL = 256
# Query tokens per sequence the kernel serves: decode and speculative decode.
_MAX_M = 8


class Rdna35HipAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    """Triton's metadata, with the batch ordered decodes first and the
    number of leading uniform decodes counted, so that a batch mixing
    prefills and decodes can send its decodes to the kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        if self.reorder_batch_threshold is not None:
            self.reorder_batch_threshold = min(self.reorder_batch_threshold, _MAX_M)

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


class Rdna35HipAttentionBackend(TritonAttentionBackend):
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

    @staticmethod
    def get_builder_cls() -> type["Rdna35HipAttentionMetadataBuilder"]:
        return Rdna35HipAttentionMetadataBuilder

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        # Keys of one head contiguous in a page.  The decode kernel is tuned
        # on it, and the Triton paths this backend keeps (prefill, batch > 1)
        # run faster on it too: 3-5 % prefill, 13-27 % batched decode.
        return "HND"

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
        try:
            from vllm.config import get_current_vllm_config

            max_seqs = get_current_vllm_config().scheduler_config.max_num_seqs
        except Exception:
            max_seqs = 1
        self._max_seqs = max(1, max_seqs)
        self._cap = 0

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
        if max_m > _MAX_M:
            self._reject(f"{max_m} query tokens per sequence, more than {_MAX_M}")
            return None
        dtype = kwargs["q"].dtype
        if dtype not in (torch.float16, torch.bfloat16):
            self._reject(f"kernel is fp16 or bf16 only, got {dtype}")
            return None
        if kv_cache.dtype != dtype:
            self._reject(f"KV cache is {kv_cache.dtype}, query is {dtype}")
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
            **_knobs_for(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                max_m,
                win,
                dtype,
                nseq > 1,
            ),
            head_size=self.head_size,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_m=max_m,
            block_size=block_size,
            layout=0 if kv_cache.stride(1) < kv_cache.stride(2) else 1,
            dtype=dtype,
            window=win,
            batch=int(nseq > 1),
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

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, *a, **kw):
        # The split of a mixed batch needs the metadata, which Triton's forward
        # does not hand down to _run_attention.
        self._metadata = attn_metadata
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata, *a, **kw
        )

    def _split_mixed(self, kv_cache: torch.Tensor, kwargs: dict) -> bool:
        """Serve a batch of decodes followed by prefills in two launches: the
        decodes on the kernel, the prefills on Triton.  Returns False, having
        done nothing, when the batch is not like that or the kernel cannot
        take its decodes.

        Not under CUDA-graph capture: the split is host arithmetic on this
        step's batch, and a captured graph would replay one step's split.
        """
        md = getattr(self, "_metadata", None)
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
        pre = dict(kwargs)
        for k in ("q", "out"):
            pre[k] = kwargs[k][ndt:]
        for k in ("seqused_k", "block_table", "k_descale", "v_descale"):
            if pre.get(k) is not None:
                pre[k] = kwargs[k][nd:]
        pre["cu_seqlens_q"] = kwargs["cu_seqlens_q"][nd:] - ndt
        super()._run_attention(kv_cache=kv_cache, **pre)
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

    def _run_attention(self, *, kv_cache: torch.Tensor, **kwargs) -> None:
        if self._split_mixed(kv_cache, kwargs):
            self.kernel_calls += 1
            return
        built = self._prepare(kv_cache, **kwargs)
        if built is None:
            self.fallback_calls += 1
            super()._run_attention(kv_cache=kv_cache, **kwargs)
            return
        self.kernel_calls += 1
        self._launch(built, kv_cache, kwargs)
