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
    expected_kv_cache_strides,
    load,
    make_scratch,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)

# The kernel's wave split assumes 8 fp16 per lane, and its GQA indexing assumes
# the q heads divide evenly over the kv heads.
_SUPPORTED_HEAD_SIZES = (64, 128, 256, 512)

# The board has 20 WGPs and the grid is (NSEG, num_q_heads), so a model with
# few heads leaves most of it idle: 8 heads is 8 workgroups.  Splitting the KV
# range across NSEG workgroups refills the machine.  Measured on Hq=8/Hkv=4,
# S=8192: 45.0% of roofline at NSEG=1 against 81.0% at NSEG=4.
#
# NSEG depends only on the head count, which is fixed when the graph is
# captured, so this stays within the one-configuration rule -- it is S that a
# configuration may not depend on.
_TARGET_WORKGROUPS = 32


def _segments_for(num_q_heads: int) -> int:
    """Largest power of two that keeps the grid near _TARGET_WORKGROUPS."""
    ratio = max(1, _TARGET_WORKGROUPS // max(1, num_q_heads))
    return 1 << (ratio.bit_length() - 1)


# KV head counts where giving two waves a query token each, rather than one
# wave all of them, wins at both ends of the context range.  This is a measured
# table, not a rule: sweeping MSPLIT against Hkv in {2,4,8,16} at a constant
# GQA of 2 and the NSEG above gives, as a change from MSPLIT=1,
#
#   Hkv    S=128    S=32768
#     2    +3.8%      -4.3%
#     4    +7.5%      +4.5%
#     8   +11.1%      +8.1%
#    16    +3.6%      -6.4%
#
# so the benefit is not monotonic in head count, bytes per token, or KV stream
# count -- all three were checked and none orders these four points.  Only 4
# and 8 win at long context, and extrapolating past what was measured is how
# the WGP-alignment rule got into the design document three refutations ago.
# Re-measure before widening this.
_MSPLIT_KV_HEADS = frozenset({4, 8})


def _msplit_for(num_kv_heads: int, max_m: int) -> int:
    """Waves sharing the query-token dimension; 1 keeps the old decomposition."""
    if num_kv_heads in _MSPLIT_KV_HEADS and max_m % 2 == 0:
        return 2
    return 1


# Best knobs measured per configuration, keyed on (Hq, Hkv, D, M).
#
# The heuristics above are rules fitted to the whole shape table; this is the
# exceptions list, and it wins where both apply.  Rows carry only the knobs
# actually measured -- a partial row is normal, and a configuration absent here
# behaves exactly as it did before the table existed, so landing a row can only
# affect the configuration it names.
#
# M is part of the key because every knob we have measured disagrees across it:
# BFLY saturates at 2 for D=128 at M=1 but wants 3 or 4 at M=4, and D=512 at
# M=4 is forced onto MSPLIT=2 by the LDS ceiling while M=1 is not.  A row tuned
# at one M says nothing about the other.
#
# Provenance is OPTIMIZATIONS.md (per knob) and golden/ (per shape).  Do not add
# a row without a measurement behind it; the point of this table is that it is
# the measured exceptions, not a second set of guesses.
class _Knobs(TypedDict, total=False):
    """Launch knobs a configuration may pin.  Total=False: a row sets only what
    was measured."""

    nseg: int
    msplit: int
    bfly: int
    gridt: int
    dpl: int
    ldsplit: int


_TUNED: dict[tuple[int, int, int, int], _Knobs] = {
    # BFLY: entry 001.  D=512 takes 3 rather than 4 because 4 regresses 7.9% at
    # S=32768; D=256 never regresses at 4 and is ~3% faster there at short
    # context.  D=128 at M=1 saturates at 2 -- 2, 3 and 4 are within noise of
    # each other, so it takes the one that spends least on the VALU.
    #
    # DPL: entry 007, on the four M=4 rows that carry it below.  DPL=32 halves
    # LPR, so the score butterfly reduces over 16 lanes in four
    # dependent stages instead of five.  Ablating the kernel's three VALU
    # blocks showed that is the only one whose cost is real: thinning P@V or
    # Q@K makes the kernel *slower* (they hide memory latency for free), while
    # thinning the butterfly is worth 11-15 %.  It needs LDSPLIT=2 because
    # SUB doubles with DPL and the partials no longer fit LDS.
    #
    # M=4 only.  At M=1 the kernel already runs at 94 % of the bus, so a
    # shorter butterfly buys nothing and the extra registers and the chunked
    # epilogue cost 2.5-6.6 % -- it loses on all five M=1 configurations.
    # (8,2,512,4) is excluded too: it regresses five of seven contexts.
    #
    # These rows regress S=128 by 4-9 %, which the old percentage rule in
    # HANDOFF 4.1 forbade.  In absolute time that is 0.5-0.7 us against 137-203
    # us saved at S=32768, so the rule now reads in microseconds; see the
    # handoff.  KPW=8 was re-examined under the same criterion and stays
    # rejected -- it does not combine with DPL=32 (SUB doubles, so KPWE does
    # too) and regresses up to +676 us.
    (16, 2, 512, 4): {"bfly": 3, "dpl": 32, "ldsplit": 2},
    (16, 2, 256, 4): {"bfly": 4},
    (32, 8, 128, 1): {"bfly": 2},
    # D=512, all five configurations, BFLY swept 0..4 at M in {1,4} over seven
    # contexts.  Chosen by best geomean among the values that regress no cell
    # by more than the 3.3% measured worst-case harness noise, so a row can be
    # a small win but never a knowingly bad trade at any context.
    #
    # M=4 is where this knob pays: -4.5% to -22.6% geomean.  M=1 is already
    # near the bus on most of these shapes and moves by -0.7% to -5.3%; those
    # rows are recorded because they were measured, not because they matter.
    # Note 8/2 and 8/1 disagree at M=1 (1 against 4) and 16/1 and 16/2 disagree
    # at M=4 (4 against 3) -- no rule fits these, which is why it is a table.
    (8, 1, 512, 1): {"bfly": 4},
    (8, 1, 512, 4): {"bfly": 3, "dpl": 32, "ldsplit": 2},
    # GRIDT: dispatching head-fastest instead of segment-fastest fixes the one
    # outlier of the M=1 table -- this configuration read 53.2% of roofline and
    # 1.60x against Triton where its neighbours were at 90-99% and ~3x. It now
    # reads 97.3% and 3.02x, and it improves at all seven contexts, by more the
    # longer the context: -1.0% at S=128 rising to -45.9% at S=32768,
    # reproduced in two independent runs.
    #
    # Deliberately not a rule. The same knob costs 4-15% on (8,1,512,4) and
    # 2-15% on (16,2,512,1), and nothing in head count, GQA or KV stream count
    # separates the three -- only measurement does. Across the whole D=512
    # matrix it is neutral (geomean 1.001), which is exactly the signature of a
    # knob that belongs in the exceptions list rather than in _knobs_for.
    (8, 2, 512, 1): {"bfly": 1, "gridt": 1},
    # Retuned after the LDS padding (OPTIMIZATIONS.md 003) moved this one: 3
    # became 741 us at S=32768 against 590 for 2, a 20% swing.  The only
    # configuration the LDS change invalidated -- the rest were re-checked and
    # held.
    (8, 2, 512, 4): {"bfly": 2},
    (16, 1, 512, 1): {"bfly": 4},
    (16, 1, 512, 4): {"bfly": 4, "dpl": 32, "ldsplit": 2},
    (16, 2, 512, 1): {"bfly": 4},
    (32, 4, 512, 1): {"bfly": 4},
    (32, 4, 512, 4): {"bfly": 4, "dpl": 32, "ldsplit": 2},
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
    knobs: _Knobs = {
        "nseg": _segments_for(num_q_heads),
        "msplit": _msplit_for(num_kv_heads, max_m),
    }
    knobs.update(_TUNED.get((num_q_heads, num_kv_heads, head_size, max_m), {}))
    return knobs


# The JIT-compiled module plus the scratch buffers sized for it.  The module is
# a pybind extension built at runtime, so it has no static type.
_Built = tuple[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


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
