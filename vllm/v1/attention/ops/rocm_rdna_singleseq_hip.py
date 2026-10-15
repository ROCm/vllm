# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookup of the RDNA3.5 attention kernels built into ``_rocm_C``.

Each row of ``rocm_rdna_singleseq_{decode,prefill}_variants.csv`` is one build,
with its shape and launch knobs as compile-time defines.
"""

import contextlib
import csv
import functools
import io
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class KernelVariant:
    """One decode build: its shape and launch knobs."""

    head_size: int
    num_q_heads: int
    num_kv_heads: int
    max_query_len: int
    page_size: int
    layout: int  # 0 = NHD, 1 = HND
    # Most KV segments per (kv head, row group).
    max_segments: int = 1
    # Workgroups sharing a kv head's query rows.
    row_groups: int = 1
    # Fewest KV blocks per segment.
    min_segment_blocks: int = 1
    # Waves per workgroup.
    waves: int = 8
    # Waves splitting the head dim of a key tile; 0 is the kernel default.
    head_dim_split: int = 0
    # Prefetch a second tile per wave.
    prefetch: int = 0
    # Stage V in LDS beside K.
    v_in_lds: int = 0
    # Per-q-head dot products instead of WMMA.
    dot_product: int = 0
    dtype: torch.dtype = torch.float16
    # Sliding window in keys; 0 is full causal attention.
    window: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"kernel is built for fp16 or bf16, not {self.dtype}")

    @property
    def name(self) -> str:
        """The CSV row joined by "_"."""
        return "rdna_singleseq_decode_" + "_".join(
            map(str, variant_defines(self).values())
        )

    @property
    def rows_padded(self) -> int:
        """Query rows per workgroup, padded to 16."""
        gqa = self.num_q_heads // self.num_kv_heads
        rows = gqa * self.max_query_len // self.row_groups
        return -(-rows // 16) * 16

    def scratch_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """Shapes of the split-KV partials (acc, m/l) and the counter count."""
        if self.dot_product:
            rows = self.num_q_heads * self.max_segments * self.max_query_len
            cnt = self.num_q_heads
        else:
            rows = (
                self.num_kv_heads
                * self.row_groups
                * self.max_segments
                * self.rows_padded
            )
            # Per (kv head, row group): arrivals, generation, two masks.
            cnt = 4 * self.num_kv_heads * self.row_groups
        return (rows, self.head_size), (rows,), cnt


# Read by CMake and the tools; the wheel uses the rows built into _rocm_C.
VARIANTS_CSV = Path(__file__).with_name("rocm_rdna_singleseq_decode_variants.csv")

# CSV column -> KernelVariant field. BF16 sets dtype; every build is HND.
_FIELDS = {
    "HEAD_DIM": "head_size",
    "NUM_Q_HEADS": "num_q_heads",
    "NUM_KV_HEADS": "num_kv_heads",
    "WINDOW": "window",
    "MAX_QUERY_LEN": "max_query_len",
    "PAGE_SIZE": "page_size",
    "MAX_SEGMENTS": "max_segments",
    "ROW_GROUPS": "row_groups",
    "MIN_SEGMENT_BLOCKS": "min_segment_blocks",
    "WAVES": "waves",
    "HEAD_DIM_SPLIT": "head_dim_split",
    "PREFETCH": "prefetch",
    "V_IN_LDS": "v_in_lds",
    "DOT_PRODUCT": "dot_product",
}
_HND = 1


def _lookup_key(v: KernelVariant) -> tuple:
    return (
        v.num_q_heads,
        v.num_kv_heads,
        v.head_size,
        v.max_query_len,
        v.page_size,
        v.window,
        v.dtype,
    )


def _built_rows(op_name: str) -> list[list[str]]:
    """The header and rows built into _rocm_C; [] if none."""
    with contextlib.suppress(ImportError):
        import vllm._rocm_C  # noqa: F401
    try:
        op = getattr(torch.ops._rocm_C, op_name)
    except (AttributeError, RuntimeError):
        return []
    return list(csv.reader(io.StringIO(op())))


@functools.cache
def _table() -> tuple[tuple[str, ...], dict[tuple, KernelVariant]]:
    first, *rows = _built_rows("rdna_singleseq_decode_table") or [[]]
    header = tuple(first)
    table = {}
    for row in rows:
        f = dict(zip(header, map(int, row), strict=True))
        v = KernelVariant(
            **{_FIELDS[c]: x for c, x in f.items() if c != "BF16"},
            layout=_HND,
            dtype=torch.bfloat16 if f["BF16"] else torch.float16,
        )
        table[_lookup_key(v)] = v
    return header, table


def variants() -> list[KernelVariant]:
    """Every decode variant built into _rocm_C."""
    return list(_table()[1].values())


def variant_defines(variant: KernelVariant) -> dict[str, int]:
    """The variant's CSV row, in column order."""
    return {
        c: int(variant.dtype == torch.bfloat16)
        if c == "BF16"
        else getattr(variant, _FIELDS[c])
        for c in _table()[0]
    }


def variant_for(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_query_len: int,
    page_size: int,
    layout: int,
    window: int,
    dtype: torch.dtype,
) -> KernelVariant | None:
    """The build for one call, or None."""
    if layout != _HND:
        return None
    return _table()[1].get(
        (
            num_q_heads,
            num_kv_heads,
            head_size,
            max_query_len,
            page_size,
            window,
            dtype,
        )
    )


class _Builtin:
    """A variant compiled into _rocm_C."""

    def __init__(self, index: int):
        self.index = index

    def decode_attn(self, q, kv_cache, block_table, out, *scratch_and_args):
        torch.ops._rocm_C.rdna_singleseq_decode_attn(
            self.index, q, kv_cache, block_table, out, *scratch_and_args
        )


_loaded: dict[KernelVariant, _Builtin | None] = {}


def load(variant: KernelVariant) -> _Builtin | None:
    """The variant in _rocm_C, or None if unavailable."""
    if variant not in _loaded:
        try:
            op = torch.ops._rocm_C.rdna_singleseq_decode_variant
        except (AttributeError, RuntimeError):
            index = -1
        else:
            index = int(op(list(variant_defines(variant).values())))
        _loaded[variant] = _Builtin(index) if index >= 0 else None
    return _loaded[variant]


def make_scratch(
    variant: KernelVariant, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the split-KV scratch, outside CUDA-graph capture."""
    acc_shape, ml_shape, cnt = variant.scratch_shapes()
    opts = {"dtype": torch.float32, "device": device}
    return (
        torch.empty(acc_shape, **opts),
        torch.empty(ml_shape, **opts),
        torch.empty(ml_shape, **opts),
        # Zeroed once; every launch leaves them zeroed.
        torch.zeros(cnt, dtype=torch.int32, device=device),
    )


def expected_kv_cache_strides(variant: KernelVariant) -> tuple[int, int, int]:
    """Strides of (block, kv head, token) of the HND cache the kernel assumes:
    (NB, HKV, PAGE_SIZE, 2D)."""
    kv_row = 2 * variant.head_size
    page = variant.page_size * variant.num_kv_heads * kv_row
    return page, variant.page_size * kv_row, kv_row


# Most query tokens the decode kernel serves.
MAX_M = 8

# Head dims the decode kernel supports.
SUPPORTED_HEAD_SIZES = (64, 96, 128, 256, 512)


# ---------------------------------------------------------------- prefill
#
# A shape has one row per step of M; a row serves M from its MIN_QUERY_LEN up
# to the next row's, and the first row also every smaller M.

PREFILL_VARIANTS_CSV = Path(__file__).with_name(
    "rocm_rdna_singleseq_prefill_variants.csv"
)
# Fewest query tokens the prefill kernel serves.
PREFILL_MIN_M = MAX_M + 1


@functools.cache
def prefill_target_workgroups(device: torch.device) -> int:
    """Workgroups below which the prefill kernel splits the KV (one per SIMD)."""
    return 4 * torch.cuda.get_device_properties(device).multi_processor_count


_PREFILL_FIELDS = {
    "HEAD_DIM": "head_size",
    "NUM_Q_HEADS": "num_q_heads",
    "NUM_KV_HEADS": "num_kv_heads",
    "WINDOW": "window",
    "PAGE_SIZE": "page_size",
    "MIN_QUERY_LEN": "min_query_len",
    "WAVES": "waves",
    "KEY_TILE": "key_tile",
    "HEAD_DIM_SPLIT": "head_dim_split",
    "V_T": "v_t",
    "DOUBLE_BUFFER": "double_buffer",
    "KEY_WAVES": "key_waves",
    "MAX_SEGMENTS": "max_segments",
    "HEAD_GROUP": "head_group",
}


@dataclass(frozen=True)
class PrefillVariant:
    """One prefill build: its shape, smallest M and launch knobs."""

    head_size: int
    num_q_heads: int
    num_kv_heads: int
    window: int
    page_size: int
    min_query_len: int
    waves: int
    key_tile: int
    head_dim_split: int
    v_t: int
    double_buffer: int
    key_waves: int
    max_segments: int
    head_group: int
    dtype: torch.dtype = torch.float16

    @property
    def name(self) -> str:
        """The CSV row joined by "_"."""
        return "rdna_singleseq_prefill_" + "_".join(
            map(str, prefill_variant_defines(self).values())
        )

    @property
    def block_rows(self) -> int:
        """Query rows per workgroup."""
        return self.waves // (self.head_dim_split * self.key_waves) * 16

    def scratch_rows(self, device: torch.device) -> int:
        """Most split-KV partial rows a launch may write."""
        return 2 * prefill_target_workgroups(device) * self.block_rows


def prefill_variant_defines(variant: PrefillVariant) -> dict[str, int]:
    """The variant's CSV row, in column order."""
    return {
        c: int(variant.dtype == torch.bfloat16)
        if c == "BF16"
        else getattr(variant, _PREFILL_FIELDS[c])
        for c in _prefill_table()[0]
    }


@functools.cache
def _prefill_table() -> tuple[tuple[str, ...], dict[tuple, list[PrefillVariant]]]:
    """Header, and rows by M per (Hq, Hkv, D, page, window, dtype)."""
    first, *rows = _built_rows("rdna_singleseq_prefill_table") or [[]]
    header = tuple(first)
    table: dict[tuple, list[PrefillVariant]] = {}
    for row in rows:
        f = dict(zip(header, map(int, row), strict=True))
        v = PrefillVariant(
            **{_PREFILL_FIELDS[c]: x for c, x in f.items() if c != "BF16"},
            dtype=torch.bfloat16 if f["BF16"] else torch.float16,
        )
        key = (
            v.num_q_heads,
            v.num_kv_heads,
            v.head_size,
            v.page_size,
            v.window,
            v.dtype,
        )
        table.setdefault(key, []).append(v)
    for by_m in table.values():
        by_m.sort(key=lambda v: v.min_query_len)
    return header, table


def prefill_variants(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    window: int,
    dtype: torch.dtype,
) -> list[PrefillVariant]:
    """Every row of one shape, sorted by M."""
    return _prefill_table()[1].get(
        (num_q_heads, num_kv_heads, head_size, page_size, window, dtype), []
    )


def prefill_variant_for(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    layout: int,
    window: int,
    dtype: torch.dtype,
    num_tokens: int,
) -> PrefillVariant | None:
    """The row for num_tokens query tokens, or None."""
    if layout != _HND or num_tokens < PREFILL_MIN_M:
        return None
    rows = prefill_variants(
        num_q_heads, num_kv_heads, head_size, page_size, window, dtype
    )
    served = [v for v in rows if v.min_query_len <= num_tokens]
    return served[-1] if served else (rows[0] if rows else None)


class _BuiltinPrefill:
    """A prefill variant compiled into _rocm_C."""

    def __init__(self, index: int):
        self.index = index

    def prefill_attn(self, q, kv_cache, block_table, out, *scratch_and_args):
        torch.ops._rocm_C.rdna_singleseq_prefill_attn(
            self.index, q, kv_cache, block_table, out, *scratch_and_args
        )


_prefill_loaded: dict[PrefillVariant, _BuiltinPrefill | None] = {}


def load_prefill(variant: PrefillVariant) -> _BuiltinPrefill | None:
    """The variant in _rocm_C, or None if unavailable."""
    if variant not in _prefill_loaded:
        try:
            op = torch.ops._rocm_C.rdna_singleseq_prefill_variant
        except (AttributeError, RuntimeError):
            index = -1
        else:
            index = int(op(list(prefill_variant_defines(variant).values())))
        _prefill_loaded[variant] = _BuiltinPrefill(index) if index >= 0 else None
    return _prefill_loaded[variant]


def make_prefill_scratch(
    variants: list[PrefillVariant], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split-KV scratch (O, max/sum, counters) for any of `variants`."""
    rows = max(v.scratch_rows(device) for v in variants)
    head_size = max(v.head_size for v in variants)
    opts = {"dtype": torch.float32, "device": device}
    return (
        torch.empty(rows * head_size, **opts),
        torch.empty(rows * 2, **opts),
        torch.zeros(
            prefill_target_workgroups(device), dtype=torch.int32, device=device
        ),
    )
