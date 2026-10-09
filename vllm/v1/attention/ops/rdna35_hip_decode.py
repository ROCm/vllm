# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RDNA3.5 paged decode-attention kernels built into ``_rocm_C``.

The kernel in ``csrc/rocm/rdna35_decode_attn.cu`` takes its shapes and launch
knobs as compile-time defines, so one build serves exactly one tuple.  The
builds are the rows of ``rdna35_variants.csv`` beside this file: CMake compiles
every row into ``_rocm_C`` for gfx1151 (``cmake/rdna35_attn.cmake``), and the
backend looks a call's row up here.  A call without a row is served by Triton.

The rows come from ``benchmarks/kernels/gfx1151_decode_attn/tools/tune.py``:
every configuration at M = 1..8, measured in fp16 and used for bf16 too, an
exhaustive search of the knob grid, every point checked against a float
reference and timed over seven contexts, and the winner confirmed against the
previous row in an interleaved A/B.  Batch rows whose winner is the dot
decomposition take the best WMMA knob set instead: the dot path buys latency
for one sequence and loses once a batch fills the machine.
"""

import csv
import functools
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class KernelVariant:
    """The compile-time shape tuple a single build is specialised for.

    One workgroup serves one (kv head, row group, KV segment): every query row
    that reads a kv head -- GQA heads times max_query_len tokens -- is packed into it,
    so the KV cache is read once per kv head rather than once per q head.
    """

    head_size: int
    num_q_heads: int
    num_kv_heads: int
    max_query_len: int
    page_size: int
    layout: int  # 0 = NHD, 1 = HND
    # Most KV segments a (kv head, row group) is split over.  The kernel
    # activates clamp(nblocks // min_segment_blocks, 1, max_segments) of them
    # at run time, so short contexts do not pay for the cross-workgroup merge a
    # long one needs.
    max_segments: int = 1
    # Row groups per kv head: its GQA*max_query_len rows split over row_groups
    # workgroups that each read the whole of its KV, sharing it through L2.
    row_groups: int = 1
    # Least KV blocks an active segment is given.
    min_segment_blocks: int = 1
    # Waves per workgroup.
    waves: int = 8
    # Waves sharing one key tile, each owning 1/head_dim_split of the head
    # dim.  0 keeps the kernel's rule.
    head_dim_split: int = 0
    # 1 keeps a second tile's loads in flight per wave (unshared tiles only;
    # it needs a second tile's registers).
    prefetch: int = 0
    # 1 stages unshared tiles' V in LDS beside K, freeing the tile's
    # registers so the next tile's loads go out before this one's compute.
    v_in_lds: int = 0
    # 1 builds for a batch of sequences (grid.y), with scratch from
    # make_scratch(max_seqs=...); 0 serves exactly one sequence.
    batched: int = 0
    # 1 runs the per-q-head dot-product decomposition instead of WMMA:
    # max_segments segments per q head, waves waves.  row_groups,
    # min_segment_blocks, head_dim_split and prefetch do not apply.
    dot_product: int = 0
    # Element type of Q, the KV cache and the output.
    dtype: torch.dtype = torch.float16
    # Sliding window in keys (a query at p sees p-window+1 .. p); 0 is full
    # causal attention.  Only the window's pages are read.
    window: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"kernel is built for fp16 or bf16, not {self.dtype}")

    @property
    def name(self) -> str:
        """The CSV row joined by "_", as _rocm_C names the variant's launch."""
        return "rdna35_decode_" + "_".join(map(str, variant_defines(self).values()))

    @property
    def rows_padded(self) -> int:
        """Query rows one workgroup carries, rounded up to whole WMMA tiles."""
        gqa = self.num_q_heads // self.num_kv_heads
        rows = gqa * self.max_query_len // self.row_groups
        return -(-rows // 16) * 16

    def scratch_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """Shapes of the (acc, m/l) partials the split-KV merge goes through,
        and the number of counters."""
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
            # Per (kv head, row group): arrival counter, merge generation,
            # and the shared merge's two committed-slice masks.
            cnt = 4 * self.num_kv_heads * self.row_groups
        return (rows, self.head_size), (rows,), cnt


VARIANTS_CSV = Path(__file__).with_name("rdna35_variants.csv")

# Column of the CSV (a define of the kernel) -> KernelVariant field.  BF16
# is the dtype; LAYOUT is not a column, every build is HND.
_FIELDS = {
    "HEAD_DIM": "head_size",
    "NUM_Q_HEADS": "num_q_heads",
    "NUM_KV_HEADS": "num_kv_heads",
    "WINDOW": "window",
    "MAX_QUERY_LEN": "max_query_len",
    "BATCHED": "batched",
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
        v.batched,
    )


@functools.cache
def _table() -> tuple[tuple[str, ...], dict[tuple, KernelVariant]]:
    with VARIANTS_CSV.open(newline="") as fh:
        reader = csv.reader(fh)
        header = tuple(next(reader))
        table = {}
        for row in reader:
            f = dict(zip(header, map(int, row), strict=True))
            v = KernelVariant(
                **{_FIELDS[c]: x for c, x in f.items() if c != "BF16"},
                layout=_HND,
                dtype=torch.bfloat16 if f["BF16"] else torch.float16,
            )
            table[_lookup_key(v)] = v
    return header, table


def variants() -> list[KernelVariant]:
    """Every variant built into _rocm_C, in the CSV's order."""
    return list(_table()[1].values())


def variant_defines(variant: KernelVariant) -> dict[str, int]:
    """The variant's row of the CSV: the kernel's defines, in column order."""
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
    batch: bool,
) -> KernelVariant | None:
    """The build that serves one call, or None if _rocm_C carries none."""
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
            int(batch),
        )
    )


class _Builtin:
    """A variant compiled into _rocm_C."""

    def __init__(self, index: int):
        self.index = index

    def decode_attn(self, q, kv_cache, block_table, out, *scratch_and_args):
        torch.ops._rocm_C.rdna35_decode_attn(
            self.index, q, kv_cache, block_table, out, *scratch_and_args
        )


_loaded: dict[KernelVariant, _Builtin | None] = {}


def load(variant: KernelVariant) -> _Builtin | None:
    """The variant in _rocm_C, or None if this _rocm_C or device has none."""
    if variant not in _loaded:
        try:
            op = torch.ops._rocm_C.rdna35_decode_variant
        except (AttributeError, RuntimeError):
            index = -1
        else:
            index = int(op(list(variant_defines(variant).values())))
        _loaded[variant] = _Builtin(index) if index >= 0 else None
    return _loaded[variant]


def make_scratch(
    variant: KernelVariant, device: torch.device, max_seqs: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the partials once.

    They are passed into the op rather than allocated inside it because the op
    runs under CUDA-graph capture, where an allocation would break the graph.
    ``max_seqs`` > 0 gives every tensor a leading sequence dimension, for
    batches of up to that many sequences; 0 is one sequence without it.
    """
    acc_shape, ml_shape, cnt = variant.scratch_shapes()
    lead = (max_seqs,) if max_seqs else ()
    opts = {"dtype": torch.float32, "device": device}
    return (
        torch.empty(lead + acc_shape, **opts),
        torch.empty(lead + ml_shape, **opts),
        torch.empty(lead + ml_shape, **opts),
        # Arrival counters, merge generations and committed-slice masks
        # (scratch_shapes).  Zeroed once: the kernel resets the counters as it
        # consumes them, resets each mask a generation ahead of its use and
        # only compares generations for change, so every later launch starts
        # clean without the host writing here -- which it could not do under
        # graph capture anyway.
        torch.zeros(lead + (cnt,), dtype=torch.int32, device=device),
    )


def scratch_bytes(variant: KernelVariant) -> int:
    """Bytes of scratch one sequence needs."""
    acc_shape, ml_shape, cnt = variant.scratch_shapes()
    return 4 * (acc_shape[0] * acc_shape[1] + 2 * ml_shape[0] + cnt)


def expected_kv_cache_strides(variant: KernelVariant) -> tuple[int, int, int]:
    """Element strides of (block, kv head, token) the kernel indexes.

    The kernel walks the KV cache with its own arithmetic instead of reading
    the tensor's strides, so the caller must confirm the tensor really is laid
    out this way. Getting this wrong is silent: the kernel would read the wrong
    addresses and still return finite numbers.
    """
    kv_row = 2 * variant.head_size
    page = variant.page_size * variant.num_kv_heads * kv_row
    if variant.layout == 0:  # NHD: (NB, PAGE_SIZE, HKV, 2D)
        return page, kv_row, variant.num_kv_heads * kv_row
    return page, variant.page_size * kv_row, kv_row  # HND: (NB, HKV, PAGE_SIZE, 2D)


# Query tokens per sequence the kernel serves: decode and speculative decode.
MAX_M = 8

# The kernel's V and K slices are one b64 or b128 per lane, which the head
# dims below allow; its GQA packing assumes the q heads divide evenly over the
# kv heads.
SUPPORTED_HEAD_SIZES = (64, 128, 256, 512)


# ---------------------------------------------------------------- prefill
#
# rdna35_prefill_attn.cu serves one sequence of M query tokens after a cached
# prefix.  M is a run-time argument; the knobs are compile-time, one build per
# row of rdna35_prefill_variants.csv, and a shape has a row per step of M
# (MIN_QUERY_LEN: the row serves M from there up to the next row's).  Rows were
# tuned in fp16 over M 128..8192 and prefixes 0..16k, against segmented and
# Triton, and bf16 shares them.

PREFILL_VARIANTS_CSV = Path(__file__).with_name("rdna35_prefill_variants.csv")
# Query tokens a prefill launch serves; below, the decode kernel or Triton.
PREFILL_MIN_M = 128
PREFILL_MAX_M = 8192
# The kernel splits the KV only below this many workgroups, so a split launch
# has fewer than twice as many row tiles (kTargetWorkgroups in the kernel).
_PREFILL_TARGET_WORKGROUPS = 80

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
    """One prefill build: a shape, the smallest M its row serves, its knobs.

    A workgroup serves (kv head, query tile, KV segment); each wave 16 query
    rows over 1/head_dim_split of the head dim and every key_waves-th 16-key
    sub-tile of a step of key_tile keys, staged in LDS for the workgroup.
    """

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
        """The CSV row joined by "_", as _rocm_C names the variant's launch."""
        return "rdna35_prefill_" + "_".join(
            map(str, prefill_variant_defines(self).values())
        )

    @property
    def block_rows(self) -> int:
        """Query rows of one workgroup's tile."""
        return self.waves // (self.head_dim_split * self.key_waves) * 16

    def scratch_rows(self) -> int:
        """Rows of split-KV partials a launch may write."""
        return 2 * _PREFILL_TARGET_WORKGROUPS * self.block_rows


def prefill_variant_defines(variant: PrefillVariant) -> dict[str, int]:
    """The variant's row of the CSV, in column order."""
    return {
        c: int(variant.dtype == torch.bfloat16)
        if c == "BF16"
        else getattr(variant, _PREFILL_FIELDS[c])
        for c in _prefill_table()[0]
    }


@functools.cache
def _prefill_table() -> tuple[tuple[str, ...], dict[tuple, list[PrefillVariant]]]:
    """Header, and per (Hq, Hkv, D, page, window, dtype) the rows by M."""
    table: dict[tuple, list[PrefillVariant]] = {}
    with PREFILL_VARIANTS_CSV.open(newline="") as fh:
        reader = csv.reader(fh)
        header = tuple(next(reader))
        for row in reader:
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
    for rows in table.values():
        rows.sort(key=lambda v: v.min_query_len)
    return header, table


def prefill_variants(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    window: int,
    dtype: torch.dtype,
) -> list[PrefillVariant]:
    """Every row of one shape, by the M it starts at."""
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
    """The row that serves a prefill of num_tokens query tokens, or None."""
    if layout != _HND or not PREFILL_MIN_M <= num_tokens <= PREFILL_MAX_M:
        return None
    rows = prefill_variants(
        num_q_heads, num_kv_heads, head_size, page_size, window, dtype
    )
    served = [v for v in rows if v.min_query_len <= num_tokens]
    return served[-1] if served else None


class _BuiltinPrefill:
    """A prefill variant compiled into _rocm_C."""

    def __init__(self, index: int):
        self.index = index

    def prefill_attn(self, q, kv_cache, block_table, out, *scratch_and_args):
        torch.ops._rocm_C.rdna35_prefill_attn(
            self.index, q, kv_cache, block_table, out, *scratch_and_args
        )


_prefill_loaded: dict[PrefillVariant, _BuiltinPrefill | None] = {}


def load_prefill(variant: PrefillVariant) -> _BuiltinPrefill | None:
    """The variant in _rocm_C, or None if this _rocm_C or device has none."""
    if variant not in _prefill_loaded:
        try:
            op = torch.ops._rocm_C.rdna35_prefill_variant
        except (AttributeError, RuntimeError):
            index = -1
        else:
            index = int(op(list(prefill_variant_defines(variant).values())))
        _prefill_loaded[variant] = _BuiltinPrefill(index) if index >= 0 else None
    return _prefill_loaded[variant]


def make_prefill_scratch(
    variants: list[PrefillVariant], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split-KV partials (O, then max and sum per row) and arrival counters,
    enough for any of `variants`.  The counters start at zero and every
    launch leaves them so."""
    rows = max(v.scratch_rows() for v in variants)
    head_size = max(v.head_size for v in variants)
    opts = {"dtype": torch.float32, "device": device}
    return (
        torch.empty(rows * head_size, **opts),
        torch.empty(rows * 2, **opts),
        torch.zeros(_PREFILL_TARGET_WORKGROUPS, dtype=torch.int32, device=device),
    )
