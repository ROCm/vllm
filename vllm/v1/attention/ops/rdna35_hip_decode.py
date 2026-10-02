# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader for the RDNA3.5 paged decode-attention kernel.

The kernel in ``csrc/rocm/rdna35_decode_attn.cu`` takes its shapes as
compile-time defines, so one build serves exactly one shape tuple.  The
variants the tables below name are compiled with the rest of vLLM into
``_rocm_C`` (the list is ``csrc/rocm/rdna35_attn/variants.def``, written by
``write_variants_def``); any other variant is served by Triton.

The kernel tools (benchmarks/kernels/gfx1151_decode_attn) search variants
that are not in the list, so with ``VLLM_RDNA35_ATTN_JIT`` set every variant
is instead compiled here with ``torch.utils.cpp_extension.load``, from a
source checkout, in a few seconds rather than a vLLM rebuild.
"""

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# resolve() matters: ninja re-expands the path through a shell, so a symlinked
# checkout whose name contains '$' would be mangled into a missing file.
_CSRC = Path(__file__).resolve().parents[4] / "csrc" / "rocm"
_SOURCE = _CSRC / "rdna35_decode_attn.cu"
_LAUNCH_H = _CSRC / "rdna35_attn" / "launch.h"


@dataclass(frozen=True)
class KernelVariant:
    """The compile-time shape tuple a single build is specialised for.

    One workgroup serves one (kv head, row group, KV segment): every query
    row that reads a kv head -- GQA heads times max_m tokens -- is packed into
    it, so the KV cache is read once per kv head rather than once per q head.
    """

    head_size: int
    num_q_heads: int
    num_kv_heads: int
    max_m: int
    block_size: int
    layout: int  # 0 = NHD, 1 = HND
    # Most KV segments a (kv head, row group) is split over.  The kernel
    # activates clamp(nblocks // minb, 1, nseg) of them at run time, so short
    # contexts do not pay for the cross-workgroup merge a long one needs.
    nseg: int = 1
    # Row groups per kv head: its GQA*max_m rows split over rg workgroups that
    # each read the whole of its KV, sharing it through L2.
    rg: int = 1
    # Least KV blocks an active segment is given.
    minb: int = 1
    # Waves per workgroup.
    nw: int = 8
    # Waves sharing one key tile, each owning 1/dspl of the head dim.  0 keeps
    # the kernel's rule.
    dspl: int = 0
    # Waves sharing one key tile, each carrying 1/rspl of the row tiles: the
    # in-workgroup form of rg, with the tile loaded once and shared in LDS.
    rspl: int = 1
    # 1 keeps a second tile's loads in flight per wave (unshared tiles only;
    # it needs a second tile's registers).
    pf: int = 0
    # 1 stages split-KV partials in LDS and writes them a line at a time
    # instead of from registers.
    cpub: int = 0
    # 1 stages unshared tiles' V in LDS beside K, freeing the tile's
    # registers so the next tile's loads go out before this one's compute.
    vinlds: int = 0
    # 1 builds for a batch of sequences (grid.y), with scratch from
    # make_scratch(max_seqs=...); 0 serves exactly one sequence.
    batch: int = 0
    # 1 runs the per-q-head dot-product decomposition instead of WMMA: nseg
    # segments per q head, nw waves, bfly its butterfly split, gt 1 to
    # dispatch heads fastest.  rg, minb, dspl, rspl, pf and cpub do not apply.
    dot: int = 0
    bfly: int = 0
    gt: int = 0
    mutate: int = 0
    # Measurement only: skips blocks of work and returns wrong numbers.  See
    # the ABLATE comment in the kernel.
    ablate: int = 0
    # Element type of Q, the KV cache and the output.
    dtype: torch.dtype = torch.float16
    # Sliding window in keys (a query at p sees p-window+1 .. p); 0 is full
    # causal attention.  Only the window's pages are read.
    window: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"kernel is built for fp16 or bf16, not {self.dtype}")

    @property
    def suffix(self) -> str:
        return (
            f"d{self.head_size}_q{self.num_q_heads}_kv{self.num_kv_heads}"
            f"_m{self.max_m}_bs{self.block_size}_l{self.layout}"
            f"_n{self.nseg}_rg{self.rg}_mb{self.minb}_w{self.nw}"
            f"{'' if not self.dspl else f'_ds{self.dspl}'}"
            f"{'' if self.rspl == 1 else f'_rs{self.rspl}'}"
            f"{'' if not self.pf else '_pf'}"
            f"{'' if not self.cpub else '_cp'}"
            f"{'' if not self.vinlds else '_vl'}"
            f"{'' if not self.batch else '_batch'}"
            f"{'' if not self.dot else f'_dot_bf{self.bfly}_gt{self.gt}'}"
            f"_mut{self.mutate}"
            f"{'' if not self.ablate else f'_ab{self.ablate}'}"
            f"{'_bf16' if self.dtype == torch.bfloat16 else ''}"
            f"{'' if not self.window else f'_win{self.window}'}"
        )

    @property
    def name(self) -> str:
        return f"rdna35_decode_{self.suffix}"

    @property
    def rows_padded(self) -> int:
        """Query rows one workgroup carries, rounded up to whole WMMA tiles."""
        gqa = self.num_q_heads // self.num_kv_heads
        rows = gqa * self.max_m // self.rg
        return -(-rows // 16) * 16

    def scratch_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """Shapes of the (acc, m/l) partials the split-KV merge goes through,
        and the number of counters."""
        if self.dot:
            rows = self.num_q_heads * self.nseg * self.max_m
            cnt = self.num_q_heads
        else:
            rows = self.num_kv_heads * self.rg * self.nseg * self.rows_padded
            cnt = 2 * self.num_kv_heads * self.rg
        return (rows, self.head_size), (rows,), cnt


_loaded: dict[KernelVariant, Any] = {}

VARIANTS_DEF = _CSRC / "rdna35_attn" / "variants.def"


def variant_key(variant: KernelVariant) -> list[int]:
    """The fields of the variant's variants.def line, in its order."""
    v = variant
    return [
        v.head_size,
        v.num_q_heads,
        v.num_kv_heads,
        v.max_m,
        v.block_size,
        v.layout,
        v.nseg,
        v.rg,
        v.minb,
        v.nw,
        v.dspl,
        v.rspl,
        v.pf,
        v.cpub,
        v.vinlds,
        v.batch,
        v.dot,
        v.bfly,
        v.gt,
        v.window,
        int(v.dtype == torch.bfloat16),
    ]


def variant_group(variant: KernelVariant) -> str:
    """The unit a variant is compiled in: one per configuration and dtype, so
    re-tuning a configuration rebuilds only its own unit."""
    v = variant
    dtype = "bf16" if v.dtype == torch.bfloat16 else "fp16"
    return f"q{v.num_q_heads}_kv{v.num_kv_heads}_d{v.head_size}_w{v.window}_{dtype}"


def write_variants_def(variants: "Iterable[KernelVariant]", path: Path) -> None:
    """Write the variants.def _rocm_C is built from."""
    lines = sorted(
        {
            f"RDNA35_VARIANT({variant_group(v)}, {', '.join(map(str, variant_key(v)))})"
            for v in variants
            if not (v.mutate or v.ablate)
        }
    )
    path.write_text(
        "// Decode-attention variants compiled into _rocm_C (see macros.h for\n"
        "// the fields).  Generated by write_variants_def from the backend's\n"
        "// tables; regenerate rather than edit:\n"
        "//   python -m vllm.v1.attention.ops.rdna35_hip_decode\n"
        + "".join(f"{line}\n" for line in lines)
    )


class _Builtin:
    """A variant compiled into _rocm_C, with the JIT module's interface."""

    def __init__(self, index: int):
        self.index = index

    def decode_attn(self, q, kv_cache, block_table, out, *scratch_and_args):
        torch.ops._rocm_C.rdna35_decode_attn(
            self.index, q, kv_cache, block_table, out, *scratch_and_args
        )


def _builtin_index(variant: KernelVariant) -> int:
    """The variant's index in _rocm_C, or -1 if it was not built in."""
    if variant.mutate or variant.ablate:
        return -1
    try:
        op = torch.ops._rocm_C.rdna35_decode_variant
    except (AttributeError, RuntimeError):
        return -1
    return int(op(variant_key(variant)))


def _hip_runtime_ldflags() -> list[str]:
    """Point the linker at libamdhip64.

    With the pip-installed ROCm SDK, torch sets ROCM_HOME to the venv root and
    so searches ``<venv>/lib``, but the runtime actually ships inside
    ``site-packages/_rocm_sdk_devel/lib``.
    """
    sdk_lib = Path(torch.__file__).resolve().parents[1] / "_rocm_sdk_devel" / "lib"
    return [f"-L{sdk_lib}"] if (sdk_lib / "libamdhip64.so").exists() else []


def _staged_source(name: str) -> Path:
    """Copy the kernel source into this variant's own build directory.

    On ROCm, torch hipifies each source before compiling it, and does so where
    the source lives: it writes ``rdna35_decode_attn.hip`` next to the ``.cu``
    and records the result in ``hipify_python.HIPIFY_FINAL_RESULT``, a global
    dict keyed by the absolute source path. Every variant compiles the same
    file, so concurrent builds collide on that one key: the later one resets
    the entry to a fresh record whose ``hipified_path`` is still the raw
    ``.cu``, and the earlier build reads that back and compiles unhipified
    CUDA, failing on ``cuda_runtime_api.h``. The generated ``.hip`` is shared
    too, and is unlinked by whichever build's ``GeneratedFileCleaner`` exits
    first, under the feet of the others. Giving each variant its own copy
    gives it its own hipify key, its own ``.hip`` and its own cleanup.

    The header it includes is staged beside it for the same reason: hipify
    writes a ``_hip`` copy of every header it follows next to that header, so
    one found in csrc/rocm would be rewritten in the source tree.
    """
    from torch.utils.cpp_extension import _get_build_directory

    build_dir = Path(_get_build_directory(name, False))
    for src, dst in (
        (_SOURCE, build_dir / _SOURCE.name),
        (_LAUNCH_H, build_dir / "rdna35_attn" / _LAUNCH_H.name),
    ):
        text = src.read_bytes()
        # Rewrite only on a real change: ninja keys off mtime, so touching
        # this every time would force a full rebuild on every warm start.
        if not dst.is_file() or dst.read_bytes() != text:
            dst.parent.mkdir(exist_ok=True)
            dst.write_bytes(text)
    return build_dir / _SOURCE.name


class UnexpectedBuildError(RuntimeError):
    """A variant was requested that precompilation did not cover."""


class VariantBuildError(RuntimeError):
    """A variant does not compile -- a static_assert refused its shape."""


# Variants that already failed to build.  A shape the kernel refuses fails the
# same way every time, and a rebuild costs ~20 s, so the first failure is
# remembered and every later request fails at once.
_failed: dict[KernelVariant, str] = {}


_sealed = False


def seal(on: bool = True) -> None:
    """Refuse to JIT-build any further variant.

    For benchmark harnesses. A build landing inside ``do_bench`` does not just
    add time: ``do_bench`` sizes its repeat count from a calibration run, so a
    build during calibration corrupts the median it reports -- measured here as
    104.59 us against a true 29.99, a 258% error, with no warning. Harnesses
    used to dodge that with a discarded warm-up call per cell, which doubled
    the run. Sealing after ``precompile`` turns the same hazard into a loud
    failure instead, and the warm-up call can go.

    Args:
        on: True to refuse builds, False to allow them again.
    """
    global _sealed
    _sealed = on


def load(variant: KernelVariant) -> Any:
    """The variant built into _rocm_C; with VLLM_RDNA35_ATTN_JIT, a JIT build.

    Raises:
        VariantBuildError: The variant is not built into _rocm_C (JIT off), or
            the JIT build failed: the kernel refuses the shape, or there is no
            source checkout or compiler.
        UnexpectedBuildError: The loader is sealed and the variant is not
            built yet.
    """
    import vllm.envs as envs

    if variant in _loaded:
        return _loaded[variant]
    if variant in _failed:
        raise VariantBuildError(_failed[variant])
    if not envs.VLLM_RDNA35_ATTN_JIT:
        index = _builtin_index(variant)
        if index < 0:
            _failed[variant] = f"{variant.name} is not built into _rocm_C"
            raise VariantBuildError(_failed[variant])
        _loaded[variant] = _Builtin(index)
        return _loaded[variant]
    if _sealed:
        raise UnexpectedBuildError(
            f"{variant.name} was not precompiled, and building it now would "
            "land inside the timed region. Precompile it or unseal."
        )

    if not _SOURCE.is_file():
        _failed[variant] = (
            f"{variant.name} is not built into _rocm_C, and there is no source "
            f"checkout to build it from ({_SOURCE})"
        )
        raise VariantBuildError(_failed[variant])

    from torch.utils.cpp_extension import load as load_extension

    source = _staged_source(variant.name)

    flags = [
        "-O3",
        "-DRDNA35_TORCH_EXT",
        # HIP resolves device functions by name across the whole process, so two
        # variants sharing the symbol `decode_attn` would both dispatch to
        # whichever registered first — returning plausible, wrong numbers. Give
        # each build its own symbols so variants can coexist.
        f"-Ddecode_attn=decode_attn_{variant.suffix}",
        f"-DHEAD_DIM={variant.head_size}",
        f"-DNUM_Q_HEADS={variant.num_q_heads}",
        f"-DNUM_KV_HEADS={variant.num_kv_heads}",
        f"-DMAXM={variant.max_m}",
        f"-DBS={variant.block_size}",
        f"-DLAYOUT={variant.layout}",
        f"-DNSEG={variant.nseg}",
        f"-DRG={variant.rg}",
        f"-DMINB={variant.minb}",
        f"-DNW={variant.nw}",
        *([f"-DDSPL={variant.dspl}"] if variant.dspl else []),
        f"-DRSPL={variant.rspl}",
        f"-DPF={variant.pf}",
        f"-DCPUB={variant.cpub}",
        f"-DVINLDS={variant.vinlds}",
        f"-DBATCH={variant.batch}",
        f"-DDOT={variant.dot}",
        f"-DBFLY={variant.bfly}",
        f"-DGT={variant.gt}",
        f"-DWIN={variant.window}",
        f"-DMUTATE={variant.mutate}",
        f"-DABLATE={variant.ablate}",
        f"-DKV_BF16={int(variant.dtype == torch.bfloat16)}",
    ]
    logger.info("Compiling %s", variant.name)
    try:
        module = load_extension(
            name=variant.name,
            sources=[str(source)],
            extra_cuda_cflags=flags,
            extra_ldflags=_hip_runtime_ldflags(),
        )
    # Not only a refused shape: a machine without the ROCm compiler or ninja
    # fails here too, and must fall back rather than fail the model.
    except Exception as exc:
        _failed[variant] = f"{variant.name} does not build: {exc}"
        raise VariantBuildError(_failed[variant]) from exc
    _loaded[variant] = module
    return module


# Measured peak RSS of one build: 1.15 GB, almost all of it torch's headers.
# Budgeted at 2 GB because the measurement is a single sample and the cost of
# being wrong is an OOM that takes the machine with it, not a slow build.
_BUILD_RSS_BYTES = 2 << 30


def _parallel_budget(n: int) -> int:
    """How many builds fit in memory, not how many cores exist.

    A build peaks at 1.15 GB, so 32 of them want more than this machine has --
    running one per core filled 30 GB of RAM and had to be killed.  Memory is
    the binding constraint here, and it is read at call time because what is
    free depends on what else is running.
    """
    free = _available_bytes()
    by_mem = max(1, int(free * 0.7) // _BUILD_RSS_BYTES)
    return max(1, min(n, os.cpu_count() or 8, by_mem))


def _available_bytes() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 << 30


def precompile(variants: "Iterable[KernelVariant]", workers: int | None = None) -> None:
    """Build several variants at once.

    Each variant is already its own translation unit, its own ninja invocation
    and -- since ``_staged_source`` gives it a private copy of the kernel --
    its own hipify output, so concurrent builds do not collide.

    This matters more than it looks. A cold build is ~19.6 s, of which the
    kernel itself is 0.57 s -- the rest is torch/extension.h and the link. So
    the cost is per-variant and fixed, and the 27 distinct variants of the
    shape table cost nine minutes serially before a single measurement can be
    taken. Spread over the machine that is well under a minute.
    """
    # Deduplicated: two threads building the same variant would race on the
    # same build directory and the same _loaded entry.  Callers hand over
    # whatever their loop produced -- roofline builds both layouts per shape
    # and shapes repeat -- so duplicates are the normal case, not a mistake.
    todo = list(dict.fromkeys(v for v in variants if v not in _loaded))
    if not todo:
        return
    n = workers or _parallel_budget(len(todo))
    logger.info(
        "Compiling %d kernel variants across %d workers (%.1f GiB available)",
        len(todo),
        n,
        _available_bytes() / (1 << 30),
    )
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(load, v): v for v in todo}
        for fut, variant in futures.items():
            try:
                fut.result()
            except Exception:
                # A shape the kernel refuses to build is a normal outcome --
                # D=96 is three fp16 per lane and static_asserts. This is a
                # warm-up, not a gate: let the real load report it where the
                # caller already handles a fallback.
                logger.debug(
                    "%s did not build; leaving it to the caller",
                    variant.name,
                    exc_info=True,
                )


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
        # Arrival counters, then merge generations, one each per (kv head,
        # row group).  Zeroed once: the kernel resets the counters as it
        # consumes them and only compares generations for change, so every
        # later launch starts clean without the host writing here -- which it
        # could not do under graph capture anyway.
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
    page = variant.block_size * variant.num_kv_heads * kv_row
    if variant.layout == 0:  # NHD: (NB, BS, HKV, 2D)
        return page, kv_row, variant.num_kv_heads * kv_row
    return page, variant.block_size * kv_row, kv_row  # HND: (NB, HKV, BS, 2D)


# Query tokens per sequence the kernel serves: decode and speculative decode.
MAX_M = 8

# The kernel's V and K slices are one b64 or b128 per lane, which the head
# dims below allow; its GQA packing assumes the q heads divide evenly over the
# kv heads.
SUPPORTED_HEAD_SIZES = (64, 128, 256, 512)

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
# The heuristics in knobs_for are rules fitted to the whole shape table; this
# is the exceptions list, and it wins where both apply.  Rows carry only the
# knobs actually measured -- a partial row is normal, and a configuration
# absent here behaves exactly as the heuristics say, so landing a row can only
# affect the configuration it names.
#
# Every row comes from a measurement (tools/tune.py); do not add one without
# it.
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
    (14, 2, 64, 1): {"nw": 4, "rg": 1, "minb": 2, "nseg": 4, "pf": 1, "cpub": 1},
    (14, 2, 64, 3): {"nw": 4, "rg": 1, "minb": 2, "nseg": 8, "rspl": 2},
    (14, 2, 64, 4): {"nseg": 4, "rg": 1, "minb": 2, "nw": 4, "vinlds": 1},
    (16, 2, 64, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (16, 2, 64, 4): {"nseg": 4, "rg": 2, "minb": 2, "nw": 4, "vinlds": 1},
    (32, 8, 64, 1): {"nw": 4, "rg": 1, "minb": 2, "nseg": 1, "pf": 1},
    (32, 8, 64, 2): {"nw": 8, "rg": 1, "minb": 1, "nseg": 1, "pf": 1},
    (32, 8, 64, 3): {"nw": 8, "rg": 2, "minb": 2, "nseg": 1},
    (32, 8, 64, 4): {"nw": 4, "rg": 1, "minb": 1, "nseg": 1, "pf": 1},
    (32, 32, 64, 1): {"nseg": 1, "rg": 1, "minb": 2, "nw": 2, "dspl": 0},
    (32, 32, 64, 2): {"nw": 2, "rg": 1, "minb": 1, "nseg": 1},
    (32, 32, 64, 3): {"nw": 2, "rg": 1, "minb": 1, "nseg": 1, "cpub": 1},
    (32, 32, 64, 4): {"nseg": 1, "rg": 1, "minb": 4, "nw": 2, "dspl": 0},
    # D=128
    (16, 2, 128, 1): {"nseg": 4, "rg": 2, "minb": 2, "nw": 4, "vinlds": 1},
    (16, 2, 128, 2): {"nw": 8, "rg": 1, "minb": 1, "nseg": 8},
    (16, 2, 128, 4): {"nw": 4, "rg": 4, "minb": 2, "nseg": 2, "vinlds": 1},
    (32, 2, 128, 1): {"nw": 4, "rg": 4, "minb": 2, "nseg": 2, "cpub": 1, "vinlds": 1},
    (32, 2, 128, 4): {"nseg": 4, "rg": 4, "minb": 2, "nw": 4, "vinlds": 1},
    (28, 4, 128, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 4, "dspl": 2},
    (28, 4, 128, 2): {"nw": 8, "rg": 1, "minb": 4, "nseg": 4, "cpub": 1},
    (28, 4, 128, 3): {
        "nw": 4,
        "rg": 1,
        "minb": 2,
        "nseg": 8,
        "dspl": 2,
        "rspl": 2,
        "pf": 1,
    },
    (28, 4, 128, 4): {"nseg": 4, "rg": 1, "minb": 1, "nw": 8, "dspl": 2},
    (32, 4, 128, 1): {"nw": 4, "rg": 2, "minb": 4, "nseg": 2, "vinlds": 1},
    (32, 4, 128, 2): {"nw": 8, "rg": 2, "minb": 4, "nseg": 4},
    (32, 4, 128, 4): {"nw": 4, "rg": 4, "minb": 1, "nseg": 1, "vinlds": 1},
    (16, 8, 128, 1): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (16, 8, 128, 4): {"nw": 4, "rg": 2, "minb": 4, "nseg": 1},
    (24, 8, 128, 1): {"nw": 4, "rg": 1, "minb": 1, "nseg": 1, "cpub": 1, "vinlds": 1},
    (24, 8, 128, 2): {"nw": 8, "rg": 1, "minb": 4, "nseg": 2},
    (24, 8, 128, 3): {"nw": 4, "rg": 1, "minb": 2, "nseg": 1, "cpub": 1, "vinlds": 1},
    (24, 8, 128, 4): {"nseg": 1, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (32, 8, 128, 1): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (32, 8, 128, 3): {"nw": 8, "rg": 2, "minb": 2, "nseg": 1},
    (32, 8, 128, 4): {"nseg": 1, "rg": 2, "minb": 2, "nw": 4, "dspl": 0},
    (40, 8, 128, 1): {"nseg": 1, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (40, 8, 128, 4): {"nw": 8, "rg": 1, "minb": 4, "nseg": 2, "dspl": 2, "rspl": 2},
    (10, 10, 128, 1): {"nseg": 1, "rg": 1, "minb": 2, "nw": 4, "vinlds": 1},
    (10, 10, 128, 2): {"nw": 4, "rg": 1, "minb": 4, "nseg": 1, "vinlds": 1},
    (10, 10, 128, 3): {"nw": 8, "rg": 1, "minb": 2, "nseg": 1, "cpub": 1},
    (10, 10, 128, 4): {"nw": 2, "rg": 1, "minb": 2, "nseg": 1, "vinlds": 1},
    (32, 32, 128, 1): {"nseg": 1, "rg": 1, "minb": 2, "nw": 2, "dspl": 0},
    (32, 32, 128, 4): {"nseg": 1, "rg": 1, "minb": 1, "nw": 2, "dspl": 0},
    # D=256
    (8, 1, 256, 1): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4},
    (8, 1, 256, 2): {"nw": 2, "rg": 1, "minb": 1, "nseg": 16},
    (8, 1, 256, 4): {"nseg": 16, "rg": 2, "minb": 1, "nw": 4, "dspl": 2},
    (8, 2, 256, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4},
    (8, 2, 256, 2): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4, "gt": 1},
    (8, 2, 256, 4): {"nseg": 4, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (16, 2, 256, 1): {"nseg": 8, "rg": 2, "minb": 2, "nw": 8, "dspl": 0},
    (16, 2, 256, 4): {"nseg": 8, "rg": 2, "minb": 1, "nw": 4, "dspl": 2},
    (8, 4, 256, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4, "gt": 1},
    (8, 4, 256, 2): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4, "gt": 1},
    (8, 4, 256, 4): {
        "nw": 4,
        "rg": 2,
        "minb": 2,
        "nseg": 2,
        "dspl": 4,
        "pf": 1,
        "cpub": 1,
    },
    (16, 4, 256, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 2, "gt": 1},
    (16, 4, 256, 2): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 2, "gt": 1},
    (16, 4, 256, 4): {"nw": 4, "rg": 2, "minb": 1, "nseg": 2},
    (24, 4, 256, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 2},
    (24, 4, 256, 2): {"dot": 1, "nw": 4, "nseg": 4, "bfly": 2, "gt": 1},
    (24, 4, 256, 3): {"nw": 4, "rg": 1, "minb": 2, "nseg": 4, "rspl": 2},
    (24, 4, 256, 4): {
        "nw": 4,
        "rg": 1,
        "minb": 1,
        "nseg": 4,
        "rspl": 2,
        "pf": 1,
        "cpub": 1,
    },
    (16, 8, 256, 1): {"dot": 1, "nw": 4, "nseg": 2, "gt": 1},
    (16, 8, 256, 2): {"nw": 2, "rg": 1, "minb": 4, "nseg": 2, "vinlds": 1},
    (16, 8, 256, 3): {"nw": 2, "rg": 1, "minb": 2, "nseg": 2, "vinlds": 1},
    (16, 8, 256, 4): {"nseg": 2, "rg": 1, "minb": 4, "nw": 4, "dspl": 4},
    # D=512
    (8, 1, 512, 1): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4},
    (8, 1, 512, 2): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4},
    (8, 1, 512, 4): {"nseg": 8, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (16, 1, 512, 1): {"nw": 8, "rg": 1, "minb": 1, "nseg": 16, "dspl": 8},
    (16, 1, 512, 3): {"nw": 8, "rg": 4, "minb": 1, "nseg": 8},
    (16, 1, 512, 4): {"nseg": 8, "rg": 4, "minb": 1, "nw": 8, "dspl": 0},
    (8, 2, 512, 1): {"dot": 1, "nw": 8, "nseg": 2},
    (8, 2, 512, 2): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4},
    (8, 2, 512, 3): {"nw": 4, "rg": 1, "minb": 2, "nseg": 4, "cpub": 1, "vinlds": 1},
    (8, 2, 512, 4): {"nw": 8, "rg": 2, "minb": 1, "nseg": 4, "dspl": 8, "pf": 1},
    (16, 2, 512, 1): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4},
    (16, 2, 512, 2): {"nw": 4, "rg": 2, "minb": 1, "nseg": 4, "vinlds": 1},
    (16, 2, 512, 3): {"nw": 8, "rg": 2, "minb": 1, "nseg": 4, "cpub": 1},
    (16, 2, 512, 4): {"nseg": 4, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (32, 4, 512, 1): {"dot": 1, "nw": 4, "nseg": 2, "bfly": 4},
    (32, 4, 512, 2): {"dot": 1, "nw": 4, "nseg": 2},
    (32, 4, 512, 3): {"nw": 8, "rg": 2, "minb": 2, "nseg": 2, "cpub": 1},
    (32, 4, 512, 4): {"nseg": 2, "rg": 2, "minb": 2, "nw": 8, "dspl": 0},
}

# bf16 rows that differ from _TUNED.  The dot decomposition's bf16 products
# (v_dot2_f32_bf16) fall behind at long context -- 0.73-0.97x of the WMMA
# rows at 32k -- so in bf16 these shapes keep their WMMA configuration.
_TUNED_BF16: dict[tuple[int, int, int, int], _Knobs] = {
    (8, 1, 256, 1): {"nseg": 8, "rg": 2, "minb": 1, "nw": 8, "dspl": 0},
    (8, 2, 256, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 4},
    (8, 2, 256, 2): {"rg": 1, "nseg": 8, "nw": 8, "minb": 1},
    (8, 4, 256, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 2, "dspl": 0},
    (8, 4, 256, 2): {"rg": 1, "nseg": 4, "nw": 8, "minb": 1},
    (16, 4, 256, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 2, "dspl": 0},
    (16, 4, 256, 2): {"rg": 1, "nseg": 4, "nw": 8, "minb": 1},
    (24, 4, 256, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 8, "dspl": 0},
    (24, 4, 256, 2): {"rg": 1, "nseg": 4, "nw": 8, "minb": 1},
    (16, 8, 256, 1): {"rg": 1, "nseg": 2, "nw": 8, "minb": 2},
    (8, 1, 512, 1): {"nseg": 16, "rg": 1, "minb": 1, "nw": 8, "dspl": 8},
    (8, 1, 512, 2): {"rg": 1, "nseg": 16, "nw": 8, "minb": 1},
    (8, 2, 512, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (8, 2, 512, 2): {"rg": 1, "nseg": 8, "nw": 8, "minb": 1},
    (16, 2, 512, 1): {"nseg": 8, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (32, 4, 512, 1): {"nseg": 4, "rg": 1, "minb": 1, "nw": 4, "dspl": 0},
    (32, 4, 512, 2): {"rg": 1, "nseg": 4, "nw": 8, "minb": 1},
}

# Sliding-window layers, keyed on (Hq, Hkv, D, M, window).  They read only the
# window's keys whatever the sequence length, so they are tuned apart from the
# full-attention layers that share their shape (gemma interleaves the two).
_TUNED_SWA: dict[tuple[int, int, int, int, int], _Knobs] = {
    (8, 1, 256, 1, 512): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4, "gt": 1},
    (8, 1, 256, 2, 512): {"dot": 1, "nw": 8, "nseg": 8, "bfly": 2},
    (8, 1, 256, 3, 512): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4},
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
    (8, 2, 256, 2, 512): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4, "gt": 1},
    (8, 2, 256, 3, 512): {"nw": 4, "rg": 1, "minb": 1, "nseg": 8, "dspl": 4, "pf": 1},
    (8, 2, 256, 4, 512): {"nw": 4, "rg": 1, "minb": 1, "nseg": 8, "dspl": 4, "pf": 1},
    (8, 4, 256, 1, 1024): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4, "gt": 1},
    (8, 4, 256, 2, 1024): {"dot": 1, "nw": 8, "nseg": 4, "bfly": 4, "gt": 1},
    (8, 4, 256, 3, 1024): {"nw": 4, "rg": 1, "minb": 2, "nseg": 2, "dspl": 4, "pf": 1},
    (8, 4, 256, 4, 1024): {
        "nw": 4,
        "rg": 1,
        "minb": 4,
        "nseg": 2,
        "dspl": 4,
        "pf": 1,
        "cpub": 1,
    },
    (8, 4, 256, 1, 4096): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4, "gt": 1},
    (8, 4, 256, 2, 4096): {"nw": 4, "rg": 2, "minb": 1, "nseg": 2},
    (8, 4, 256, 4, 4096): {
        "nw": 4,
        "rg": 1,
        "minb": 1,
        "nseg": 2,
        "dspl": 2,
        "vinlds": 1,
    },
    (16, 8, 256, 1, 1024): {"dot": 1, "nw": 4, "nseg": 1, "bfly": 4, "gt": 1},
    (16, 8, 256, 2, 1024): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4, "gt": 1},
    (16, 8, 256, 3, 1024): {"dot": 1, "nw": 8, "nseg": 2, "bfly": 4, "gt": 1},
    (16, 8, 256, 4, 1024): {"nw": 4, "rg": 2, "minb": 2, "nseg": 1},
    (32, 16, 256, 1, 1024): {"dot": 1, "nw": 4, "nseg": 1, "bfly": 4, "gt": 1},
    (32, 16, 256, 2, 1024): {"nw": 4, "rg": 1, "minb": 2, "nseg": 1, "dspl": 4},
    (32, 16, 256, 3, 1024): {"nw": 2, "rg": 1, "minb": 4, "nseg": 1, "vinlds": 1},
    (32, 16, 256, 4, 1024): {"nw": 2, "rg": 1, "minb": 2, "nseg": 1, "vinlds": 1},
}


def knobs_for(
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


def variant_for(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_m: int,
    block_size: int,
    layout: int,
    window: int,
    dtype: torch.dtype,
    batch: bool,
) -> KernelVariant:
    """The kernel build that serves one call."""
    return KernelVariant(
        **knobs_for(num_q_heads, num_kv_heads, head_size, max_m, window, dtype, batch),
        head_size=head_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        max_m=max_m,
        block_size=block_size,
        layout=layout,
        dtype=dtype,
        window=window,
        batch=int(batch),
    )


# Page sizes _rocm_C carries kernels for, 16 everywhere plus what vLLM gives
# these layers when it evens out page sizes across a model's layers -- two
# head sizes in gemma-4, the GDN state in Qwen3.5/3.6 -- measured per model.
# Any other page size is served by Triton, which at these sizes is 2.5-6.5x
# slower than the kernel on the bs=16 knobs.
_BUILT_BLOCK_SIZES: dict[tuple[int, int, int, int], tuple[int, ...]] = {
    (8, 1, 256, 512): (16, 32),  # gemma-4-E2B sliding layers
    (8, 2, 256, 512): (16, 32),  # gemma-4-E4B sliding layers
    (16, 2, 512, 0): (16, 32),  # gemma-4-26B-A4B full layers
    (32, 4, 512, 0): (16, 32),  # gemma-4-31B full layers
    (8, 2, 256, 0): (16, 544),  # Qwen3.5-0.8B, -2B
    (16, 4, 256, 0): (16, 528),  # Qwen3.5-9B
    (16, 2, 256, 0): (16, 1056),  # Qwen3.5-35B-A3B, Qwen3.6-35B-A3B
    (24, 4, 256, 0): (16, 784),  # Qwen3.6-27B
}


def built_variants() -> list[KernelVariant]:
    """Every variant built into _rocm_C (variants.def): each configuration
    of the tables at every query length the kernel serves, in both dtypes, for
    one sequence and for a batch, on the HND cache the backend requires."""
    configs = {(*k[:3], 0) for k in _TUNED} | {(*k[:3], k[4]) for k in _TUNED_SWA}
    return [
        variant_for(hq, hkv, d, m, bs, 1, window, dtype, batch)
        for hq, hkv, d, window in sorted(configs)
        for m in range(1, MAX_M + 1)
        for dtype in (torch.float16, torch.bfloat16)
        for batch in (False, True)
        for bs in _BUILT_BLOCK_SIZES.get((hq, hkv, d, window), (16,))
    ]


if __name__ == "__main__":
    write_variants_def(built_variants(), VARIANTS_DEF)
    print(f"wrote {VARIANTS_DEF}")
