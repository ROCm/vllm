# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT loader for the RDNA3.5 paged decode-attention kernel.

The kernel in ``csrc/rocm/rdna35_decode_attn.cu`` takes its shapes as
compile-time defines, so one build serves exactly one shape tuple.  It is
compiled here with ``torch.utils.cpp_extension.load`` rather than through
CMake, which keeps kernel iteration at a few seconds instead of a full vLLM
rebuild.  It is consequently a development and benchmarking path, not part of
a shipped wheel.
"""

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# resolve() matters: ninja re-expands the path through a shell, so a symlinked
# checkout whose name contains '$' would be mangled into a missing file.
_CSRC = Path(__file__).resolve().parents[4] / "csrc" / "rocm"
_SOURCE = _CSRC / "rdna35_decode_attn.cu"


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
    mutate: int = 0
    # Measurement only: skips blocks of work and returns wrong numbers.  See
    # the ABLATE comment in the kernel.
    ablate: int = 0

    @property
    def suffix(self) -> str:
        return (
            f"d{self.head_size}_q{self.num_q_heads}_kv{self.num_kv_heads}"
            f"_m{self.max_m}_bs{self.block_size}_l{self.layout}"
            f"_n{self.nseg}_rg{self.rg}_mb{self.minb}_w{self.nw}"
            f"{'' if not self.dspl else f'_ds{self.dspl}'}"
            f"_mut{self.mutate}"
            f"{'' if not self.ablate else f'_ab{self.ablate}'}"
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

    def scratch_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Shapes of the (acc, m/l) partials the split-KV merge goes through."""
        rows = self.num_kv_heads * self.rg * self.nseg * self.rows_padded
        return (rows, self.head_size), (rows,)


_loaded: dict[KernelVariant, Any] = {}


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
    """
    from torch.utils.cpp_extension import _get_build_directory

    staged = Path(_get_build_directory(name, False)) / _SOURCE.name
    text = _SOURCE.read_bytes()
    # Rewrite only on a real change: ninja keys off mtime, so touching this
    # every time would force a full rebuild on every warm start.
    if not staged.is_file() or staged.read_bytes() != text:
        staged.write_bytes(text)
    return staged


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
    """Compile (or fetch) the extension for ``variant``.

    Raises:
        VariantBuildError: The kernel refuses this shape.
        UnexpectedBuildError: The loader is sealed and the variant is not
            built yet.
    """
    if variant in _loaded:
        return _loaded[variant]
    if variant in _failed:
        raise VariantBuildError(_failed[variant])
    if _sealed:
        raise UnexpectedBuildError(
            f"{variant.name} was not precompiled, and building it now would "
            "land inside the timed region. Precompile it or unseal."
        )

    if not _SOURCE.is_file():
        raise RuntimeError(
            f"kernel source not found at {_SOURCE}; this loader only works "
            "from a source checkout, not an installed wheel"
        )

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
        f"-DMUTATE={variant.mutate}",
        f"-DABLATE={variant.ablate}",
    ]
    logger.info("Compiling %s", variant.name)
    try:
        module = load_extension(
            name=variant.name,
            sources=[str(source)],
            extra_cuda_cflags=flags,
            extra_ldflags=_hip_runtime_ldflags(),
        )
    except RuntimeError as exc:
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
    variant: KernelVariant, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the partials once.

    They are passed into the op rather than allocated inside it because the op
    runs under CUDA-graph capture, where an allocation would break the graph.
    """
    acc_shape, ml_shape = variant.scratch_shapes()
    opts = {"dtype": torch.float32, "device": device}
    return (
        torch.empty(acc_shape, **opts),
        torch.empty(ml_shape, **opts),
        torch.empty(ml_shape, **opts),
        # Arrival counters, then merge generations, one each per (kv head,
        # row group).  Zeroed once: the kernel resets the counters as it
        # consumes them and only compares generations for change, so every
        # later launch starts clean without the host writing here -- which it
        # could not do under graph capture anyway.
        torch.zeros(
            2 * variant.num_kv_heads * variant.rg, dtype=torch.int32, device=device
        ),
    )


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
