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

# Mirrors the kernel's own defines; NWAVE sizes the per-wave partials.
_WAVE = 32
_BLOCK = 256
_NWAVE = _BLOCK // _WAVE

# resolve() matters: ninja re-expands the path through a shell, so a symlinked
# checkout whose name contains '$' would be mangled into a missing file.
_CSRC = Path(__file__).resolve().parents[4] / "csrc" / "rocm"
_SOURCE = _CSRC / "rdna35_decode_attn.cu"


@dataclass(frozen=True)
class KernelVariant:
    """The compile-time shape tuple a single build is specialised for."""

    head_size: int
    num_q_heads: int
    num_kv_heads: int
    max_m: int
    block_size: int
    layout: int  # 0 = NHD, 1 = HND
    nseg: int = 1
    kpw: int = 4
    mutate: int = 0
    # Threads per workgroup; block // 32 waves cooperate on one head-segment.
    block: int = _BLOCK
    # How many waves share the query-token dimension. 1 gives every wave all
    # max_m tokens over its own KV slice; max_m gives every wave one token and
    # a slice shared with its neighbours.
    msplit: int = 1
    # Finish the cross-workgroup reduction inside decode_attn instead of
    # launching reduce_segments for it.  On by default: worth ~1.2 us flat, and
    # a no-op at nseg == 1 where no second kernel runs anyway.
    fusedred: bool = True

    def __post_init__(self) -> None:
        # Raise msplit until the partials fit LDS.  Each of the
        # (max_m / msplit) * waves rows costs head_size floats plus the m and l
        # scalars, so max_m = 8 at head_size = 256 does not fit at msplit = 1.
        # Resolved here rather than in the kernel so the variant name, and
        # therefore the compiled symbol, matches what is actually built.
        waves = self.block // _WAVE
        row = self.head_size * 4 + 8
        ms = self.msplit
        while (
            ms < self.max_m
            and (self.max_m // ms) * waves * row > 65536
            and self.max_m % (ms * 2) == 0
            and waves % (ms * 2) == 0
        ):
            ms *= 2
        object.__setattr__(self, "msplit", ms)

    # Merge the per-wave partials in LDS inside the main kernel rather than in
    # a second pass over global memory. Only valid with nseg == 1.
    fused: bool = True

    @property
    def suffix(self) -> str:
        return (
            f"d{self.head_size}_q{self.num_q_heads}_kv{self.num_kv_heads}"
            f"_m{self.max_m}_bs{self.block_size}_l{self.layout}"
            f"_n{self.nseg}_k{self.kpw}_mut{self.mutate}"
            f"_b{self.block}_ms{self.msplit}{'' if self.fusedred else '_nofr'}"
            f"_f{int(self.fused)}"
        )

    @property
    def name(self) -> str:
        return f"rdna35_decode_{self.suffix}"

    @property
    def partials_per_head(self) -> int:
        """How many partials the global reduction merges.

        The fused epilogue collapses a workgroup's NWAVE partials in LDS, so
        only the NSEG cross-workgroup ones survive.
        """
        waves = self.block // _WAVE
        return self.nseg if self.fused else self.nseg * waves

    def scratch_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Shapes of the (acc, m/l) partials the kernel writes."""
        segments = self.num_q_heads * self.partials_per_head * self.max_m
        return (segments, self.head_size), (segments,)


_loaded: dict[KernelVariant, Any] = {}


def _hip_runtime_ldflags() -> list[str]:
    """Point the linker at libamdhip64.

    With the pip-installed ROCm SDK, torch sets ROCM_HOME to the venv root and
    so searches ``<venv>/lib``, but the runtime actually ships inside
    ``site-packages/_rocm_sdk_devel/lib``.
    """
    sdk_lib = Path(torch.__file__).resolve().parents[1] / "_rocm_sdk_devel" / "lib"
    return [f"-L{sdk_lib}"] if (sdk_lib / "libamdhip64.so").exists() else []


def load(variant: KernelVariant) -> Any:
    """Compile (or fetch) the extension for ``variant``."""
    if variant in _loaded:
        return _loaded[variant]

    source = _SOURCE
    if not source.is_file():
        raise RuntimeError(
            f"kernel source not found at {source}; this loader only works "
            "from a source checkout, not an installed wheel"
        )

    from torch.utils.cpp_extension import load as load_extension

    flags = [
        "-O3",
        "-DDECODE_ATTN_NO_MAIN",
        "-DRDNA35_TORCH_EXT",
        # HIP resolves device functions by name across the whole process, so two
        # variants sharing the symbol `decode_attn` would both dispatch to
        # whichever registered first — returning plausible, wrong numbers. Give
        # each build its own symbols so variants can coexist.
        f"-Ddecode_attn=decode_attn_{variant.suffix}",
        f"-Dreduce_segments=reduce_segments_{variant.suffix}",
        f"-DHEAD_DIM={variant.head_size}",
        f"-DNUM_Q_HEADS={variant.num_q_heads}",
        f"-DNUM_KV_HEADS={variant.num_kv_heads}",
        f"-DMAXM={variant.max_m}",
        f"-DBS={variant.block_size}",
        f"-DLAYOUT={variant.layout}",
        f"-DNSEG={variant.nseg}",
        f"-DKPW={variant.kpw}",
        f"-DMUTATE={variant.mutate}",
        f"-DFUSED={int(variant.fused)}",
        f"-DBLOCK={variant.block}",
        f"-DMSPLIT={variant.msplit}",
        f"-DFUSEDRED={int(variant.fusedred)}",
    ]
    logger.info("Compiling %s", variant.name)
    module = load_extension(
        name=variant.name,
        sources=[str(source)],
        extra_cuda_cflags=flags,
        extra_ldflags=_hip_runtime_ldflags(),
    )
    _loaded[variant] = module
    return module


def precompile(variants: "Iterable[KernelVariant]", workers: int | None = None) -> None:
    """Build several variants at once.

    Each variant is already its own translation unit and its own ninja
    invocation, so they are independent; torch takes a file lock per extension
    name, which differs per variant, so concurrent builds do not collide.

    This matters more than it looks. A cold build is ~19.6 s, of which the
    kernel itself is 0.57 s -- the rest is torch/extension.h and the link. So
    the cost is per-variant and fixed, and the 27 distinct variants of the
    shape table cost nine minutes serially before a single measurement can be
    taken. Spread over the machine that is well under a minute.
    """
    todo = [v for v in variants if v not in _loaded]
    if not todo:
        return
    n = workers or min(len(todo), (os.cpu_count() or 8))
    logger.info("Compiling %d kernel variants across %d workers", len(todo), n)
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
                logger.debug("%s did not build; leaving it to the caller", variant.name)


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
        # Arrival counters, one per q head.  Zeroed once: the kernel resets
        # them as it consumes them, so every later launch starts clean without
        # the host writing here -- which it could not do under graph capture
        # anyway.
        torch.zeros(variant.num_q_heads, dtype=torch.int32, device=device),
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
