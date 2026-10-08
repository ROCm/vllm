# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builds of the RDNA3.5 decode kernel outside _rocm_C, and hooks into ROCM_ATTN.

_rocm_C carries the rows of rdna35_variants.csv.  Tuning measures knob sets
that are not rows, so here a variant is the kernel source compiled by itself
with clang (no torch headers, ~1 s) into a shared object exporting
`rdna35_launch(const rdna35::LaunchArgs*)` -- the launch() _rocm_C's registry
calls -- and launched through ctypes.  Builds are cached by a hash of the
sources and the command.

`install()` makes ROCM_ATTN run these builds, with knobs from `override()` on
top of the CSV row (or of the kernel's defaults where there is no row).
`paths()` reports which path the backend took.
"""

import contextlib
import ctypes
import dataclasses
import functools
import hashlib
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import vllm.v1.attention.ops.rdna35_hip_decode as R

_CSRC = Path(__file__).resolve().parents[4] / "csrc" / "rocm"
_SOURCES = (_CSRC / "rdna35_decode_attn.cu", _CSRC / "rdna35_decode_attn.h")
_ARCH = "gfx1151"

_WRAPPER = """#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "rdna35_decode_attn.cu"
extern "C" __attribute__((visibility("default"))) void rdna35_launch(
    const rdna35::LaunchArgs* a) {
  launch(*a);
}
"""


class _LaunchArgs(ctypes.Structure):
    """rdna35::LaunchArgs (csrc/rocm/rdna35_decode_attn.h), field for field."""

    _fields_ = [
        ("q", ctypes.c_void_p),
        ("kv", ctypes.c_void_p),
        ("bt", ctypes.c_void_p),
        ("acc", ctypes.c_void_p),
        ("m", ctypes.c_void_p),
        ("l", ctypes.c_void_p),
        ("cnt", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("seq_lens", ctypes.c_void_p),
        ("bt_width", ctypes.c_int),
        ("bt_stride", ctypes.c_int),
        ("acc_stride", ctypes.c_int),
        ("ml_stride", ctypes.c_int),
        ("cnt_stride", ctypes.c_int),
        ("nseq", ctypes.c_int),
        ("scale", ctypes.c_float),
        ("stream", ctypes.c_void_p),
    ]


_cache: Path | None = None
_ablate = 0
_sealed = False
_overrides: dict = {}
_loaded: dict = {}
_failed: dict[R.KernelVariant, str] = {}
_lock = threading.Lock()


def source_digest() -> str:
    """The kernel source a build compiles; points of another digest measured
    another kernel."""
    h = hashlib.sha256()
    for p in _SOURCES:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@functools.cache
def _toolchain() -> tuple[list[str], str]:
    """(clang command up to the sources, rpath of the HIP runtime).

    The pip ROCm SDK ships no unversioned libamdhip64.so, which the HIP link
    step expects, so it gets a --rocm-path of its own: its tree plus that link.
    """
    sdk = Path(torch.__file__).resolve().parents[1] / "_rocm_sdk_core"
    if sdk.is_dir():
        shim = _cache / "rocm"
        if not (shim / "lib" / "libamdhip64.so").exists():
            tmp = _cache / f"rocm.{os.getpid()}"
            (tmp / "lib").mkdir(parents=True)
            for name in ("bin", "include", "share", "libexec", "etc"):
                if (sdk / name).exists():
                    (tmp / name).symlink_to(sdk / name)
            for entry in (sdk / "lib").iterdir():
                (tmp / "lib" / entry.name).symlink_to(entry)
            hip = next((sdk / "lib").glob("libamdhip64.so.*"))
            (tmp / "lib" / "libamdhip64.so").symlink_to(hip)
            try:
                os.rename(tmp, shim)
            except OSError:  # another process made it first
                shutil.rmtree(tmp)
        bitcode = sdk / "lib" / "llvm" / "amdgcn" / "bitcode"
        return [
            str(sdk / "lib" / "llvm" / "bin" / "clang++"),
            f"--rocm-path={shim}",
            f"--rocm-device-lib-path={bitcode}",
        ], str(sdk / "lib")
    from torch.utils.cpp_extension import ROCM_HOME

    if ROCM_HOME is None:
        raise RuntimeError("no ROCm found: neither the pip SDK nor ROCM_HOME")
    rocm = Path(ROCM_HOME)
    return [str(rocm / "lib" / "llvm" / "bin" / "clang++"), f"--rocm-path={rocm}"], (
        str(rocm / "lib")
    )


def _build(v: R.KernelVariant) -> Path:
    """Compile one variant, or find it built: named by a hash of the sources,
    the command and the compiler, written under a temporary name and renamed."""
    tag = f"{v.name}_l{v.layout}_ab{_ablate}"
    clang, rpath = _toolchain()
    cmd = [
        *clang,
        "-x",
        "hip",
        f"--offload-arch={_ARCH}",
        "-fno-gpu-rdc",
        "-O3",
        "-std=c++20",
        "-fPIC",
        "-shared",
        "-Wno-unused-variable",
        # HIP resolves kernels by name across the process: every build loaded
        # side by side needs its own.
        f"-Ddecode_attn=decode_attn_{hashlib.sha256(tag.encode()).hexdigest()[:16]}",
        *(f"-D{c}={x}" for c, x in R.variant_defines(v).items()),
        f"-DABLATE={_ablate}",
        f"-I{_CSRC}",
        f"-Wl,-rpath,{rpath}",
        str(_cache / "wrapper.hip"),
    ]
    h = hashlib.sha256((source_digest() + " ".join(cmd)).encode()).hexdigest()
    so = _cache / f"{tag}-{h[:16]}.so"
    if so.is_file():
        return so
    tmp = so.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    proc = subprocess.run([*cmd, "-o", str(tmp)], capture_output=True, text=True)
    if proc.returncode:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(proc.stderr.strip()[-2000:] or f"clang {proc.returncode}")
    os.replace(tmp, so)
    return so


class Jit:
    """One build, with the decode_attn interface of _rocm_C's variants."""

    def __init__(self, v: R.KernelVariant, path: Path):
        self.variant = v
        self._launch = ctypes.CDLL(str(path)).rdna35_launch
        self._launch.argtypes = [ctypes.POINTER(_LaunchArgs)]
        self._launch.restype = None

    def decode_attn(
        self, q, kv_cache, block_table, out, acc, m, ls, cnt, seq_lens, scale
    ):
        v = self.variant
        if not (q.is_contiguous() and out.is_contiguous()):
            raise RuntimeError("q and out must be contiguous")
        if not q.dtype == kv_cache.dtype == out.dtype == v.dtype:
            raise RuntimeError("dtype does not match the build")
        nseq = q.size(0) // v.max_query_len
        if nseq * v.max_query_len != q.size(0) or (not v.batched and nseq != 1):
            raise RuntimeError(f"q has {q.size(0)} tokens for {v.name}")
        batched = acc.dim() == 3
        args = _LaunchArgs(
            q.data_ptr(),
            kv_cache.data_ptr(),
            block_table.data_ptr(),
            acc.data_ptr(),
            m.data_ptr(),
            ls.data_ptr(),
            cnt.data_ptr(),
            out.data_ptr(),
            seq_lens.data_ptr(),
            block_table.size(-1),
            block_table.stride(0) if block_table.dim() == 2 else 0,
            acc.stride(0) if batched else 0,
            m.stride(0) if batched else 0,
            cnt.stride(0) if batched else 0,
            nseq,
            scale,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
        self._launch(ctypes.byref(args))


def load(v: R.KernelVariant) -> Jit | None:
    """The variant's build, or None if it does not build (`failure` says why).

    Raises:
        RuntimeError: Sealed, and the variant was not built before.

    """
    if v in _loaded:
        return _loaded[v]
    if v in _failed:
        return None
    if _sealed:
        raise RuntimeError(
            f"{v.name} was not precompiled; building it now would land in the "
            "timed region"
        )
    try:
        mod = Jit(v, _build(v))
    except Exception as exc:  # noqa: BLE001 - a refused shape is a normal outcome
        with _lock:
            _failed[v] = f"{v.name} does not build: {exc}"
        return None
    with _lock:
        _loaded[v] = mod
    return mod


def failure(v: R.KernelVariant) -> str | None:
    return _failed.get(v)


def precompile(variants) -> None:
    """Build in parallel, one clang process per variant, then load."""
    todo = [v for v in dict.fromkeys(variants) if v not in _loaded]
    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 8) - 2)) as pool:
        list(pool.map(load, todo))


def seal() -> None:
    """Refuse to build from here on: a build inside do_bench corrupts the
    median it reports (one measured 258 % off)."""
    global _sealed
    _sealed = True


def variant_for(hq, hkv, d, m, page_size, layout, window, dtype, batch):
    """ROCM_ATTN's variant_for with the `override()` knobs on top: of the CSV
    row, or of the kernel's defaults where there is no row."""
    base = R.variant_for(hq, hkv, d, m, page_size, layout, window, dtype, batch)
    if base is None:
        base = R.KernelVariant(
            d,
            hq,
            hkv,
            m,
            page_size,
            layout,
            batched=int(batch),
            dtype=dtype,
            window=window,
        )
    return dataclasses.replace(base, **_overrides) if _overrides else base


@contextlib.contextmanager
def override(knobs: dict):
    """Run ROCM_ATTN with these launch knobs (`install` first)."""
    _overrides.clear()
    _overrides.update(knobs)
    try:
        yield
    finally:
        _overrides.clear()


def install(ablate: int = 0) -> None:
    """Make ROCM_ATTN run builds from here instead of _rocm_C's, built under
    VLLM_CACHE_ROOT/rdna35_jit.

    Args:
        ablate: ABLATE of every build.  MEASUREMENT ONLY, wrong numbers; see
            ABLATE in the kernel.

    """
    global _cache, _ablate
    import vllm.envs as envs
    import vllm.v1.attention.backends.rocm_attn as B

    _ablate = ablate
    _cache = (Path(envs.VLLM_CACHE_ROOT) / "rdna35_jit").resolve()
    _cache.mkdir(parents=True, exist_ok=True)
    wrapper = _cache / "wrapper.hip"
    if not wrapper.is_file() or wrapper.read_text() != _WRAPPER:
        tmp = wrapper.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(_WRAPPER)
        os.replace(tmp, wrapper)
    B.load = load
    B.variant_for = variant_for


@contextlib.contextmanager
def paths():
    """The path ROCM_ATTN took for the calls inside: "kernel" (every call on
    the kernel), "split" (decodes on the kernel, the rest on Triton) or
    "triton".  Yields a callable that reports it."""
    import vllm.v1.attention.backends.rocm_attn as B

    impls: list = []
    cls = B.RocmAttentionRdna35Impl
    init = cls.__init__

    def spy(self, *a, **kw):
        init(self, *a, **kw)
        impls.append(self)

    def path() -> str:
        if any(i.split_calls for i in impls):
            return "split"
        ran = any(i.kernel_calls for i in impls)
        fell_back = any(i.fallback_calls for i in impls)
        return "kernel" if ran and not fell_back else "triton"

    cls.__init__ = spy
    try:
        yield path
    finally:
        cls.__init__ = init
