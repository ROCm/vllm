"""Phase 1 gate: does ROCM_FLASHINFER register, resolve, and validate?"""

import torch

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

ok = True


def check(label, got, want=None, truthy=False):
    global ok
    good = bool(got) if truthy else (got == want)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {label}: {got!r}")


print("=== enum member")
b = AttentionBackendEnum.ROCM_FLASHINFER
check("get_path()", b.get_path(),
      "vllm.v1.attention.backends.rocm_flashinfer.RocmFlashInferBackend")

print("=== class import")
cls = b.get_class()
check("get_name()", cls.get_name(), "ROCM_FLASHINFER")
check("name matches enum member", cls.get_name() == b.name, True)
check("forward_includes_kv_cache_update", cls.forward_includes_kv_cache_update, False)
check("required kv layout", cls.get_required_kv_cache_layout(), "NHD")
check("supported dtypes", cls.supported_dtypes, [torch.float16, torch.bfloat16])
check("head sizes", cls.get_supported_head_sizes(), [64, 128, 256])
check("kernel block sizes", cls.get_supported_kernel_block_sizes(), [16, 32, 64])
check("supports_sink", cls.supports_sink(), False)
check("supports_sliding_window", cls.supports_sliding_window(), False)

print("=== kv cache shape / stride order")
shape = cls.get_kv_cache_shape(100, 16, 8, 128)
check("shape (B,H,N,2*hs)", shape, (100, 8, 16, 256))
check("stride order NHD", cls.get_kv_cache_stride_order(), (0, 2, 1, 3))
check("stride order w/ layers", cls.get_kv_cache_stride_order(True), (1, 0, 3, 2, 4))
check("block dim discovery", cls.get_kv_cache_block_dim(16, 8, 128), 0)

print("=== compute capability")
cap = current_platform.get_device_capability()
print(f"  (this GPU capability = {cap})")
check("accepts this gfx942 host", cls.supports_compute_capability(cap), True)

print("=== validate_configuration")
base = dict(
    head_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto", block_size=16,
    use_mla=False, has_sink=False, use_sparse=False, use_mm_prefix=False,
    use_per_head_quant_scales=False, device_capability=cap,
    attn_type="decoder",
)
check("valid config -> no reasons", cls.validate_configuration(**base), [])
check("fp32 rejected",
      cls.validate_configuration(**{**base, "dtype": torch.float32}), truthy=True)
check("head_size 96 rejected",
      cls.validate_configuration(**{**base, "head_size": 96}), truthy=True)
check("fp8 kv rejected",
      cls.validate_configuration(**{**base, "kv_cache_dtype": "fp8"}), truthy=True)
check("sinks rejected",
      cls.validate_configuration(**{**base, "has_sink": True}), truthy=True)
check("MLA rejected",
      cls.validate_configuration(**{**base, "use_mla": True}), truthy=True)
check("encoder attn rejected",
      cls.validate_configuration(**{**base, "attn_type": "encoder"}), truthy=True)
check("sliding window rejected",
      cls.validate_configuration(**base, has_sliding_window=True), truthy=True)
# supports_compute_capability() deliberately ignores its DeviceCapability
# argument and queries the hardware via get_cdna_version() (same convention as
# rocm_aiter_fa.py: DeviceCapability is unreliable on ROCm). So simulate an
# older arch by patching that source, not by passing a fake capability.
import vllm.platforms.rocm as _rocm

_real_cdna = _rocm.get_cdna_version
try:
    _rocm.get_cdna_version = lambda: 2  # CDNA2 == gfx90a
    check("CDNA2 (gfx90a) rejected by capability check",
          cls.supports_compute_capability(cap), False)
    check("CDNA2 rejected by validate_configuration",
          cls.validate_configuration(**base), ["compute capability not supported"])
    _rocm.get_cdna_version = lambda: 4  # CDNA4 == gfx950
    check("CDNA4 (gfx950) accepted", cls.supports_compute_capability(cap), True)
finally:
    _rocm.get_cdna_version = _real_cdna

print("=== env var")
from vllm import envs

check("VLLM_ROCM_USE_FLASHINFER default False", envs.VLLM_ROCM_USE_FLASHINFER, False)
check("is a known env var", "VLLM_ROCM_USE_FLASHINFER" in envs.environment_variables,
      True)
from vllm.envs import compile_factors

check("in compile hash factors",
      "VLLM_ROCM_USE_FLASHINFER" in compile_factors(), True)

print("=== platform priority wiring")
from vllm.platforms.rocm import _get_backend_priorities

prio_off = _get_backend_priorities(use_mla=False, use_sparse=False)
check("absent when env off", AttentionBackendEnum.ROCM_FLASHINFER not in prio_off, True)
import os

os.environ["VLLM_ROCM_USE_FLASHINFER"] = "1"
envs.environment_variables["VLLM_ROCM_USE_FLASHINFER"]()
import importlib

importlib.reload(envs)
prio_on = _get_backend_priorities(use_mla=False, use_sparse=False)
check("present when env on", AttentionBackendEnum.ROCM_FLASHINFER in prio_on, True)
check("first in priority order when on",
      prio_on[0] is AttentionBackendEnum.ROCM_FLASHINFER, True)
check("MLA list unaffected",
      AttentionBackendEnum.ROCM_FLASHINFER
      not in _get_backend_priorities(use_mla=True, use_sparse=False), True)

print()
print("PHASE 1 GATE:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
