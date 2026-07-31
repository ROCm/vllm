"""Make amd-flashinfer's JIT compile against PyTorch 2.11.

PyTorch 2.11 moved the entire `c10::hip` hipify-v2 backward-compat namespace
behind `#ifdef USE_ROCM` (torch/include/c10/hip/HIPStream.h, "hipify v2
backward compat in external projects"). amd-flashinfer 0.5.3+amd.1 emits
`c10::hip::getCurrentHIPStream()` in its JIT-generated kernels but does not
pass -DUSE_ROCM, so every JIT compile fails with:

    error: no member named 'getCurrentHIPStream' in namespace 'c10::hip';
           did you mean 'c10::cuda::getCurrentCUDAStream'?

This was fine on torch 2.9/2.10, where the compat namespace was unguarded.

Idempotent. Run inside the container after installing amd-flashinfer.
"""

import re
import sys
from pathlib import Path

FLAG = '"-DUSE_ROCM",'


def main() -> int:
    import flashinfer

    target = Path(flashinfer.__file__).parent / "compilation_context_hip.py"
    if not target.exists():
        print(f"ERROR: {target} not found", file=sys.stderr)
        return 1

    src = target.read_text()
    if "-DUSE_ROCM" in src:
        print(f"already patched: {target}")
        return 0

    patched, n = re.subn(
        r"(COMMON_HIPCC_FLAGS[^=]*=\s*\[)",
        lambda m: m.group(1) + "\n        " + FLAG,
        src,
        count=1,
    )
    if n != 1:
        print("ERROR: COMMON_HIPCC_FLAGS list not found", file=sys.stderr)
        return 1

    target.write_text(patched)
    print(f"patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
