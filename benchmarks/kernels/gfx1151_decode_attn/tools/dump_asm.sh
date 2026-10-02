#!/usr/bin/env bash
# Emit gfx1151 ISA for one kernel variant.
#
#   tools/dump_asm.sh [out.s] [-DNSEG=4 -DNUM_Q_HEADS=8 -DKV_BF16=1 ...]
#
# Built without RDNA35_TORCH_EXT the source instantiates the kernel itself, so
# it compiles as is.  Two findings this session came from reading the output:
# four redundant block-table loads per tile, and eleven instructions spent on
# one IEEE division.
set -euo pipefail

VENV=${VENV:-/scratch/rogarcia/vllm/.venv}
HIPCC=${HIPCC:-$VENV/lib/python3.12/site-packages/_rocm_sdk_devel/bin/hipcc}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC="$(cd "$HERE/../../../../csrc/rocm" && pwd)"
SRC=${SRC:-rdna35_decode_attn.cu}
OUT=${1:-/tmp/kernel.s}; shift || true

"$HIPCC" -O3 --offload-arch=gfx1151 --cuda-device-only -S -o "$OUT" \
    -DHEAD_DIM=256 -DNUM_Q_HEADS=32 -DNUM_KV_HEADS=16 -DMAXM=4 \
    -DBS=16 -DLAYOUT=1 -DNSEG=1 "$@" "$CSRC/$SRC" 2> >(grep -v hip-link >&2)

echo "$OUT"
printf 'instructions   %s\n' "$(grep -cE '^\s+[a-z]' "$OUT")"
for op in global_load_b128 global_load_b64 v_wmma s_waitcnt v_dual \
          v_perm_b32 s_barrier; do
    printf '%-18s %s\n' "$op" "$(grep -c "$op" "$OUT" || true)"
done
printf '%-18s %s\n' vgpr_count "$(grep -m1 '\.vgpr_count' "$OUT" | tr -dc 0-9)"
printf '%-18s %s\n' vgpr_spill "$(grep -m1 '\.vgpr_spill_count' "$OUT" | tr -dc 0-9)"
