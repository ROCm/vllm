#!/usr/bin/env bash
# Emit gfx1151 ISA for one kernel variant.
#
#   tools/dump_asm.sh [out.s] [-DNSEG=4 -DNUM_Q_HEADS=8 ...]
#
# The kernels are templates, so nothing is instantiated unless something asks
# for it; this wraps the source in a translation unit that does.  Two findings
# this session came from reading the output: four redundant block-table loads
# per tile, and eleven instructions spent on one IEEE division.
set -euo pipefail

VENV=${VENV:-/scratch/rogarcia/vllm/.venv}
HIPCC=${HIPCC:-$VENV/lib/python3.12/site-packages/_rocm_sdk_devel/bin/hipcc}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC="$(cd "$HERE/../../../../csrc/rocm" && pwd)"
SRC=${SRC:-rdna35_decode_attn.cu}          # or rdna35_decode_attn_smallgrid.cu
OUT=${1:-/tmp/kernel.s}; shift || true

TU=$(mktemp /tmp/inst_XXXX.cu)
cat > "$TU" <<EOF
#include "$SRC"
template __global__ void decode_attn<_Float16>(const _Float16*, const _Float16*,
    const int*, float*, float*, float*, _Float16*, int, float);
template __global__ void reduce_segments<_Float16>(const float*, const float*,
    const float*, _Float16*);
EOF

"$HIPCC" -O3 --offload-arch=gfx1151 --cuda-device-only -S -o "$OUT" -I "$CSRC" \
    -DDECODE_ATTN_NO_MAIN \
    -DHEAD_DIM=256 -DNUM_Q_HEADS=32 -DNUM_KV_HEADS=16 -DMAXM=4 \
    -DBS=16 -DLAYOUT=1 -DNSEG=1 -DKPW=4 -DFUSED=1 "$@" "$TU"
rm -f "$TU"

echo "$OUT"
printf 'instructions   %s\n' "$(grep -cE '^\s+[a-z]' "$OUT")"
for op in global_load_b128 global_load_b32 ds_bpermute_b32 s_waitcnt v_dual \
          v_div_scale s_barrier; do
    printf '%-18s %s\n' "$op" "$(grep -c "$op" "$OUT" || true)"
done
