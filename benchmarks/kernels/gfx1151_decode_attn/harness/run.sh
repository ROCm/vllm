#!/usr/bin/env bash
# run.sh [M ...]  — build decode_attn for each M, verify against PyTorch.
set -uo pipefail
# Environment: set HIPCC/PY to your ROCm venv, or source your own helper.
: "${HIPCC:=hipcc}"
: "${PY:=python3}"
# build/measure default to plain exec; wrap them if you have a GPU lock.
command -v build   >/dev/null 2>&1 || build()   { "$@"; }
command -v measure >/dev/null 2>&1 || measure() { "$@"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../kernels" && pwd)"
OUT="$HERE"/out
mkdir -p "$OUT"
S=${S:-2048}
NSEG=${NSEG:-16}
KV_PAD=${KV_PAD:-0}
ITERS=${ITERS:-0}
MS=${@:-1 2 3 4 5}

for M in $MS; do
  tag="m${M}_s${S}_n${NSEG}_p${KV_PAD}"
  build "$HIPCC" -O3 --offload-arch=gfx1151 -o "$OUT/attn_$tag" \
      -DMAXM=$M -DNSEG=$NSEG -DKV_PAD=$KV_PAD "$HERE/decode_attn.hip" || exit 1
  "$PY" "$HERE/ref_attn.py" gen "$S" "$M" $((100 + M)) "$OUT/data_$tag" || exit 1
  measure "$OUT/attn_$tag" "$S" "$ITERS" \
      "$OUT/data_$tag/q.bin" "$OUT/data_$tag/k.bin" "$OUT/data_$tag/v.bin" \
      "$OUT/data_$tag/got.bin"
  rc=$?
  [ $rc -eq 75 ] && { echo "EXIT 75 — GPU/lock ocupado, reintentar"; exit 75; }
  [ $rc -ne 0 ] && { echo "run failed rc=$rc"; exit $rc; }
  "$PY" "$HERE/ref_attn.py" check "$OUT/data_$tag" "$OUT/data_$tag/got.bin" "$M"
done
