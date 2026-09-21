#!/usr/bin/env bash
# v2.sh — build + validate + (optionally) time decode_attn_v2.
#
#   SRC=... M=4 ITERS=50 DEFS="-DKPW=4 -DNT=1" ./v2.sh
#
# Validation runs S=2048 (perf case) and S=48 (bug case), max_rel <= 1e-3.
set -uo pipefail
# Environment: set HIPCC/PY to your ROCm venv, or source your own helper.
: "${HIPCC:=hipcc}"
: "${PY:=python3}"
# build/measure default to plain exec; wrap them if you have a GPU lock.
command -v build   >/dev/null 2>&1 || build()   { "$@"; }
command -v measure >/dev/null 2>&1 || measure() { "$@"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../kernels" && pwd)"
OUT=$HERE/out
mkdir -p "$OUT"
SRC=${SRC:-$HERE/decode_attn_v2.hip}
M=${M:-4}
NSEG=${NSEG:-16}
KV_PAD=${KV_PAD:-0}
ITERS=${ITERS:-0}
DEFS=${DEFS:-}
TAG=${TAG:-v2}
SMALL=${SMALL:-48}
BIN=$OUT/$TAG

build "$HIPCC" -O3 --offload-arch=gfx1151 -o "$BIN" \
    -DMAXM=$M -DNSEG=$NSEG -DKV_PAD=$KV_PAD $DEFS "$SRC" || exit 1

rc_all=0
for S in 2048 $SMALL; do
  d=$OUT/data_m${M}_s${S}
  [ -f "$d/ref.bin" ] || "$PY" "$HERE/check.py" gen "$S" "$M" $((100 + M)) "$d" || exit 1
  it=0; [ "$S" = 2048 ] && it=$ITERS
  measure "$BIN" "$S" "$it" "$d/q.bin" "$d/k.bin" "$d/v.bin" "$OUT/got_$TAG.bin"
  rc=$?
  [ $rc -eq 75 ] && { echo "EXIT 75 — GPU ocupada, reintentar"; exit 75; }
  [ $rc -ne 0 ] && { echo "run failed rc=$rc S=$S"; exit $rc; }
  "$PY" "$HERE/check.py" check "$d" "$OUT/got_$TAG.bin" "$M" "S=$S" || rc_all=1
done
exit $rc_all
