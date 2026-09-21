#!/usr/bin/env bash
# psweep.sh "<defs1>" "<defs2>" ... — paged-layout sweep.
# Batches N builds under one lock and N measurements under another.
# env: M (4) S (2048) ITERS (200) SHUF (0) POOLX (1) TAG (p)
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
SRC=${SRC:-$HERE/decode_attn_paged.hip}
M=${M:-4}
ITERS=${ITERS:-200}
S=${S:-2048}
SHUF=${SHUF:-0}
POOLX=${POOLX:-1}
TAG=${TAG:-p}

d=$OUT/data_m${M}_s${S}_${HQ:-32}_${HKV:-16}_${D:-256}
[ -f "$d/ref.bin" ] || "$PY" "$HERE/check.py" gen "$S" "$M" $((100 + M)) "$d"

cmds=""
i=0
for defs in "$@"; do
  cmds+="\"$HIPCC\" -O3 --offload-arch=gfx1151 -o \"$OUT/${TAG}$i\" -DMAXM=$M $defs \"$SRC\" 2>&1 | grep -v hip-link & "
  i=$((i+1))
done
cmds+="wait"
build bash -c "$cmds" 2>&1 | grep -vE 'hip-link|^$|quiet-lock'

mcmd=""
i=0
for defs in "$@"; do
  mcmd+="\"$OUT/${TAG}$i\" $S $ITERS \"$d/q.bin\" \"$d/k.bin\" \"$d/v.bin\" \"$OUT/got_${TAG}$i.bin\" 4 $SHUF $POOLX; "
  i=$((i+1))
done
measure bash -c "$mcmd" 2>&1 | grep -vE 'amd-gpu-lock|^OK'
rc=${PIPESTATUS[0]}
[ "$rc" = 75 ] && { echo "EXIT 75 (lock busy)"; exit 75; }

i=0
for defs in "$@"; do
  "$PY" "$HERE/check.py" check "$d" "$OUT/got_${TAG}$i.bin" "$M" "$defs"
  i=$((i+1))
done
