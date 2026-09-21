#!/usr/bin/env bash
# sweep.sh "<defs1>" "<defs2>" ...  — build all variants, then measure all.
# Builds and measurements are batched so the lock is taken twice, not 2N times.
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
ITERS=${ITERS:-50}
S=${S:-2048}

d=$OUT/data_m${M}_s${S}
[ -f "$d/ref.bin" ] || "$PY" "$HERE/check.py" gen "$S" "$M" $((100 + M)) "$d"

i=0
names=()
build bash -c '
  set -e
  for spec in "$@"; do :; done
' _ >/dev/null 2>&1 || true

# --- build phase (single lock) ---
cmds=""
for defs in "$@"; do
  n="sw$i"; names+=("$n:$defs")
  cmds+="\"$HIPCC\" -O3 --offload-arch=gfx1151 -o \"$OUT/$n\" -DMAXM=$M $defs \"$SRC\" & "
  i=$((i+1))
done
cmds+="wait"
build bash -c "$cmds" 2>&1 | grep -vE 'hip-link|^$'

# --- measure phase (single lock) ---
mcmd=""
i=0
for defs in "$@"; do
  mcmd+="echo \"### $defs\"; \"$OUT/sw$i\" $S $ITERS \"$d/q.bin\" \"$d/k.bin\" \"$d/v.bin\" \"$OUT/got_sw$i.bin\"; "
  i=$((i+1))
done
measure bash -c "$mcmd" 2>&1 | grep -vE 'amd-gpu-lock'
rc=$?
[ $rc -eq 75 ] && { echo "EXIT 75"; exit 75; }

# --- validate all ---
i=0
for defs in "$@"; do
  "$PY" "$HERE/check.py" check "$d" "$OUT/got_sw$i.bin" "$M" "$defs"
  i=$((i+1))
done
