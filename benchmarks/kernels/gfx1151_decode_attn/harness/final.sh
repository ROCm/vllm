#!/usr/bin/env bash
# final.sh — full validation of the winning v5 config.
# M in 1..5, S in {2048, 48, 2048+KV_PAD variant}, plus the negative control.
set -uo pipefail
# Environment: set HIPCC/PY to your ROCm venv, or source your own helper.
: "${HIPCC:=hipcc}"
: "${PY:=python3}"
# build/measure default to plain exec; wrap them if you have a GPU lock.
command -v build   >/dev/null 2>&1 || build()   { "$@"; }
command -v measure >/dev/null 2>&1 || measure() { "$@"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../kernels" && pwd)"
OUT=$HERE/out
SRC=$HERE/decode_attn_v5.hip
ITERS=${ITERS:-200}

# --- build every (M, variant) once, under a single lock ---
cmds=""
for M in 1 2 3 4 5; do
  for pad in 0 64; do
    cmds+="\"$HIPCC\" -O3 --offload-arch=gfx1151 -o \"$OUT/fin_m${M}_p${pad}\" -DMAXM=$M -DNSEG=1 -DKPW=4 -DKV_PAD=$pad \"$SRC\" & "
  done
done
cmds+="\"$HIPCC\" -O3 --offload-arch=gfx1151 -o \"$OUT/fin_mut\" -DMAXM=4 -DNSEG=1 -DKPW=4 -DMUTATE=1 \"$SRC\" & wait"
build bash -c "$cmds" 2>&1 | grep -vE 'hip-link|^$|quiet-lock'

for M in 1 2 3 4 5; do
  for S in 2048 48; do
    d=$OUT/data_m${M}_s${S}
    [ -f "$d/ref.bin" ] || "$PY" "$HERE/check.py" gen "$S" "$M" $((100 + M)) "$d"
  done
done

# --- measure + dump every case under a single lock ---
mcmd=""
for M in 1 2 3 4 5; do
  for pad in 0 64; do
    for S in 2048 48; do
      it=0; [ "$S" = 2048 ] && [ "$M" = 4 ] && it=$ITERS
      d=$OUT/data_m${M}_s${S}
      mcmd+="echo \"### M=$M S=$S KV_PAD=$pad\"; \"$OUT/fin_m${M}_p${pad}\" $S $it \"$d/q.bin\" \"$d/k.bin\" \"$d/v.bin\" \"$OUT/g_${M}_${pad}_${S}.bin\"; "
    done
  done
done
d=$OUT/data_m4_s48
mcmd+="echo '### NEGATIVE CONTROL (MUTATE=1, S=48)'; \"$OUT/fin_mut\" 48 0 \"$d/q.bin\" \"$d/k.bin\" \"$d/v.bin\" \"$OUT/g_mut.bin\"; "
measure bash -c "$mcmd" 2>&1 | grep -vE 'amd-gpu-lock|^OK'
[ $? -eq 75 ] && { echo "EXIT 75"; exit 75; }

echo "===== validation ====="
fail=0
for M in 1 2 3 4 5; do
  for pad in 0 64; do
    for S in 2048 48; do
      "$PY" "$HERE/check.py" check "$OUT/data_m${M}_s${S}" \
          "$OUT/g_${M}_${pad}_${S}.bin" "$M" "S=$S pad=$pad" || fail=1
    done
  done
done
echo "--- negative control: MUST say FAIL ---"
"$PY" "$HERE/check.py" check "$OUT/data_m4_s48" "$OUT/g_mut.bin" 4 "MUTATE" \
    && echo "!!! negative control PASSED -> the test is blind" || echo "negative control correctly detected"
exit $fail
