#!/usr/bin/env bash
# asmstat.sh <file.s> [kernel-symbol]  — instruction mix for the decode kernel.
set -uo pipefail
F=$1
K=${2:-decode_attn}
SYM=$(grep -oE "^_Z[0-9A-Za-z_]*${K}[0-9A-Za-z_]*:" "$F" | head -1 | tr -d ':')
[ -z "$SYM" ] && SYM=$(grep -oE "^${K}[0-9A-Za-z_]*:" "$F" | head -1 | tr -d ':')
awk -v s="$SYM" '$0 ~ "^"s":" {on=1} on && /s_endpgm/ {print; exit} on' "$F" > /tmp/_k.s
echo "kernel: $SYM   ($(wc -l < /tmp/_k.s) lines)"
echo "--- vector memory ---"
grep -oE '\b(global|buffer|flat|scratch)_(load|store)_[a-z0-9_]+' /tmp/_k.s | sort | uniq -c | sort -rn
echo "--- lds ---"
grep -oE '\bds_(read|write)[a-z0-9_]*' /tmp/_k.s | sort | uniq -c | sort -rn
echo "--- compute / sync ---"
for p in 'v_dot2c?_f32_f16' 'v_fmac?_mix' 'v_pk_' 'v_dual' 'v_fmac?_f32' 'v_exp_f32' \
         's_barrier' 's_waitcnt vmcnt' 's_waitcnt lgkmcnt' 'v_permlane|ds_swizzle|ds_bpermute|v_mov_b32_dpp|row_xmask|row_shl|row_shr' 's_setprio'; do
  printf '%-52s %s\n' "$p" "$(grep -cE "$p" /tmp/_k.s)"
done
echo "--- resources ---"
grep -A20 "^  - \.name:.*$K\|$SYM" "$F" | grep -E '\.vgpr_count|\.sgpr_count|vgpr_spill|private_segment_fixed_size|group_segment_fixed_size' | head -6
echo "--- max loads in flight (longest run of global_load before a waitcnt) ---"
awk '/global_load/{n++; if(n>mx)mx=n} /s_waitcnt vmcnt/{n=0} END{print mx+0}' /tmp/_k.s
