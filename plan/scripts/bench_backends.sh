#!/usr/bin/env bash
# Phase 4: sweep attention backends over a latency grid.
#
# Runs inside the vllm-fi:dev container. Each (backend, config) pair is a fresh
# `vllm bench latency` invocation, so engine load cost is paid per run and does
# not pollute the measured latency.
#
# GPU state is sampled before and after every run: this box is shared, and a
# benchmark taken while another workload is active is not comparable to one
# taken while it is idle. The sampled values are written next to the results so
# a contaminated run can be spotted rather than silently averaged in.
set -uo pipefail

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
OUTDIR="${OUTDIR:-/vllm/bench-results}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
BACKENDS="${BACKENDS:-ROCM_AITER_FA ROCM_FLASHINFER TRITON_ATTN}"

# name:input_len:output_len:batch_size
CONFIGS="${CONFIGS:-\
decode_bs1:128:512:1 \
decode_bs32:128:512:32 \
prefill_bs1:1024:32:1 \
prefill_bs8:1024:32:8 \
mixed_bs8:512:128:8}"

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
printf 'config\tbackend\tavg_latency_s\tp50_s\tp99_s\tgpu_use_before\tgpu_use_after\tvram_pct\n' > "$SUMMARY"

gpu_use() { rocm-smi --showuse 2>/dev/null | grep -oP 'GPU use \(%\): \K[0-9]+' | head -1; }
vram_pct() { rocm-smi --showmemuse 2>/dev/null | grep -oP 'VRAM%\): \K[0-9]+' | head -1; }

for cfg in $CONFIGS; do
  IFS=: read -r name isl osl bs <<< "$cfg"
  for be in $BACKENDS; do
    log="$OUTDIR/${name}__${be}.log"
    echo "=== $name / $be (isl=$isl osl=$osl bs=$bs)"
    before=$(gpu_use)
    VLLM_ROCM_USE_FLASHINFER=1 vllm bench latency \
      --model "$MODEL" \
      --attention-backend "$be" \
      --input-len "$isl" --output-len "$osl" --batch-size "$bs" \
      --num-iters-warmup "$WARMUP" --num-iters "$ITERS" \
      --enforce-eager \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      > "$log" 2>&1
    rc=$?
    after=$(gpu_use)
    vram=$(vram_pct)

    if [ $rc -ne 0 ]; then
      printf '%s\t%s\tFAILED\t-\t-\t%s\t%s\t%s\n' \
        "$name" "$be" "${before:-?}" "${after:-?}" "${vram:-?}" >> "$SUMMARY"
      echo "    FAILED (rc=$rc), see $log"
      continue
    fi

    avg=$(grep -oP 'Avg latency: \K[0-9.]+' "$log" | tail -1)
    p50=$(grep -oP '50% percentile latency: \K[0-9.]+' "$log" | tail -1)
    p99=$(grep -oP '99% percentile latency: \K[0-9.]+' "$log" | tail -1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$be" "${avg:-?}" "${p50:-?}" "${p99:-?}" \
      "${before:-?}" "${after:-?}" "${vram:-?}" >> "$SUMMARY"
    echo "    avg=${avg}s p50=${p50}s p99=${p99}s (gpu_use ${before}%->${after}%)"
  done
done

echo
echo "===== SUMMARY ====="
column -t -s $'\t' "$SUMMARY"
