#!/bin/bash
# GPU Cooldown Script
# Waits for AMD GPU temperature to drop below a threshold before proceeding.
# Requires: amd-smi
#
# Usage:
#   gpu-cooldown.sh [temperature_threshold] [max_wait_seconds] [check_interval_seconds]
#
# Arguments:
#   temperature_threshold  - Target GPU temperature in Celsius (default: 55)
#   max_wait_seconds      - Maximum wait time in seconds (default: 120)
#   check_interval_seconds - Temperature check interval in seconds (default: 10)
#
# Example:
#   gpu-cooldown.sh 50 180 15  # Wait for 50°C, max 3 minutes, check every 15s
#   gpu-cooldown.sh            # Use defaults: 55°C, 120s, 10s

set -e

# Parse arguments with defaults
TEMP_THRESHOLD=${1:-55}
MAX_WAIT=${2:-120}
INTERVAL=${3:-10}

echo "=== GPU Cooldown ==="
echo "Temperature threshold: ${TEMP_THRESHOLD}°C"
echo "Max wait time: ${MAX_WAIT}s"
echo "Check interval: ${INTERVAL}s"
echo ""

# Verify amd-smi is available
if ! command -v amd-smi &> /dev/null; then
  echo "ERROR: amd-smi command not found"
  echo "This script requires amd-smi to be installed and available in PATH"
  exit 1
fi

echo "Waiting for GPU to cool down..."
elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
  temp_output=$(amd-smi metric -t 2>&1)

  if [ -n "$temp_output" ]; then
    # Extract EDGE temperature value (primary temperature sensor)
    temp=$(echo "$temp_output" | grep "EDGE:" | grep -oP '\d+' | head -1)

    if [ -n "$temp" ]; then
      echo "Current GPU edge temperature: ${temp}°C (${elapsed}s/${MAX_WAIT}s elapsed)..."

      if [ "$temp" -lt "$TEMP_THRESHOLD" ]; then
        echo "✓ GPU temperature is below ${TEMP_THRESHOLD}°C"
        exit 0
      fi
    else
      echo "Warning: Could not parse EDGE temperature from amd-smi output"
      echo "Output was:"
      echo "$temp_output"
    fi
  else
    echo "Warning: amd-smi command returned no output"
  fi

  sleep $INTERVAL
  elapsed=$((elapsed + INTERVAL))
done

echo "GPU cooldown: timeout reached, GPU still hotter than threshold (${TEMP_THRESHOLD})"
exit 0
