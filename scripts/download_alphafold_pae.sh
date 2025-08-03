#!/bin/bash
#
# Downloads AlphaFold Predicted Aligned Error (PAE) data for given proteins (parallel optimized version)
#
# Usage: bash download_alphafold_pae.sh /path/to/alphafold_directory [MAX_CONCURRENCY]
set -e

if [[ $# -eq 0 ]]; then
  echo "Error: Alphafold directory must be provided as an argument."
  exit 1
fi

DOWNLOAD_DIR="$1"
ALPHAFOLD_DIR="${DOWNLOAD_DIR}/raw/alphafold2"
PAE_URL="https://alphafold.ebi.ac.uk/files/AF-{}-F1-predicted_aligned_error_v2.json"
MAX_CONCURRENCY="${2:-8}"  # Default concurrency 8, can be adjusted via parameter

# Check required tools
if ! command -v xargs &> /dev/null; then
  echo "Error: 'xargs' is required for parallel downloads. Install via 'apt-get install findutils' or similar."
  exit 1
fi

# Extract protein names
echo "Collecting protein names from PDB files..."
mapfile -t proteins < <(
  find "$ALPHAFOLD_DIR" -maxdepth 1 -name 'AF-*-F1-model_v2.pdb' -print0 |
  xargs -0 -n1 basename |
  sed 's/AF-//; s/-F1-model_v2.pdb//'
)

# Check if protein files were found
if [[ ${#proteins[@]} -eq 0 ]]; then
  echo "Error: No protein files found in $ALPHAFOLD_DIR"
  exit 1
fi

# Create temporary file list
LIST_FILE=$(mktemp)
printf "%s\n" "${proteins[@]}" > "$LIST_FILE"

# Time formatting function
format_time() {
  seconds=$1
  if [[ $seconds -lt 0 ]]; then
    printf "--:--:--"
    return
  fi
  hours=$((seconds / 3600))
  minutes=$(((seconds % 3600) / 60))
  seconds=$((seconds % 60))
  printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# Progress monitoring function
monitor_progress() {
  total=${#proteins[@]}
  start_time=$(date +%s)
  last_update=$start_time
  completed=0
  success=0
  failed=0
  skipped=0
  declare -A start_times

  while read -r line; do
    current_time=$(date +%s)

    case "$line" in
      START:*)
        protein="${line#START:}"
        start_times["$protein"]=$current_time
        continue
        ;;
      SKIP:*)
        protein="${line#SKIP:}"
        skipped=$((skipped + 1))
        ;;
      OK:*)
        protein="${line#OK:}"
        success=$((success + 1))
        ;;
      FAIL:*)
        protein="${line#FAIL:}"
        failed=$((failed + 1))
        ;;
      *) continue ;;
    esac

    completed=$((success + failed + skipped))
    elapsed=$((current_time - start_time))

    # Calculate remaining time (based on average processing time)
    if [[ $completed -gt 0 ]]; then
      avg_time=$((elapsed / completed))
      remaining=$(( (total - completed) * avg_time / MAX_CONCURRENCY ))
      eta_str=$(format_time $remaining)
    else
      eta_str="--:--:--"
    fi

    # Calculate current speed (tasks per minute)
    speed_str="--"
    if [[ $elapsed -gt 0 ]]; then
      speed=$((completed * 60 / elapsed))
      speed_str="$speed"
    fi

    # Update display at most once per second
    if [[ $((current_time - last_update)) -ge 1 || $completed -eq $total ]]; then
      printf "\rProgress: %d/%d [S:%d ✔:%d ✖:%d] %s/min | ETA: %s" \
        "$completed" "$total" "$skipped" "$success" "$failed" "$speed_str" "$eta_str"
      last_update=$current_time
    fi
  done

  # Final display update
  elapsed=$((last_update - start_time))
  printf "\rProgress: %d/%d [S:%d ✔:%d ✖:%d] Time: %s\n" \
    "$completed" "$total" "$skipped" "$success" "$failed" "$(format_time $elapsed)"
}

# Download task function
download_task() {
  protein="$1"
  output_file="${ALPHAFOLD_DIR}/AF-${protein}-F1-pae.json"

  # Notify monitor that task is starting
  echo "START:${protein}"

  # Skip if file already exists
  if [[ -f "$output_file" ]]; then
    echo "SKIP:${protein}"
    return
  fi

  url="${PAE_URL/\{\}/$protein}"
  tmpfile="${ALPHAFOLD_DIR}/.tmp.${protein}.pae"

  if curl -sSfL --connect-timeout 30 --max-time 120 "$url" > "$tmpfile" 2>/dev/null; then
    mv "$tmpfile" "$output_file"
    echo "OK:${protein}"
  else
    rm -f "$tmpfile" >/dev/null 2>&1
    echo "FAIL:${protein}"
  fi
}

# Export function for parallel use
export -f download_task
export ALPHAFOLD_DIR PAE_URL

echo "Starting parallel downloads with ${MAX_CONCURRENCY} workers..."
echo

# Run parallel download queue
cat "$LIST_FILE" | xargs -P "$MAX_CONCURRENCY" -I {} bash -c 'download_task "{}"' 2>/dev/null | monitor_progress

# Cleanup temporary file
rm -f "$LIST_FILE"

# Results summary
echo
echo "Results in:"
echo "  ${ALPHAFOLD_DIR}/AF-*-F1-pae.json"