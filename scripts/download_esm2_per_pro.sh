#!/bin/bash
#
# Processes protein sequences with ESM-2 model.
#
# Usage: bash process_esm.sh /path/to/output/directory sequence_file_name
set -e

if [[ $# -lt 2 ]]; then
  echo "Error: both output directory and sequence file name must be provided."
  echo "Usage: bash $0 /path/to/output/directory sequence_file_name"
  exit 1
fi

# Check required commands
REQUIRED_CMDS=(git python)
for cmd in "${REQUIRED_CMDS[@]}"; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "Error: $cmd could not be found. Please install it."
    exit 1
  fi
done

OUTPUT_DIR="$1"
SEQUENCE_FILE_NAME="$2"
SEQUENCE_FILE="${OUTPUT_DIR}/${SEQUENCE_FILE_NAME}"

# Check if input sequence file exists
if [[ ! -f "${SEQUENCE_FILE}" ]]; then
  echo "Error: Sequence file not found at ${SEQUENCE_FILE}"
  exit 1
fi

python scripts/run_ESM2.py esm2_t33_650M_UR50D "${SEQUENCE_FILE}" \
  "${OUTPUT_DIR}" --repr_layers 33 --include per_tok --truncation_seq_length 100000
