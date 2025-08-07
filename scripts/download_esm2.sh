#!/bin/bash
#
# Processes protein sequences with ESM-2 model.
#
# Usage: bash process_esm.sh /path/to/download/directory
set -e

if [[ $# -eq 0 ]]; then
  echo "Error: download directory must be provided as an input argument."
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

DOWNLOAD_DIR="$1"

# Create target directory structure
mkdir -p "${DOWNLOAD_DIR}/esm2"
SEQUENCE_FILE="${DOWNLOAD_DIR}/features/MetaESI_seq.fasta"

# Create target directory structure
echo "Generating protein sequences..."
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PY_REL_PATH="../metaesi/preprocessing/obtain_protein_sequence.py"
PY_ABS_PATH="${SCRIPT_DIR}/${PY_REL_PATH}"

if [ ! -f "$PY_ABS_PATH" ]; then
    echo "Error: obtain_protein_sequence.py not found in current directory"
    exit 1
fi
python "$PY_ABS_PATH"

echo "Processing sequences with ESM-2..."
python scripts/run_ESM2.py esm2_t33_650M_UR50D "${SEQUENCE_FILE}" \
  "${DOWNLOAD_DIR}/esm2" --repr_layers 33 --include per_tok --truncation_seq_length 100000

echo "Successfully processed protein sequences with ESM-2 in:"
echo "  ${DOWNLOAD_DIR}/esm2"
