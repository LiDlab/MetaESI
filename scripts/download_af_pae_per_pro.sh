#!/bin/bash
# Script: download_af_pae_per_protein.sh
# Description: Downloads AlphaFold PDB and PAE files for specified Uniprot ID (skips if exists)
# Parameter 1: Uniprot ID of E3 ligase (e.g., P12345)
# Parameter 2: Output directory path

set -e  # Exit immediately on error

if [ $# -ne 2 ]; then
  echo "Usage: $0 <Uniprot ID> <Output Directory>"
  exit 1
fi

E3_ID=$1
OUTPUT_DIR=$2

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Define filenames and URLs
PDB_FILE="${OUTPUT_DIR}/AF-${E3_ID}-F1-model_v4.pdb"
PAE_FILE="${OUTPUT_DIR}/AF-${E3_ID}-F1-predicted_aligned_error_v4.json"
PDB_URL="https://alphafold.ebi.ac.uk/files/AF-${E3_ID}-F1-model_v4.pdb"
PAE_URL="https://alphafold.ebi.ac.uk/files/AF-${E3_ID}-F1-predicted_aligned_error_v4.json"

# Download function (skips if file exists)
download_if_missing() {
  local file=$1
  local url=$2
  if [ ! -f "$file" ]; then
    wget -q -O "$file" "$url"
  fi
}

# Download files if missing
download_if_missing "$PDB_FILE" "$PDB_URL"
download_if_missing "$PAE_FILE" "$PAE_URL"

# Exit with error code if files are missing
if [ ! -f "${PDB_FILE}" ] || [ ! -f "${PAE_FILE}" ]; then
  exit 1
fi