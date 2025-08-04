#!/bin/bash
#
# Downloads and unzips all required data for MetaESI.
#
# Usage: bash download_all_data.sh /path/to/download/directory
set -e

if [[ $# -eq 0 ]]; then
  echo "Error: download directory must be provided as an input argument."
  exit 1
fi

if ! command -v aria2c &> /dev/null ; then
  echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
  exit 1
fi


DOWNLOAD_DIR="$1"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Downloading Alphafold DB v2..."
bash "${SCRIPT_DIR}/download_alphafold.sh" "${DOWNLOAD_DIR}"

echo "Downloading Alphafold per-residue predicted aligned error (PAE) files..."
bash "${SCRIPT_DIR}/download_alphafold_pae.sh" "${DOWNLOAD_DIR}"

echo "Downloading ESM2..."
bash "${SCRIPT_DIR}/download_esm2.sh" "${DOWNLOAD_DIR}"

echo "All data downloaded."