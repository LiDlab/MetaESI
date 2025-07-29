#!/bin/bash
#
# Downloads and processes AlphaFold-predicted human proteome monomer structures.
#
# Usage: bash download_alphafold.sh /path/to/download/directory
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
TARGET_DIR="${DOWNLOAD_DIR}/raw/alphafold2"
SOURCE_URL="https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000005640_9606_HUMAN_v2.tar"
ARCHIVE_NAME=$(basename "${SOURCE_URL}")

# Create target directory
mkdir -p "${TARGET_DIR}"

echo "Downloading AlphaFold human proteome structures..."
aria2c "${SOURCE_URL}" --dir="${TARGET_DIR}"
pushd "${TARGET_DIR}"
tar -xf "${ARCHIVE_NAME}"

echo "Decompressing PDB files..."
gunzip *.pdb.gz

echo "Cleaning up..."
rm "${ARCHIVE_NAME}"  # Remove the original tar archive
rm -f *.cif.gz
popd > /dev/null

echo "Successfully processed AlphaFold human proteome structures in:"
echo "  ${TARGET_DIR}"