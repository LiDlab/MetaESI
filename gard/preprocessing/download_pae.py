import numpy as np
import glob
import os
import socket
from tqdm import tqdm
import logging
import h5py
import json
import tempfile
import sys
import urllib.request

"""
This scripts downloads predicted aligned error (PAE) data from AlphaFold for a list of proteins.

Main Functions:
- download_alphafold_pae(): Downloads PAE data from EBI AlphaFold database for given proteins
  and saves them as HDF5 files with compression.

Workflow:
1. Gets list of protein names from PDB files in specified directory
2. Downloads corresponding PAE data from AlphaFold for each protein
3. Saves PAE matrices as compressed HDF5 files
4. Tracks valid, invalid and existing downloads

Parameters:
- proteins: List of UniProt protein IDs to download
- out_folder: Directory to save HDF5 files 
- out_format: Format for output filenames
- alphafold_pae_url: URL template for AlphaFold PAE JSON data
- timeout: Connection timeout in seconds
- verbose_log: Whether to show detailed logging

Returns tuple of:
- valid_proteins: Successfully downloaded proteins
- invalid_proteins: Proteins not available in AlphaFold
- existing_proteins: Proteins already downloaded previously
"""


def download_alphafold_pae(
    proteins: list,
    out_folder: str,
    out_format: str = "pae_{}.hdf",
    alphafold_pae_url: str = 'https://alphafold.ebi.ac.uk/files/AF-{}-F1-predicted_aligned_error_v2.json',
    timeout: int = 60,
    verbose_log: bool = False,
) -> tuple:

    socket.setdefaulttimeout(timeout)
    valid_proteins = []
    invalid_proteins = []
    existing_proteins = []
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for protein in tqdm(proteins):
        name_out = os.path.join(
            out_folder,
            out_format.format(protein)
        )
        if os.path.isfile(name_out):
            existing_proteins.append(protein)
        else:
            try:
                name_in = alphafold_pae_url.format(protein)
                with tempfile.TemporaryDirectory() as tmp_pae_dir:
                    tmp_pae_file_name = os.path.join(
                        tmp_pae_dir,
                        "pae_{protein}.json"
                    )
                    urllib.request.urlretrieve(name_in, tmp_pae_file_name)
                    with open(tmp_pae_file_name) as tmp_pae_file:
                        data = json.loads(tmp_pae_file.read())
                dist = np.array(data[0]['distance'])
                data_list = [('dist', dist)]
                if getattr(sys, 'frozen', False):
                    print('Using frozen h5py w/ gzip compression')
                    with h5py.File(name_out, 'w') as hdf_root:
                        for key, data in data_list:
                            print(f'h5py {key}')
                            hdf_root.create_dataset(
                                                name=key,
                                                data=data,
                                                compression="gzip",
                                                shuffle=True,
                                            )
                    print('Done')
                else:
                    with h5py.File(name_out, 'w') as hdf_root:
                        for key, data in data_list:
                            hdf_root.create_dataset(
                                                name=key,
                                                data=data,
                                                compression="lzf",
                                                shuffle=True,
                                            )

                valid_proteins.append(protein)
            except urllib.error.HTTPError:
                if verbose_log:
                    logging.info(f"Protein {protein} not available for PAE download.")
                # @ Include HDF IO errors as well, which should probably be handled differently.
                invalid_proteins.append(protein)
    logging.info(f"Valid proteins: {len(valid_proteins)}")
    logging.info(f"Invalid proteins: {len(invalid_proteins)}")
    logging.info(f"Existing proteins: {len(existing_proteins)}")
    return(valid_proteins, invalid_proteins, existing_proteins)


af2_path = "../../data/human/raw/alphafold2/"
af2_files = glob.glob(af2_path + '*.pdb')
all_proteins_name = [i.split("-")[1] for i in af2_files]

try:
    valid_proteins_pae, invalid_proteins_pae, existing_proteins_pae = download_alphafold_pae(
        proteins=all_proteins_name,
        out_folder=af2_path,
        timeout=240)
except:
    print("error")