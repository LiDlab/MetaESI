import os
import urllib.request, json
import tqdm
import h5py
import numpy as np
import pandas as pd
import re

"""
Downloads and processes IUPred3 disorder predictions for benchmark proteins.

Key Components:
1. download_iupred():
   - Fetches disorder predictions from IUPred2/3 web API
   - Handles protein batches with progress tracking
   - Saves results as compressed HDF5 files
   - Returns success/failure statistics

2. format_iupred_estimates():
   - Processes downloaded HDF5 files
   - Extracts position, amino acid, and disorder scores
   - Calculates negative disorder scores (1 - iupred3)
   - Formats data into pandas DataFrame

Workflow:
- Reads benchmark protein IDs from file
- Downloads IUPred3 predictions (commented out in example)
- Processes predictions into CSV format with:
  - Protein ID
  - Amino acid sequence
  - Position
  - IUPred3 disorder score (0-1)
  - Negative disorder score (1 - value)

Output:
- iupred3.csv: Contains all disorder predictions for benchmark set

Note: The download function is shown but commented out, assuming files exist locally.
"""

def download_iupred(
        proteins: list,
        out_folder: str,
        iupred_version: int = 2
):
    """
    Function to download IDR estimates from IUPred2/3.
    Parameters
    ----------
    proteins : list
        List of UniProt protein accessions for which to download iupred estimates.
    out_folder : str
        Path to the output folder.
    iupred_version : int
        Version of IUPred to use. Can be 2 or 3. Default is 3.
    Returns
    -------
    : (int, int, int)
    """
    valid_proteins = []
    invalid_proteins = []
    existing_proteins = []

    if iupred_version == 2:
        link = 'https://iupred2a.elte.hu/iupred2a/long/'
    elif iupred_version == 3:
        link = 'https://iupred3.elte.hu/iupred3/long/'
    else:
        raise ValueError("The parameter 'link' must be 2 or 3.")

    for protein in tqdm.tqdm(proteins):
        name_out = os.path.join(
            out_folder,
            f"iupred3_{protein}.hdf"
        )
        if os.path.isfile(name_out):
            existing_proteins.append(protein)
        else:
            try:
                name_in = f'{link}{protein}.json'
                with urllib.request.urlopen(name_in) as url:
                    data = json.loads(url.read().decode())

                pos = list(data['sequence'])
                iupred3 = data['iupred2']

                data_list = [('pos', pos), ('iupred3', iupred3)]

                with h5py.File(name_out, 'w') as hdf_root:
                    for key, data in data_list:
                        hdf_root.create_dataset(
                            name=key,
                            data=data,
                            compression="lzf",
                            shuffle=True,
                        )
                valid_proteins.append(protein)
            except:
                invalid_proteins.append(protein)
    print(f"Valid proteins: {len(valid_proteins)}")
    print(f"Invalid proteins: {len(invalid_proteins)}")
    print(f"Existing proteins: {len(existing_proteins)}")
    return (valid_proteins, invalid_proteins, existing_proteins)


def format_iupred_estimates(
        directory: str,
        protein_ids: list):

    res = list()

    for file in tqdm.tqdm(os.listdir(directory)):

        if file.endswith("hdf"):
            filepath = os.path.join(directory, file)

            protein_id = re.sub(r'.hdf', '', file)
            protein_id = re.sub(r'iupred3_', '', protein_id)

            if protein_id in protein_ids:
                with h5py.File(r'' + directory + '/iupred3_' + protein_id + '.hdf', 'r') as hdf_root:
                    AA = hdf_root['pos'][...]
                    iupred3 = hdf_root['iupred3'][...]

                AA = [a.decode("utf-8") for a in AA]
                ipred_name = os.path.basename(directory)
                res_p = pd.DataFrame({'protein_id': protein_id,
                                      'AA': AA,
                                      'position': np.arange(1, len(AA) + 1),
                                      ipred_name: iupred3,
                                      'neg_iupred3': 1 - iupred3})

                res.append(res_p)

    res = pd.concat(res)

    return (res)

data_path = "../../results/idr_benchmark/iupred3/"
save_path = "../../results/idr_benchmark/"

benchmark = open(save_path + "benchmark_proteins.txt","r")
benchmark_proteins = benchmark.readlines()
benchmark_proteins = [x.strip() for x in benchmark_proteins]
# valid_proteins_iupred3, invalid_proteins_iupred3, existing_proteins_iupred3 = download_iupred(
#     proteins=benchmark_proteins,
#     out_folder=iupred3_dir,
#     iupred_version=3)

iupred3_data = format_iupred_estimates(data_path, benchmark_proteins)
iupred3_data.to_csv(save_path + 'iupred3.csv', index=False)