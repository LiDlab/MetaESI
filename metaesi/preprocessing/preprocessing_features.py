import h5py
import numpy as np
from tqdm import tqdm
import torch

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from gard import GARD

"""
Processes protein features for MetaESI model by combining ESM-2 embeddings with structural features.

Key Components:
1. Feature Integration:
   - Loads ESM-2 embeddings (33-layer representations)
   - Combines with GARD disorder predictions
   - Incorporates contact maps (A) from AlphaFold
   - Includes protein sequences

2. Dual Output Structure:
   - METAESI_features_e3.hdf5: Structured residues (GARD > 1.5)
     * Contains embeddings, positions, contact maps and sequences
   - METAESI_features_sub.hdf5: IDR regions (GARD ≤ 1.5) 
     * Segments contiguous disordered regions
     * Stores embeddings and positions for each segment

3. Processing Logic:
   - Uses GARD threshold to separate structured/disordered regions
   - Groups adjacent disordered residues into segments
   - Maintains original residue numbering
   - Preserves sequence information

Dependencies:
- Requires pre-computed:
  * GARD features
  * ESM-2 embeddings (.pt files)
  * Protein sequences (FASTA)
- Uses custom GARD module for disorder calculations

Output Files:
- METAESI_features_e3.hdf5: Structured residue features
- METAESI_features_sub.hdf5: Disordered region features
"""

def list_of_groups(init_list, childern_list_len):
    list_of_groups = zip(*(iter(init_list),) *childern_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list

def read_fasta(file):
    sequences = {}
    with open(file, 'r') as f:
        key = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                key = line[1:]
                sequences[key] = ''
            else:
                sequences[key] += line
    return sequences

def load_features_esm2(AC, data_path):
    embedding_file = torch.load(os.path.join(data_path, AC + ".pt"))
    embedding = embedding_file['representations'][33].numpy()

    return embedding

def extract_metaesi_feature(save_path, feature_path, fasta_file, verbose=True):


    gard_features = h5py.File(os.path.join(save_path, "GARD_features.hdf5"), "r")
    sequences_dict = read_fasta(fasta_file)
    METAESI_features_e3 = h5py.File(os.path.join(save_path, "MetaESI_features_e3.hdf5"), "a")
    METAESI_features_sub = h5py.File(os.path.join(save_path, "MetaESI_features_sub.hdf5"), "a")

    gard_thr = 1.5

    protein = list(gard_features.keys())

    protein_iter = tqdm(protein) if verbose else protein

    for i in protein_iter:

        if i in METAESI_features_e3 or i in METAESI_features_sub:
            continue

        embedding = load_features_esm2(i, feature_path)
        A = np.array(gard_features[i + "/A"])

        if embedding.shape[0] != A.shape[0]:
            print(f"Skipping {i}: FASTA and PDB sequence lengths appear inconsistent.")

        i_gard = GARD(gard_features, i, cmap_thresh1 = 15, cmap_thresh2 = 6, smooth = 75)   # numpy

        S = sequences_dict[i]

        order = np.arange(1, i_gard.shape[0] + 1)

        if len(i_gard) > 0:
            order_surface = order[i_gard > gard_thr]

            if len(order_surface) > 0:
                group_e3 = METAESI_features_e3.create_group(i)
                group_e3.create_dataset("E", data = embedding[order_surface - 1])
                group_e3.create_dataset("O", data = order_surface)
                group_e3.create_dataset("A", data = A[order_surface - 1, :][:, order_surface - 1])
                dt = h5py.special_dtype(vlen=str)
                group_e3.create_dataset("S", dtype=dt, data=S)


        order_idr = order[i_gard <= gard_thr]
        len_order = len(order_idr)
        if len_order > 0:
            group_sub = METAESI_features_sub.create_group(i)
            add = 1
            iter = 1
            order_open = order_idr[0]
            while iter < len_order:
                order_now = order_idr[iter]
                order_before = order_idr[iter-1]
                if order_now - order_before >= 2:
                    # if order_before - order_open + 1 >= 5:
                    group_sub.create_dataset(str(add), data = embedding[order_open-1:order_before])
                    group_sub.create_dataset("O"+str(add), data = order[order_open-1:order_before])
                    add += 1
                    order_open = order_now
                    # else:
                    #     order_open = order_now
                iter += 1

            # if order_idr[iter-1] - order_open + 1 >= 5:
            group_sub.create_dataset(str(add), data = embedding[order_open-1:order_idr[iter-1]])
            group_sub.create_dataset("O" + str(add), data=order[order_open-1:order_idr[iter-1]])

            # if len(group_sub.keys()) == 0:
            #     del METAESI_features_sub[i]
            # else:
            dt = h5py.special_dtype(vlen=str)
            group_sub.create_dataset("S", dtype=dt, data=S)

    gard_features.close()
    METAESI_features_e3.close()
    METAESI_features_sub.close()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "../../data/human/features/")
    feature_path = os.path.join(script_dir, "../../data/human/esm2/")

    extract_metaesi_feature(save_path, feature_path, fasta_file=save_path + "MetaESI_seq.fasta")