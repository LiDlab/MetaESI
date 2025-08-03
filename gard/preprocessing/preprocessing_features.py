import os
import glob
import numpy as np
import h5py
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth
import multiprocessing as mp
from functools import partial
import json

"""
This scripts computes and stores structural features (Residue Depth, Distance Matrix, PAE) 

Key Functions:
- list_of_groups(): Splits list into chunks for parallel processing
- cal_RD(): Calculates residue depth using Biopython
- cal_adj(): Computes CA-CA distance matrix
- load_error_distance(): Loads predicted aligned error (PAE) from JSON
- load_features(): Main function that loads PDB and computes all features

Workflow:
1. Gets list of protein IDs from existing JSON files
2. Processes proteins in parallel batches (800 proteins)
3. For each protein:
   - Calculates residue depth (RD)
   - Computes contact distance matrix (A) 
   - Loads predicted aligned error (PAE)
4. Stores all features in a master HDF5 file with protein IDs as groups

Output:
- GARD_features.hdf5 containing:
  - RD: Residue depth matrix
  - A: CA-CA distance matrix 
  - PAE: Predicted aligned error matrix
"""

def list_of_groups(init_list, childern_list_len):
    list_of_groups = zip(*(iter(init_list),) *childern_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list

def cal_RD(model):
    RD_numpy = list()
    RD = ResidueDepth(model)
    for i in RD.keys():
        RD_numpy.append(list(RD[i]))
    RD = np.array(RD_numpy)

    return RD

def cal_adj(structure):
    residues = [r for r in structure.get_residues()]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one - two)

    return distances

def load_error_distance(AC, data_path):
    with open(data_path + "AF-{}-F1-pae.json".format(AC)) as f_pae:
        data = json.load(f_pae)
    error_dist = np.array(data[0]['distance'])
    size = int(np.sqrt(len(error_dist)))
    error_dist = error_dist.reshape(size, size)

    return error_dist

def load_features(AC, data_path):
    pdbfile = data_path + "AF-{}-F1-model_v2.pdb".format(AC)

    parser = PDBParser()
    structure = parser.get_structure(AC, pdbfile)
    model = structure[0]

    RD = cal_RD(model)
    A = cal_adj(structure)
    PAE = load_error_distance(AC, data_path)

    if RD.shape[0] != A.shape[0]:
        print("ERROR:" + AC)

    return (AC,RD,A,PAE)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    af2_path = os.path.join(script_dir, "../../data/human/raw/alphafold2/")
    af2_files = glob.glob(af2_path + '*.json')
    all_proteins = [os.path.basename(i).split('-')[1] for i in af2_files]
    all_proteins = list(set(all_proteins))

    save_path = os.path.join(script_dir, "../../data/human/features/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f = h5py.File(save_path + "GARD_features.hdf5", "a")

    existing_proteins = set(f.keys())
    proteins_to_process = [p for p in all_proteins if p not in existing_proteins]

    proteins_to_process_lists = list_of_groups(proteins_to_process, 800)

    load_features_dir = partial(load_features,data_path=af2_path)

    for i in tqdm(proteins_to_process_lists):

        pool = mp.Pool()
        res = pool.map(load_features_dir, i)

        for j in res:
            group = f.create_group(j[0])
            group.create_dataset("RD", data=j[1])
            group.create_dataset("A", data=j[2])
            group.create_dataset("PAE", data=j[3])

    f.close()

if __name__ == "__main__":
    main()