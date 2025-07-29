import os
import glob
import numpy as np
import h5py
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth
import multiprocessing as mp
from functools import partial

"""
This scripts computes and stores structural features (Residue Depth, Distance Matrix, PAE) 

Key Functions:
- list_of_groups(): Splits list into chunks for parallel processing
- cal_RD(): Calculates residue depth using Biopython
- cal_adj(): Computes CA-CA distance matrix
- load_error_distance(): Loads predicted aligned error (PAE) from HDF
- load_features(): Main function that loads PDB and computes all features

Workflow:
1. Gets list of protein IDs from existing HDF files
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
    with h5py.File(data_path + "pae_{}.hdf".format(AC)) as hdf_root:
        error_dist = hdf_root['dist'][...]
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



af2_path = "../../data/human/raw/alphafold2/"
af2_files = glob.glob(af2_path + '*.hdf')
all_proteins = [i.split('_')[-1].split('.')[0] for i in af2_files]

save_path = "../../data/human/features/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

f = h5py.File(save_path + "GARD_features.hdf5", "a")

all_proteins_lists = list_of_groups(all_proteins, 800)

load_features_dir = partial(load_features,data_path=af2_path)

for i in tqdm(all_proteins_lists):

    pool = mp.Pool()
    res = pool.map(load_features_dir, i)

    for j in res:
        group = f.create_group(j[0])
        group.create_dataset("RD", data=j[1])
        group.create_dataset("A", data=j[2])
        group.create_dataset("PAE", data=j[3])

f.close()