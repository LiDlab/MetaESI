import tqdm
import pandas as pd
import numpy as np
import scipy.sparse as sp
import h5py
from scipy.sparse import diags

"""
Calculates GARD (Graph-aware Residue Depth) scores for protein disorder prediction.

Key Functions:
- normalize(): Row-normalizes sparse matrices
- calculate_a_hat(): Creates normalized adjacency matrices with thresholds
- calculate_D(): Computes degree matrices for graph operations  
- calculate_band_matrix(): Generates banded matrices for smoothing
- GARD(): Core function implementing feature enhancement and graph operations
- extract_GARD_values(): Main pipeline for batch processing proteins

Workflow:
1. Loads benchmark protein IDs and precomputed features (RD, A, PAE)
2. For each protein:
   - Processes residue depth (RD) values with graph operations
   - Applies multiple adjacency matrices with different thresholds
   - Performs graph smoothing operations
3. Combines results into DataFrame with positions and GARD scores
4. Saves final GARD predictions to CSV

Parameters:
- cmap_thresh1/cmap_thresh2: Contact map thresholds for graph construction
- smooth: Smoothing parameter for band matrix

Output:
- GARD.csv: Contains predicted disorder scores (GARD) for all benchmark proteins
"""

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def calculate_a_hat(A, P, thr):
    AP = A + P
    AP = np.double(AP < thr)
    A_hat = normalize(AP)

    return A_hat

def calculate_D(A, P, thr):
    AP = A + P
    AP = np.double(AP < thr)
    D = np.array(AP.sum(1))

    return D

def calculate_band_matrix(matrix_len):
    a = np.array(diags([1, 1, 1], [-1, 0, 1], shape=(matrix_len, matrix_len)).toarray())
    A_hat = normalize(a)
    return A_hat

def GARD(features,
         protein_id,
         cmap_thresh1,
         cmap_thresh2,
         smooth):

    fea = np.array(features[protein_id + "/RD"], dtype=np.float32)
    A = np.array(features[protein_id + "/A"], dtype=np.float32)
    P = np.array(features[protein_id + "/PAE"], dtype=np.float32)
    fea = fea[:, 0].T

    # 1. Data Enhancement
    fea = fea * fea
    fea[fea > 10] = 10

    # 2. Structure-Aware Smoothing
    fea = fea * calculate_D(A, P, 9) * calculate_D(A, P, 9) * calculate_D(A, P, 9)
    fea[fea > 22000] = 22000
    A_hat1 = calculate_a_hat(A, P, cmap_thresh1)
    fea_graph = A_hat1.dot(fea)
    A_hat2 = calculate_a_hat(A, P, cmap_thresh2)
    fea_graph = np.linalg.matrix_power(A_hat2, 5).dot(fea_graph)

    # 3. Sequence-Based Smoothing
    fea_graph = np.linalg.matrix_power(calculate_band_matrix(len(fea_graph)), smooth).dot(fea_graph)
    gard = fea_graph / 1000

    return gard


def extract_GARD_values(features,
                       protein_ids: list,
                       cmap_thresh1,
                       cmap_thresh2,
                       smooth):

    res = list()

    for protein_id in tqdm.tqdm(protein_ids):

        gard = GARD(
            features=features,
            protein_id=protein_id,
            cmap_thresh1=cmap_thresh1,
            cmap_thresh2=cmap_thresh2,
            smooth=smooth
        )

        gard = list(gard)
        pos = list(range(1, len(gard)+1))

        res_p = pd.DataFrame({'protein_id': [protein_id]*len(pos),
                              'position': pos,
                              'GARD': gard})

        res.append(res_p)

    res = pd.concat(res)

    return (res)

def main():
    data_path = "../../data/human/features/"
    save_path = "../../results/idr_benchmark/"

    benchmark = open(save_path + "benchmark_proteins.txt","r")
    benchmark_proteins = benchmark.readlines()
    benchmark_proteins = [x.strip() for x in benchmark_proteins]

    GARD = h5py.File(data_path + "GARD_features.hdf5", "r")

    GARD_data = extract_GARD_values(GARD, protein_ids = benchmark_proteins, cmap_thresh1 = 15, cmap_thresh2 = 6, smooth = 75)

    GARD_data.to_csv(save_path + 'GARD.csv', index=False)

if __name__ == "__main__":
    main()