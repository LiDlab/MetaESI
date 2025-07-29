import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from Bio import PDB
from emd_utils import calculate_normalized_emd_2d, calculate_normalized_emd_1d


def extract_alphaC_bfactors(pdb_file_path):
    """
    Extract C-alpha atom B-factors from PDB file.

    Args:
        pdb_file_path (str): Path to PeSTo prediction PDB file

    Returns:
        np.ndarray: Array of B-factor values for C-alpha atoms
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)

    bfactors = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    if 'CA' in residue:
                        alpha_c_atom = residue['CA']
                        bfactors.append(alpha_c_atom.get_bfactor())

    bfactors_array = np.array(bfactors)

    return bfactors_array


def update_bfactor_scores_to_one(bfactor_array, top_n=10):
    """
    Converts top N B-factor scores to 1 and others to 0.

    Args:
        bfactor_array (numpy.ndarray): Input array of B-factor values
        top_n (int): Number of top scores to preserve

    Returns:
        numpy.ndarray: Binary array with 1s indicating top scores
    """
    top_n_indices = np.argsort(bfactor_array)[-top_n:]
    updated_bfactor_array = np.zeros_like(bfactor_array)
    updated_bfactor_array[top_n_indices] = 1

    return updated_bfactor_array


def construct_interaction_map(e3_interface, sub_interface, top_n=10):
    """
    Creates binary interaction matrix between top-scoring residues.

    Args:
        e3_interface (numpy.ndarray): B-factor array for E3 ligase
        sub_interface (numpy.ndarray): B-factor array for substrate
        top_n (int): Number of top residues to consider

    Returns:
        numpy.ndarray: Binary matrix indicating potential interactions
    """
    e3_updated = update_bfactor_scores_to_one(e3_interface, top_n)
    sub_updated = update_bfactor_scores_to_one(sub_interface, top_n)

    i_map = np.zeros((len(e3_interface), len(sub_interface)))

    for i in range(len(e3_interface)):
        if e3_updated[i] == 1:
            for j in range(len(sub_interface)):
                if sub_updated[j] == 1:
                    i_map[i, j] = 1

    return i_map


def generate_interface_EMD(val):
    """
    Main function to calculate interface prediction metrics and EMD distances.

    Args:
        val (pd.DataFrame): Input dataframe containing interaction data

    Returns:
        pd.DataFrame: Enriched dataframe with prediction metrics and EMD results
    """

    X_all = np.array(val[['E3_Entry', 'sub_Entry']])
    residue = np.array(val[['Site', 'Start', 'End']])

    df = pd.DataFrame(columns=('Interface Residues (E3)', 'Interface Residues (SUB)'
                               , 'True Positive Residues (E3)', 'True Positive Residues (SUB)'
                               , 'EMD 2D', 'EMD E3', 'EMD SUB'))

    # Process each E3 entry
    all_e3 = list(set(val["E3_Entry"]))
    for e3 in tqdm(all_e3):

        df_val_e3 = val[val["E3_Entry"] == e3].copy()
        dataset_val_e3 = np.array(df_val_e3[["E3_Entry", "sub_Entry"]])
        dataset_val_e3 = np.concatenate([dataset_val_e3, np.ones((dataset_val_e3.shape[0], 1))], axis=1)

        # Process each interaction
        for idx in range(len(dataset_val_e3)):

            t = df_val_e3.iloc[idx].name

            x0 = str(X_all[t, 0])
            x1 = str(X_all[t, 1])

            if (f"AF-{x0}-F1-model_v2_i0.pdb" in os.listdir("../../results/pesto")) & (f"AF-{x1}-F1-model_v2_i0.pdb" in os.listdir("../../results/pesto")):

                # Interface prediction and processing
                e3_file_path = os.path.join("../../results/pesto", f"AF-{x0}-F1-model_v2_i0.pdb")
                sub_file_path = os.path.join("../../results/pesto", f"AF-{x1}-F1-model_v2_i0.pdb")

                e3_interface = extract_alphaC_bfactors(e3_file_path)
                sub_interface = extract_alphaC_bfactors(sub_file_path)

                i_map = construct_interaction_map(e3_interface, sub_interface, top_n=10)

                # Ground truth processing
                pred1_index, pred2_index = np.where(i_map == 1)
                pred1_index = list(set(pred1_index))
                pred2_index = list(set(pred2_index))
                pred1 = sorted(pred1_index)
                pred2 = sorted(pred2_index)
                pred1 = [i + 1 for i in pred1]
                pred2 = [j + 1 for j in pred2]

                if pred1 and pred2:
                    # Process ground truth data
                    val1 = residue[t, 0]
                    all_site = re.split('[-;]', val1)
                    val1 = []
                    for l in range(int(len(all_site) / 2)):
                        val1 += [x for x in range(int(all_site[2 * l]), int(all_site[2 * l + 1]) + 1)]

                    val2_start = residue[t, 1]
                    val2_end = residue[t, 2]
                    val2 = [x for x in range(int(val2_start), int(val2_end) + 1)]

                    # Calculate EMD distances
                    e3_len = i_map.shape[0]
                    sub_len = i_map.shape[1]
                    im_val = np.zeros([e3_len, sub_len])
                    for i in val1:
                        for j in val2:
                            im_val[i - 1][j - 1] = 1

                    twod_dist = calculate_normalized_emd_2d(i_map, im_val, e3_len, sub_len)
                    e3_dist = calculate_normalized_emd_1d(i_map, im_val, 1)
                    sub_dist = calculate_normalized_emd_1d(i_map, im_val, 0)

                    # Record results
                    df.loc[t] = [str(pred1), str(pred2), str(list(np.intersect1d(pred1, val1))),
                             str(list(np.intersect1d(pred2, val2))), twod_dist, e3_dist, sub_dist]

    results = pd.concat([val, df], axis=1)

    return results


if __name__=="__main__":

    ######
    data_path = "../../data/human/processed/"
    ESI_interface = pd.read_csv(data_path + "ELM_ESI_interface.csv")

    boxplot = generate_interface_EMD(ESI_interface)

    save_path = "../../results/interface_benchmark/"
    boxplot.to_csv(save_path + "Interface_EMD_by_PeSTo.csv", index=False)

