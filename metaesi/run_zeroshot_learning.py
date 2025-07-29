import h5py
import numpy as np
import pandas as pd
import copy
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'metaesi'))
from metaesi import *


def calculate_similarity_matrix(e3_features, protein_ids, rare_proteins, head_proteins):
    """
    Calculate cosine similarity matrix between rare E3s and head E3s

    Parameters:
    e3_features: np.array
        Feature matrix with shape [n_proteins, n_features]
    protein_ids: list
        List of protein IDs corresponding to rows in e3_features
    rare_proteins: list
        List of rare protein IDs to calculate similarities for
    head_proteins: list
        List of head protein IDs to compare against

    Returns:
    pd.DataFrame
        Similarity matrix with rare proteins as rows and head proteins as columns
    """

    all_e3_features = {pid: e3_features[i] for i, pid in enumerate(protein_ids)}

    similarity_results = {}
    for rare_entry in rare_proteins:
        if rare_entry in all_e3_features:
            rare_vector = all_e3_features[rare_entry]
            similarities = []

            for head_entry in head_proteins:
                if head_entry in all_e3_features:
                    head_vector = all_e3_features[head_entry]
                    cosine_sim = np.dot(rare_vector, head_vector) / (
                            np.linalg.norm(rare_vector) * np.linalg.norm(head_vector))
                    similarities.append((head_entry, cosine_sim))

            similarity_results[rare_entry] = similarities

    similarity_matrix = pd.DataFrame(0.0, index=rare_proteins, columns=head_proteins)
    for rare_entry, similarities in similarity_results.items():
        for head_entry, score in similarities:
            similarity_matrix.at[rare_entry, head_entry] = score

    return similarity_matrix

# Function to Get Top-N Similar E3
def get_top_n_similar_entries(similarity_matrix, query_rare_e3, n):
    if query_rare_e3 not in similarity_matrix.index:
        raise ValueError(f"{query_rare_e3} not found in similarity matrix.")
    similarities = similarity_matrix.loc[query_rare_e3]

    top_n_similar = similarities.nlargest(n)

    return top_n_similar.index.tolist()

# Zero-shot Learning Function Definition
def Zeroshot_learning(model_name, model_path, Dataset_train, Dataset_val, Feature, similarity_matrix, coeffs, nway, kshot, epochs, save_path, gpu):

    # Model Initialization and Loading
    model = MetaESI(1280).to(device=try_gpu(gpu))
    model.load_state_dict(torch.load(model_path + model_name + '.pth', map_location=try_gpu(gpu)))

    # Zero-shot Learning Loop
    for rare_e3 in Dataset_val.e3:
        head_e3s = get_top_n_similar_entries(similarity_matrix, rare_e3, nway)

        MetaESI_e3 = train_MetaESI_rare(Dataset_train, rare_e3, head_e3s, kshot, copy.deepcopy(model), Feature, epochs,
                                            gpu, mini_batch=-1)

        torch.save(MetaESI_e3.state_dict(), save_path + rare_e3 + ".pth")



if __name__ == "__main__":

    # Parameter Initialization and Data Loading
    coeffs = {
        'common_threshold': 5,
        'task_num': 10,
        'support_num': 2,
        'query_num': 2
    }

    print("Importing data...")
    dataset_path = "../../data/human/dataset/"
    esi_count = pd.read_csv(dataset_path + "esi_count_per_e3.csv")
    dataset = pd.read_csv(dataset_path + "dataset.csv")

    features_path = "../../data/human/features/"
    e3_features = h5py.File(features_path + "MetaESI_features_e3.hdf5", "r")
    sub_features = h5py.File(features_path + "MetaESI_features_sub.hdf5", "r")

    Feature = ESIfeature(e3_features, sub_features, c_map_thr = 9)

    Dataset_train = ESIdataset(dataset, esi_count, coeffs['common_threshold'], coeffs['task_num'],
                               coeffs['support_num'], coeffs['query_num'], type='head')

    Dataset_val = ESIdataset(dataset, esi_count, coeffs['common_threshold'], coeffs['task_num'], coeffs['support_num'], coeffs['query_num'], type = 'rare')

    # Similarity Matrix Preparation
    head_e3 = Dataset_train.e3
    rare_e3 = Dataset_val.e3

    features_path = "/home/huawei/lidianke/connect/TransDSI/src/VGAE.npy"
    uniprot_path = "/home/huawei/lidianke/connect/TransDSI/data/uniprot.tsv"
    e3_features = np.load(features_path)
    uniprot_df = pd.read_table(uniprot_path)
    protein_ids = list(uniprot_df['Entry'])

    similarity_matrix = calculate_similarity_matrix(
        e3_features=e3_features,
        protein_ids=protein_ids,
        rare_proteins=rare_e3,
        head_proteins=head_e3
    )

    model_path = "../models/meta_model/"
    save_path = "../models/e3_specific_model/"

    # Zero-shot Learning Execution
    Zeroshot_learning("MetaESI", model_path, Dataset_train, Dataset_val, Feature, similarity_matrix, coeffs, nway = 8, kshot = 2, epochs = 10, save_path = save_path, gpu = 1)
