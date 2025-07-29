import h5py
import numpy as np
import pandas as pd
from eval_MetaESI_rare import Zeroshot_learning, calculate_similarity_matrix

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'metaesi'))
from metaesi import *



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

    Feature = ESIfeature_wo_Seq(e3_features, sub_features, c_map_thr = 9)

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

    save_path = "../../results/esi_benchmark/architecture_ablation/MetaESI_woSeq/"

    # Zero-shot Learning Execution
    Zeroshot_learning("MetaESI_woSeq", save_path, Dataset_train, Dataset_val, Feature, similarity_matrix, coeffs, nway = 8, kshot = 2, epochs = 10, save_path = save_path, gpu = 1)
