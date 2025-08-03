import h5py
import pandas as pd
import torch

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from metaesi import *

# Meta-learning Function Definition
def Meta_learning(dataset, esi_count, Feature, coeffs, episodes, gpu):

    Dataset = ESIdataset_with_E3_balanced(dataset, esi_count, coeffs['common_threshold'], coeffs['task_num'],
                               coeffs['support_num'], coeffs['query_num'], type = 'common')

    model = meta_train_MetaESI(Dataset, Feature, episodes, gpu)

    return model


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

    # Meta-learning and Model Saving
    model = Meta_learning(dataset, esi_count, Feature, coeffs, episodes = 10000, gpu = 0)
    model_path = "../../results/esi_benchmark/dataset_ablation/MetaESI_wo_SUB_balanced/"
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path + "MetaESI_wo_SUB_balanced.pth")
