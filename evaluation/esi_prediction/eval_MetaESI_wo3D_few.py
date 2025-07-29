import h5py
import pandas as pd
from eval_MetaESI_few import Kshot_learning

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

    Feature = ESIfeature_wo_3D(e3_features, sub_features, c_map_thr = 9)

    save_path = "../../results/esi_benchmark/architecture_ablation/MetaESI_wo3D/"

    # K-shot Learning Execution
    Kshot_learning("MetaESI_wo3D", save_path, dataset, esi_count, Feature, coeffs, epochs = 10, save_path = save_path, gpu = 0)
