import h5py
import pandas as pd
import torch

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from metaesi import *

# Pan-learning Function Definition
def Pan_learning(dataset, esi_count, Feature, coeffs, epochs, gpu):

    Dataset = ESIdataset(dataset, esi_count, coeffs['common_threshold'], coeffs['task_num'],
                               coeffs['support_num'], coeffs['query_num'], type = 'common')

    dataset = Dataset.sample_all_ESIs()

    PanESI = train_MetaESI_woMeta(dataset, Feature, epochs, gpu)

    return PanESI


if __name__ == "__main__":

    # Parameter Initialization and Data Loading
    coeffs = {
        'common_threshold': 5,
        'task_num': 10,
        'support_num': 2,
        'query_num': 2
    }

    print("Importing data...")
    dataset_path = "../data/human/dataset/"
    esi_count = pd.read_csv(dataset_path + "esi_count_per_e3.csv")
    dataset = pd.read_csv(dataset_path + "dataset.csv")

    features_path = "../data/human/features/"
    e3_features = h5py.File(features_path + "MetaESI_features_e3.hdf5", "r")
    sub_features = h5py.File(features_path + "MetaESI_features_sub.hdf5", "r")
    Feature = ESIfeature(e3_features, sub_features, c_map_thr = 9)

    # Pan-learning and Model Saving
    model = Pan_learning(dataset, esi_count, Feature, coeffs, epochs = 10, gpu = 0)
    model_path = "../models/meta_model/"
    torch.save(model.state_dict(), model_path + "PanESI.pth")
