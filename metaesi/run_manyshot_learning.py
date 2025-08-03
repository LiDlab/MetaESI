import h5py
import numpy as np
import pandas as pd
import copy
import torch

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from metaesi import *

# Manyshot Learning Function Definition
def manyshot_learning(model_name, model_path, dataset, esi_count, Feature, coeffs, epochs, save_path, gpu):

    Dataset = ESIdataset(dataset, esi_count, coeffs['common_threshold'], coeffs['task_num'], coeffs['support_num'], coeffs['query_num'], type = 'common')

    model = MetaESI(1280).to(device=try_gpu(gpu))
    model.load_state_dict(torch.load(model_path + model_name + '.pth', map_location=try_gpu(gpu)))

    # Majority Learning Loop
    for common_e3 in Dataset.esi_count['E3'].values:

        dataset_train_e3 = Dataset.sample_specific_ESIs(common_e3, "train")
        MetaESI_e3 = train_MetaESI(copy.deepcopy(model), dataset_train_e3, Feature, epochs, gpu)
        torch.save(MetaESI_e3.state_dict(), save_path + common_e3 + ".pth")


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

    model_path = "../models/meta_model/"
    save_path = "../models/e3_specific_model/"

    # Cross-validation Execution
    manyshot_learning("MetaESI", model_path, dataset, esi_count, Feature, coeffs, epochs = 100, save_path = save_path, gpu = 0)
