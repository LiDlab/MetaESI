import h5py
import numpy as np
import pandas as pd
import copy
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'metaesi'))
from metaesi import *

# K-shot Learning Function Definition
def Kshot_learning(model_name, model_path, dataset, esi_count, Feature, coeffs, epochs, save_path, gpu):

    Dataset = ESIdataset(dataset, esi_count, coeffs['common_threshold'], coeffs['task_num'],
                               coeffs['support_num'], coeffs['query_num'], type = 'few')

    model = MetaESI(1280).to(device=try_gpu(gpu))
    model.load_state_dict(torch.load(model_path + model_name + '.pth', map_location=try_gpu(gpu)))

    logits = []

    # K-shot Learning Loop
    for few_e3, count_value in zip(Dataset.esi_count['E3'].values, Dataset.esi_count['count'].values):

        for i in range(count_value):
            support_samples, query_samples = Dataset.sample_specific_task(few_e3, i)    # leave-one-out
            MetaESI_e3 = train_MetaESI(copy.deepcopy(model), support_samples, Feature, epochs, gpu, mini_batch = -1)
            logits.append(eval_MetaESI(MetaESI_e3, query_samples, Feature, gpu))

    # Result Compilation and Evaluation
    logits = np.concatenate(logits)
    logits_csv = save_logits(logits, save_path, model_name, "fewshot.csv")
    auc, aupr = evaluate_logits(logits_csv, model_name)

    print('AUC {:.4f} | AUPR {:.4f}'.format(auc, aupr))


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

    # features_path = "../../data/human/features/"
    features_path = "/home/huawei/lidianke/connect/MetaESINMI/data/human/features/"
    e3_features = h5py.File(features_path + "MetaESI_features_e3.hdf5", "r")
    sub_features = h5py.File(features_path + "MetaESI_features_sub.hdf5", "r")

    Feature = ESIfeature(e3_features, sub_features, c_map_thr = 9)

    model_path = "../../models/meta_model/"
    save_path = "../../results/esi_benchmark/comparison_with_other_methods/MetaESI/"

    # K-shot Learning Execution
    Kshot_learning("MetaESI", model_path, dataset, esi_count, Feature, coeffs, epochs = 10, save_path = save_path, gpu = 0)
