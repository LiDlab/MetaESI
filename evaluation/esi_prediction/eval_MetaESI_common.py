import h5py
import numpy as np
import pandas as pd
import copy
import torch

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from metaesi import *

# Cross-validation Function Definition
def Cross_val(model_name, dataset, esi_count, Feature, coeffs, episodes, epochs, save_path, gpu):

    # K-fold Index Generation
    cv_index_set = kfold(k = 5, sample_num = dataset.shape[0])
    fold = 0

    # Training and Validation Data Preparation
    for train_idx, val_idx in cv_index_set:
        train_data = dataset.loc[train_idx].copy()
        train_data.reset_index(inplace = True, drop = True)
        val_data = dataset.loc[val_idx].copy()
        val_data.reset_index(inplace=True, drop=True)

        Dataset_train = ESIdataset(train_data, esi_count, coeffs['common_threshold'], coeffs['task_num'], coeffs['support_num'], coeffs['query_num'], type = 'common')
        Dataset_val = ESIdataset(val_data, esi_count, coeffs['common_threshold'], coeffs['task_num'], coeffs['support_num'], coeffs['query_num'], type = 'common')

        # Meta-training and Model Saving
        model = meta_train_MetaESI(Dataset_train, Feature, episodes, gpu)
        torch.save(model.state_dict(), save_path + "{}_crossval_{}.pth".format(model_name, fold))

        logits = []

        # Fine-tuning and Evaluation Loop
        for common_e3 in list(Dataset_train.esi_count['E3'].values):
            dataset_train_e3 = Dataset_train.sample_specific_ESIs(common_e3, "train")
            MetaESI_e3 = train_MetaESI(copy.deepcopy(model), dataset_train_e3, Feature, epochs, gpu)
            dataset_val_e3 = Dataset_val.sample_specific_ESIs(common_e3, "train")
            logits.append(eval_MetaESI(MetaESI_e3, dataset_val_e3, Feature, gpu))

        # Result Compilation and Performance Evaluation
        logits = np.concatenate(logits)
        logits_csv = save_logits(logits, save_path, model_name, "crossval_{}.csv".format(fold))
        fold += 1
        auc, aupr = evaluate_logits(logits_csv, model_name)

        print('AUC {:.4f} | AUPR {:.4f}'.format(auc, aupr))

# Meta-learning Function Definition
def Meta_learning(dataset, esi_count, Feature, coeffs, episodes, gpu):

    Dataset = ESIdataset(dataset, esi_count, coeffs['common_threshold'], coeffs['task_num'],
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

    save_path = "../../results/esi_benchmark/comparison_with_other_methods/MetaESI/"
    os.makedirs(save_path, exist_ok=True)

    # Cross-validation Execution
    Cross_val("MetaESI", dataset, esi_count, Feature, coeffs, episodes = 10000, epochs = 100, save_path = save_path, gpu = 0)
