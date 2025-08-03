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
def Cross_val(model_name, dataset, esi_count, Feature, coeffs, epochs, save_path, gpu):

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

        # Training and Model Saving
        dataset_train_e3 = Dataset_train.sample_all_ESIs()
        dataset_val_e3 = Dataset_val.sample_all_ESIs()
        model = train_MetaESI_woMeta(dataset_train_e3, Feature, epochs, gpu)
        torch.save(model.state_dict(), save_path + "{}_crossval_{}.pth".format(model_name, fold))

        logits = eval_MetaESI(model, dataset_val_e3, Feature, gpu)

        # Result Compilation and Performance Evaluation
        logits_csv = save_logits(logits, save_path, model_name, "crossval_{}.csv".format(fold))
        fold += 1
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

    features_path = "../../data/human/features/"
    e3_features = h5py.File(features_path + "MetaESI_features_e3.hdf5", "r")
    sub_features = h5py.File(features_path + "MetaESI_features_sub.hdf5", "r")

    Feature = ESIfeature(e3_features, sub_features, c_map_thr = 9)

    save_path = "../../results/esi_benchmark/architecture_ablation/MetaESI_woMeta/"
    os.makedirs(save_path, exist_ok=True)

    # Cross-validation Execution
    Cross_val("MetaESI_woMeta", dataset, esi_count, Feature, coeffs, epochs = 10, save_path = save_path, gpu = 0)
