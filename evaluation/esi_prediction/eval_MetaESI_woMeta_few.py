import h5py
import pandas as pd
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'metaesi'))
from metaesi import *

# Independent Testing Function Definition
def ind_test(model_name, dataset_val_e3, model, Feature, save_path, gpu):
    logits = eval_MetaESI(model, dataset_val_e3, Feature, gpu)

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

    features_path = "../../data/human/features/"
    e3_features = h5py.File(features_path + "MetaESI_features_e3.hdf5", "r")
    sub_features = h5py.File(features_path + "MetaESI_features_sub.hdf5", "r")
    Feature = ESIfeature(e3_features, sub_features, c_map_thr = 9)

    # Model Initialization and Loading
    gpu = 0
    model = MetaESI(1280).to(device=try_gpu(gpu))
    model_path = "../../results/esi_benchmark/architecture_ablation/MetaESI_woMeta/"
    model.load_state_dict(torch.load(model_path + "MetaESI_woMeta.pth", map_location=try_gpu(gpu)))

    # Independent Test Execution
    Dataset_val = ESIdataset(dataset, esi_count, coeffs['common_threshold'], coeffs['task_num'], coeffs['support_num'],
                             coeffs['query_num'], type='few')
    dataset_val_e3 = Dataset_val.sample_all_ESIs()

    save_path = "../../results/esi_benchmark/architecture_ablation/MetaESI_woMeta/"
    ind_test("MetaESI_woMeta", dataset_val_e3, model, Feature, save_path, gpu)