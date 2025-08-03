import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import torch

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from metaesi import *

from emd_utils import calculate_normalized_emd_2d, calculate_normalized_emd_1d


# === Data Handling Functions ===
# Functions for data processing and result generation
def generate_interface_EMD(val, Dataset, Feature, gpu):
    """Generate comparative analysis data for interface prediction performance.

    Args:
        val (pd.DataFrame): Validation dataset
        Dataset (ESIdataset): Custom dataset object
        Feature (ESIfeature): Feature handling object
        gpu (int): GPU device index
        num_rand (int): Number of random iterations for baseline

    Returns:
        pd.DataFrame: Combined results with metrics and predictions
    """

    # Initialize data containers
    X_all = np.array(val[['E3_Entry', 'sub_Entry']])
    residue = np.array(val[['Site', 'Start', 'End']])
    df = pd.DataFrame(columns=('Interface Residues (E3)', 'Interface Residues (SUB)'
                               , 'True Positive Residues (E3)', 'True Positive Residues (SUB)'
                               , 'EMD 2D', 'EMD E3', 'EMD SUB'))

    # Process each E3 instance
    all_e3 = list(set(val["E3_Entry"]))

    for e3 in tqdm(all_e3):

        model = fetch_e3_specific_model(e3, Dataset, Feature, gpu
                                        , meta_model_path="../../results/esi_benchmark/architecture_ablation/MetaESI_wo3D/"
                                        , e3_model_path="../../results/interface_benchmark/e3_specific_model_wo3D/"
                                        , meta_model_name = "MetaESI_wo3D.pth")

        df_val_e3 = val[val["E3_Entry"] == e3].copy()
        dataset_val_e3 = np.array(df_val_e3[["E3_Entry", "sub_Entry"]])
        dataset_val_e3 = np.concatenate([dataset_val_e3, np.ones((dataset_val_e3.shape[0], 1))], axis=1)

        Feature.load_ESIs(dataset_val_e3, shuffle=False)

        for idx, (e3_E, e3_A, sub_E_list, label) in enumerate(Feature):

            t = df_val_e3.iloc[idx].name

            val1 = residue[t, 0]
            all_site = re.split('[-;]', val1)
            val1 = []
            for l in range(int(len(all_site) / 2)):
                val1 += [x for x in range(int(all_site[2 * l]), int(all_site[2 * l + 1]) + 1)]

            val2_start = residue[t, 1]
            val2_end = residue[t, 2]
            val2 = [x for x in range(int(val2_start), int(val2_end) + 1)]

            model.eval()
            i_map = []

            for sub_idx in range(len(sub_E_list)):
                sub_E = sub_E_list[sub_idx]

                with torch.no_grad():
                    im = model(e3_E.to(device=try_gpu(gpu)), e3_A.to(device=try_gpu(gpu)),
                               sub_E.to(device=try_gpu(gpu)))
                i_map.append(im)

            i_map = torch.cat(i_map, 1)
            p_hat = torch.max(i_map)
            i_map = i_map.cpu().squeeze().data.numpy()

            if (float(p_hat) < 0.75):
                pred1 = list()
                pred2 = list()
            else:
                order1, order2 = Feature.get_order(dataset_val_e3, idx)
                seq1, seq2 = Feature.get_seq(dataset_val_e3, idx)
                pred1, pred2 = extract_interface_residues(i_map, p_hat, order1, order2, threshold=0.1)

                e3_len = len(seq1)
                sub_len = len(seq2)
                im_val = np.zeros([e3_len, sub_len])    # ground truth
                for i in val1:
                    for j in val2:
                        im_val[i - 1][j - 1] = 1

                im_metaesi = im_completion(e3_len, sub_len, order1, order2, i_map)  # interface map completion

                twod_dist = calculate_normalized_emd_2d(im_metaesi, im_val, e3_len, sub_len)
                e3_dist = calculate_normalized_emd_1d(im_metaesi, im_val, 1)
                sub_dist = calculate_normalized_emd_1d(im_metaesi, im_val, 0)

                # Record results
                df.loc[t] = [str(pred1), str(pred2), str(list(np.intersect1d(pred1, val1))),
                             str(list(np.intersect1d(pred2, val2))), twod_dist, e3_dist, sub_dist]

    # Combine validation data with calculated metrics
    results = pd.concat([val, df], axis=1)

    return results


if __name__=="__main__":
    # Load data
    print("Importing data...")
    dataset_path = "../../data/human/dataset/"
    esi_count = pd.read_csv(dataset_path + "esi_count_per_e3.csv", names=["E3", "count"])
    dataset = pd.read_csv(dataset_path + "dataset.csv")
    Dataset = ESIdataset(dataset, esi_count, type='all')

    features_path = "../../data/human/features/"
    e3_features = h5py.File(features_path + "MetaESI_features_e3.hdf5", "r")
    sub_features = h5py.File(features_path + "MetaESI_features_sub.hdf5", "r")
    Feature = ESIfeature_wo_3D(e3_features, sub_features, c_map_thr=9)

    gpu = 0

    ######
    data_path = "../../data/human/processed/"
    ESI_interface = pd.read_csv(data_path + "ELM_ESI_interface.csv")

    metaesi_EMD = generate_interface_EMD(ESI_interface, Dataset, Feature, gpu)

    save_path = "../../results/interface_benchmark/"
    metaesi_EMD.to_csv(save_path + "Interface_EMD_by_MetaESI_wo3D.csv", index=False)