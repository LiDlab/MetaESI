import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset

class ESIdataset(Dataset):

    """
    Custom PyTorch Dataset for E3-Substrate Interaction (ESI) prediction with meta learning support.

    Key Features:
    1. Flexible Task Sampling:
       - Supports different E3 types (common/few/rare/all/head)
       - Generates balanced tasks with support/query sets
       - Enables both random and specific task sampling

    2. Data Organization:
       - Pads samples to create balanced positive/negative samples (E3-centric and SUB-centric)
       - Formats data as [E3, SUB, label] triplets
       - Maintains original dataset structure while enabling few-shot learning

    3. Sampling Methods:
       - sample_tasks(): Randomly samples tasks for meta learning
       - sample_specific_task(): Samples specific E3 with controlled support/query for few-shot learning
       - sample_kshot_ESIs(): Samples k examples for zero-shot learning
       - sample_specific_ESIs(): Gets all interactions for specific E3
       - sample_all_ESIs(): Returns complete dataset

    Parameters:
    - esi_dataset: DataFrame containing ESI interactions
    - esi_count: DataFrame with interaction counts per E3
    - common_threshold: Count threshold for "common" E3s
    - task_num: Number of tasks per episode
    - support_num: Support set size per task
    - query_num: Query set size per task
    - type: E3 type ('common','few','rare','all','head')

    Usage:
    - Designed for meta-learning approaches
    - Supports both training and evaluation modes
    - Enables flexible few-shot learning scenarios
    """

    def __init__(self, esi_dataset, esi_count, common_threshold = 5, task_num = 10, support_num = 2, query_num = 2, type = 'common'):
        self.task_num = task_num
        self.support_num = support_num
        self.query_num = query_num
        self.type = type

        if self.type == 'common':
            esi_count = esi_count[esi_count["count"] >= common_threshold]
            esi_count.reset_index(inplace=True, drop=True)
        if self.type == 'few':
            esi_count = esi_count[(esi_count["count"] < common_threshold) & (esi_count["count"] > 1)]
            esi_count.reset_index(inplace=True, drop=True)
        if self.type == 'rare':
            esi_count = esi_count[esi_count["count"] == 1]
            esi_count.reset_index(inplace=True, drop=True)
        if self.type == 'all':
            esi_count.reset_index(inplace=True, drop=True)
        if self.type == 'head':
            esi_count = esi_count[esi_count["count"] > 1]
            esi_count.reset_index(inplace=True, drop=True)
        self.esi_count = esi_count
        self.e3 = list(self.esi_count['E3'])

        esi_dataset = esi_dataset[["E3", "SUB", "SUB_random", "E3_random"]].copy()
        self.esi_dataset = esi_dataset

    def pad_samples(self, raw_samples):
        # ["E3", "SUB", "SUB_random", "E3_random"] to ["E3","SUB","label"]
        samples_num = raw_samples.shape[0]

        samples = np.zeros([samples_num * 3, 2]).astype("object")
        samples[:samples_num * 2, 0] = np.tile([raw_samples[:, 0]], 2)
        samples[samples_num * 2:, 0] = raw_samples[:, 3]
        samples[:, 1] = np.concatenate([raw_samples[:, 1], raw_samples[:, 2], raw_samples[:, 1]])
        labels = np.concatenate([np.ones((samples_num, 1)), np.zeros((samples_num * 2, 1))], axis=0)
        samples_labels = np.concatenate([samples, labels], axis=1)
        return samples_labels


    def sample_tasks(self):
        select_e3s = self.esi_count.sample(n=self.task_num)
        select_e3s = list(select_e3s["E3"].values)

        self.support_samples = []
        self.query_samples = []

        for select_e3 in select_e3s:
            select_esi = self.esi_dataset[self.esi_dataset["E3"] == select_e3].copy()
            select_esi = np.array(select_esi.sample(n= self.support_num + self.query_num))
            self.support_samples += [select_esi[:self.support_num]]
            self.query_samples += [select_esi[self.support_num:]]


    def sample_specific_task(self, e3, esi_index, K = None, shuffle: bool = False):
        select_esi = self.esi_dataset[self.esi_dataset["E3"] == e3].copy()
        select_esi = np.array(select_esi)

        query_samples = self.pad_samples(select_esi[esi_index: esi_index+1])
        other_samples = np.delete(select_esi, esi_index, axis=0)
        if shuffle:
            np.random.shuffle(other_samples)
        if K:
            support_samples = self.pad_samples(other_samples[:K])
        else:
            support_samples = self.pad_samples(other_samples)

        return support_samples, query_samples


    def sample_kshot_ESIs(self, e3, K = None, shuffle: bool = False):
        select_esi = self.esi_dataset[self.esi_dataset["E3"] == e3].copy()
        select_esi = np.array(select_esi)
        if shuffle:
            np.random.shuffle(select_esi)
        if K:
            kshot_esi = self.pad_samples(select_esi[:K])
        else:
            kshot_esi = self.pad_samples(select_esi)

        return kshot_esi


    def sample_specific_ESIs(self, e3, state = "train"):
        select_esi = self.esi_dataset[self.esi_dataset["E3"] == e3].copy()
        select_esi = np.array(select_esi)
        if state == "train":
            select_esi = self.pad_samples(select_esi)
        else:
            select_esi = np.concatenate([select_esi[:,:2], np.ones((select_esi.shape[0], 1))], axis=1)

        return select_esi


    def sample_all_ESIs(self):
        all_e3 = list(self.esi_count["E3"].values)
        all_esi = self.esi_dataset[self.esi_dataset["E3"].isin(all_e3)].copy()
        all_esi = np.array(all_esi)
        all_esi = self.pad_samples(all_esi)

        return all_esi


    def __len__(self):
        return self.task_num

    def __getitem__(self, idx):
        support_samples = self.pad_samples(self.support_samples[idx])
        query_samples = self.pad_samples(self.query_samples[idx])
        return support_samples, query_samples



class ESIdataset_with_E3_balanced(ESIdataset):
    def __init__(self, esi_dataset, esi_count, common_threshold=5, task_num=10, support_num=2, query_num=2, type='common'):
        super().__init__(esi_dataset, esi_count, common_threshold, task_num, support_num, query_num, type)

    def pad_samples(self, raw_samples):
        # ["E3", "SUB", "SUB_random", "E3_random"] to ["E3","SUB","label"]
        samples_num = raw_samples.shape[0]

        samples = np.zeros([samples_num * 2, 2]).astype("object")
        samples[:, 0] = np.tile([raw_samples[:, 0]], 2)
        samples[:, 1] = np.concatenate([raw_samples[:, 1], raw_samples[:, 2]])
        labels = np.concatenate([np.ones((samples_num, 1)), np.zeros((samples_num, 1))], axis=0)
        samples_labels = np.concatenate([samples, labels], axis=1)
        return samples_labels



class ESIdataset_with_SUB_balanced(ESIdataset):
    def __init__(self, esi_dataset, esi_count, common_threshold=5, task_num=10, support_num=2, query_num=2, type='common'):
        super().__init__(esi_dataset, esi_count, common_threshold, task_num, support_num, query_num, type)

    def pad_samples(self, raw_samples):
        # ["E3", "SUB", "SUB_random", "E3_random"] to ["E3","SUB","label"]
        samples_num = raw_samples.shape[0]

        samples = np.zeros([samples_num * 2, 2]).astype("object")
        samples[:, 0] = np.concatenate([raw_samples[:, 0], raw_samples[:, 3]])
        samples[:, 1] = np.tile([raw_samples[:, 1]], 2)
        labels = np.concatenate([np.ones((samples_num, 1)), np.zeros((samples_num, 1))], axis=0)
        samples_labels = np.concatenate([samples, labels], axis=1)
        return samples_labels



class ESIfeature(Dataset):

    """
    Custom PyTorch Dataset for processing E3-Substrate Interaction (ESI) features.

    Key Functionality:
    1. Feature Loading:
       - Loads E3 enzyme features (structured region residues) from HDF5
       - Loads substrate features (disordered regions) from HDF5
       - Handles both sequence and structural features

    2. Data Processing:
       - Converts contact maps to sparse adjacency matrices (for pyg)
       - Normalizes features using specified contact threshold
       - Organizes substrate features into lists for variable-length regions

    3. Access Methods:
       - get_order(): Retrieves residue numbering information
       - get_seq(): Extracts protein sequences
       - __getitem__(): Provides formatted features for model input

    Parameters:
    - e3_features: HDF5 file containing E3 enzyme features
    - sub_features: HDF5 file containing substrate features
    - c_map_thr: Contact map threshold for adjacency matrix creation

    Output Format:
    For each interaction returns:
    1. e3_E: E3 embeddings (tensor)
    2. e3_A: E3 contact map adjacency matrix (sparse)
    3. sub_E_list: List of substrate region embeddings (tensors)
    4. label: Interaction label (1/0) (tensor)

    Usage Notes:
    - Designed for graph-based interaction prediction
    - Handles variable-length substrate regions
    """

    def __init__(self, e3_features, sub_features, c_map_thr):
        self.e3_features = e3_features
        self.sub_features = sub_features
        self.c_map_thr = c_map_thr

    def load_ESIs(self, samples, shuffle: bool = True):
        np.random.seed(666)
        if shuffle:
            np.random.shuffle(samples)
        self.samples = samples

    def get_order(self, samples, idx):
        e3 = samples[idx, 0]
        sub = samples[idx, 1]
        e3_O = np.array(self.e3_features[e3 + "/O"], dtype=np.float32)
        sub_split = list(self.sub_features[sub].keys())
        sub_O_num = int(len(sub_split) / 2)
        sub_O_list = list()
        for s in range(1, sub_O_num + 1):
            sub_O_list.append(np.array(self.sub_features[sub + "/O" + str(s)]))
        sub_O = np.concatenate(sub_O_list)

        return e3_O, sub_O

    def get_seq(self, samples, idx):
        e3 = samples[idx, 0]
        sub = samples[idx, 1]
        e3_seq = str(np.array(self.e3_features[e3 + "/S"])).split("'")[1]
        sub_seq = str(np.array(self.sub_features[sub + "/S"])).split("'")[1]

        return e3_seq, sub_seq

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        e3 = self.samples[idx, 0]
        sub = self.samples[idx, 1]
        label = self.samples[idx, 2]

        e3_E = torch.from_numpy(np.array(self.e3_features[e3 + "/E"], dtype=np.float32))
        e3_A = np.array(self.e3_features[e3 + "/A"], dtype=np.float32)
        e3_A = np.double(e3_A < self.c_map_thr)
        tmp_coo = sp.coo_matrix(e3_A)
        e3_A_indices = np.vstack((tmp_coo.row, tmp_coo.col))
        e3_A = torch.LongTensor(e3_A_indices)  # torch.LongTensor

        sub_split = list(self.sub_features[sub].keys())
        sub_E_num = int(len(sub_split) / 2)
        sub_E_list = list()
        for s in range(1, sub_E_num + 1):
            sub_E_list.append(torch.from_numpy(np.array(self.sub_features[sub + "/" + str(s)], dtype=np.float32)))

        label = torch.from_numpy(np.array(label, dtype=np.float32))

        return e3_E, e3_A, sub_E_list, label



class ESIfeature_wo_Seq(Dataset):

    """
    Variant of ESIfeature with randomized feature order (ablation study).
    """

    def __init__(self, e3_features, sub_features, c_map_thr):
        self.e3_features = e3_features
        self.sub_features = sub_features
        self.c_map_thr = c_map_thr

    def load_ESIs(self, samples, shuffle: bool = True):
        np.random.seed(666)
        if shuffle:
            np.random.shuffle(samples)
        self.samples = samples

    def get_order(self, samples, idx):
        e3 = samples[idx, 0]
        sub = samples[idx, 1]
        e3_O = np.array(self.e3_features[e3 + "/O"], dtype=np.float32)
        np.random.seed(666)
        np.random.shuffle(e3_O)

        sub_split = list(self.sub_features[sub].keys())
        sub_O_num = int(len(sub_split) / 2)
        sub_O_list = list()
        for s in range(1, sub_O_num + 1):
            sub_O_list.append(np.array(self.sub_features[sub + "/O" + str(s)]))
        sub_O = np.concatenate(sub_O_list)
        np.random.seed(666)
        np.random.shuffle(sub_O)

        return e3_O, sub_O

    def get_seq(self, samples, idx):
        e3 = samples[idx, 0]
        sub = samples[idx, 1]
        e3_seq = str(np.array(self.e3_features[e3 + "/S"])).split("'")[1]
        sub_seq = str(np.array(self.sub_features[sub + "/S"])).split("'")[1]

        return e3_seq, sub_seq

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        e3 = self.samples[idx, 0]
        sub = self.samples[idx, 1]
        label = self.samples[idx, 2]

        e3_E = np.array(self.e3_features[e3 + "/E"], dtype=np.float32)
        np.random.seed(666)
        np.random.shuffle(e3_E)
        e3_E = torch.from_numpy(e3_E)

        e3_A = np.array(self.e3_features[e3 + "/A"], dtype=np.float32)
        e3_A = np.double(e3_A < self.c_map_thr)
        tmp_coo = sp.coo_matrix(e3_A)
        e3_A_indices = np.vstack((tmp_coo.row, tmp_coo.col))
        e3_A = torch.LongTensor(e3_A_indices)  # torch.LongTensor

        sub_split = list(self.sub_features[sub].keys())
        sub_E_num = int(len(sub_split) / 2)
        sub_E_list = list()
        for s in range(1, sub_E_num + 1):
            sub_E = np.array(self.sub_features[sub + "/" + str(s)], dtype=np.float32)
            np.random.seed(666)
            np.random.shuffle(sub_E)
            sub_E_list.append(torch.from_numpy(sub_E))
        label = torch.from_numpy(np.array(label, dtype=np.float32))

        return e3_E, e3_A, sub_E_list, label


class ESIfeature_wo_3D(Dataset):

    """
    Variant of ESIfeature with simplified adjacency structure (3D ablation).

    Key Differences from ESIfeature:
       - Replaces original contact map (e3_A) with a simple band matrix
       - Band matrix only encodes sequential neighbors (i±1 connectivity)
       - Removes all long-range 3D structural contacts while keeping chain continuity

    """

    def __init__(self, e3_features, sub_features, c_map_thr):
        self.e3_features = e3_features
        self.sub_features = sub_features
        self.c_map_thr = c_map_thr

    def load_ESIs(self, samples, shuffle: bool = True):
        np.random.seed(666)
        if shuffle:
            np.random.shuffle(samples)
        self.samples = samples

    def get_order(self, samples, idx):
        e3 = samples[idx, 0]
        sub = samples[idx, 1]
        e3_O = np.array(self.e3_features[e3 + "/O"], dtype=np.float32)
        sub_split = list(self.sub_features[sub].keys())
        sub_O_num = int(len(sub_split) / 2)
        sub_O_list = list()
        for s in range(1, sub_O_num + 1):
            sub_O_list.append(np.array(self.sub_features[sub + "/O" + str(s)]))
        sub_O = np.concatenate(sub_O_list)

        return e3_O, sub_O

    def get_seq(self, samples, idx):
        e3 = samples[idx, 0]
        sub = samples[idx, 1]
        e3_seq = str(np.array(self.e3_features[e3 + "/S"])).split("'")[1]
        sub_seq = str(np.array(self.sub_features[sub + "/S"])).split("'")[1]

        return e3_seq, sub_seq

    def create_band_matrix(self, e3_A):
        N = e3_A.shape[0]
        band_matrix = np.zeros((N, N), dtype=np.float32)
        np.fill_diagonal(band_matrix, 1)
        band_matrix[np.arange(N - 1), np.arange(1, N)] = 1
        band_matrix[np.arange(1, N), np.arange(N - 1)] = 1

        return band_matrix

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        e3 = self.samples[idx, 0]
        sub = self.samples[idx, 1]
        label = self.samples[idx, 2]

        e3_E = torch.from_numpy(np.array(self.e3_features[e3 + "/E"], dtype=np.float32))
        e3_A = np.array(self.e3_features[e3 + "/A"], dtype=np.float32)
        e3_A = self.create_band_matrix(e3_A)

        tmp_coo = sp.coo_matrix(e3_A)
        e3_A_indices = np.vstack((tmp_coo.row, tmp_coo.col))
        e3_A = torch.LongTensor(e3_A_indices)  # torch.LongTensor

        sub_split = list(self.sub_features[sub].keys())
        sub_E_num = int(len(sub_split) / 2)
        sub_E_list = list()
        for s in range(1, sub_E_num + 1):
            sub_E_list.append(torch.from_numpy(np.array(self.sub_features[sub + "/" + str(s)], dtype=np.float32)))
        label = torch.from_numpy(np.array(label, dtype=np.float32))

        return e3_E, e3_A, sub_E_list, label



class ESIfeature_wo_GARD(Dataset):

    """
    Variant of ESIfeature with disabled GARD (Graph-Aware Residue Depth) information.

    """

    def __init__(self, e3_features, sub_features, c_map_thr):
        self.e3_features = e3_features
        self.sub_features = sub_features
        self.c_map_thr = c_map_thr

    def load_ESIs(self, samples, shuffle: bool = True):
        np.random.seed(666)
        if shuffle:
            np.random.shuffle(samples)
        self.samples = samples

    def get_order(self, samples, idx):
        e3 = samples[idx, 0]
        sub = samples[idx, 1]
        e3_E = np.array(self.e3_features[e3 + "/E"], dtype=np.float32)
        sub_E = np.array(self.sub_features[sub + "/E"], dtype=np.float32)

        e3_O = np.arange(1, e3_E.shape[0] + 1)
        sub_O = np.arange(1, sub_E.shape[0] + 1)

        return e3_O, sub_O

    def get_seq(self, samples, idx):
        e3 = samples[idx, 0]
        sub = samples[idx, 1]
        e3_seq = str(np.array(self.e3_features[e3 + "/S"])).split("'")[1]
        sub_seq = str(np.array(self.sub_features[sub + "/S"])).split("'")[1]

        return e3_seq, sub_seq

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        e3 = self.samples[idx, 0]
        sub = self.samples[idx, 1]
        label = self.samples[idx, 2]

        e3_E = torch.from_numpy(np.array(self.e3_features[e3 + "/E"], dtype=np.float32))
        e3_A = np.array(self.e3_features[e3 + "/A"], dtype=np.float32)
        e3_A = np.double(e3_A < self.c_map_thr)
        tmp_coo = sp.coo_matrix(e3_A)
        e3_A_indices = np.vstack((tmp_coo.row, tmp_coo.col))
        e3_A = torch.LongTensor(e3_A_indices)  # torch.LongTensor

        sub_E_list = list()
        sub_E_list.append(torch.from_numpy(np.array(self.sub_features[sub + "/E"], dtype=np.float32)))

        label = torch.from_numpy(np.array(label, dtype=np.float32))

        return e3_E, e3_A, sub_E_list, label
