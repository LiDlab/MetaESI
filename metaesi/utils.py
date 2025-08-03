import torch
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import average_precision_score, roc_auc_score


def try_gpu(i=0):
    """
    Get available GPU device with specified index, fallback to CPU if unavailable.

    Parameters:
        i (int): Index of GPU device to use

    Returns:
        torch.device: Available CUDA device if exists, otherwise CPU device
    """

    if i == -1:
        return torch.device('cpu')

    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')



def kfold(k, sample_num):
    """
    Generate k-fold cross-validation indices splits.

    Parameters:
        k (int): Number of folds
        sample_num (int): Total number of samples

    Returns:
        list: List of k tuples containing (train_indices, test_indices) pairs
    """

    cv = []
    split = []

    for j in range(k):
        cv.append([i for i in range(sample_num) if i % k == j])

    for j in range(k):
        cv_copy = copy.deepcopy(cv)
        test = cv_copy.pop(j)
        train = [element for list in cv_copy for element in list]
        train.sort()
        split.append((train, test))
    return split



def save_logits(all_pred, path, item, filename):
    """
    Save prediction logits to CSV file.

    Parameters:
        all_pred (array-like): Prediction data to save
        path (str): Directory path for saving
        item (str): Column name containing prediction method
        filename (str): Output filename

    Returns:
        pd.DataFrame: Saved DataFrame containing predictions
    """

    pred = pd.DataFrame(all_pred, columns=["E3", "Substrate", "label", item])
    pred.to_csv(path + item + "_" + filename, index = False)

    return pred



def evaluate_logits(file, item):
    """
    Calculate evaluation metrics (AUC and AUPR) from logit file.

    Parameters:
        file (pd.DataFrame): DataFrame containing labels and predictions
        item (str): Column name containing prediction method

    Returns:
        tuple: (AUC score, AUPR score)
    """

    auc = roc_auc_score(np.array(file["label"],dtype=np.float32), np.array(file[item],dtype=np.float32))
    aupr = average_precision_score(np.array(file["label"],dtype=np.float32), np.array(file[item],dtype=np.float32))

    return auc, aupr



def extract_interface_residues(i_map, p_hat, order1, order2, threshold = 0.05):
    """
    Extract interface residues from interface map based on threshold.

    Parameters:
        i_map (np.ndarray): Interface map matrix
        p_hat (float): Probability threshold cutoff
        order1 (array-like): Mapping indices for first dimension (E3)
        order2 (array-like): Mapping indices for second dimension (Substrate)
        threshold (float): Value threshold for residue selection (default: 0.05)

    Returns:
        tuple:
            - pred1 (list): Sorted unique residue indices from first dimension (top 20)
            - pred2 (list): Sorted unique residue indices from second dimension (top 10)
    """

    pred1_index, pred2_index = np.where(i_map > float(p_hat) - threshold)
    values_with_indices = np.column_stack((i_map[pred1_index, pred2_index], pred1_index, pred2_index))
    sorted_indices = np.argsort(values_with_indices[:, 0])[::-1]
    sorted_values_with_indices = values_with_indices[sorted_indices]

    unique_pred1_indices = []
    unique_pred2_indices = []
    for value, ind1, ind2 in sorted_values_with_indices:
        if len(unique_pred1_indices) < 20 and ind1 not in unique_pred1_indices:
            unique_pred1_indices.append(int(ind1))
        if len(unique_pred2_indices) < 10 and ind2 not in unique_pred2_indices:
            unique_pred2_indices.append(int(ind2))
        if len(unique_pred1_indices) == 20 and len(unique_pred2_indices) == 10:
            break

    pred1 = [order1[i] for i in unique_pred1_indices]
    pred2 = [order2[j] for j in unique_pred2_indices]

    pred1 = sorted(pred1)
    pred2 = sorted(pred2)

    return pred1, pred2



def im_completion(e3_len, sub_len, order1, order2, im):
    """
    Reconstruct complete interface map.

    Parameters:
        e3_len (int): Length of E3 protein sequence
        sub_len (int): Length of substrate protein sequence
        order1 (array-like): E3 residue indices mapping
        order2 (array-like): Substrate residue indices mapping
        im (np.ndarray): Interface map

    Returns:
        np.ndarray: Complete interface map with shape (e3_len, sub_len)
    """

    im_metaesi = np.zeros([e3_len, sub_len])
    for i in range(len(order1)):
        e3_index = order1[i] - 1
        im_metaesi[int(e3_index), (order2 - np.ones(len(order2))).astype('int64')] = im[i, :]

    return im_metaesi

