import numpy as np

from scipy.stats import wasserstein_distance
import cv2
from cv2 import EMD

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from metaesi import im_completion

# === Helper Functions ===
# Utility functions for data transformation and matrix operations
def convert_matrix_to_emd_signature(arr):
    """Convert a 2D array to a signature format required by cv2.EMD.

    Args:
        arr (np.ndarray): Input 2D array with non-zero values representing mass

    Returns:
        np.ndarray: Signature array in format [[mass, x, y], ...] with float32 dtype
    """
    sig = list()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] != 0:
                sig.append(np.array([arr[i, j], i, j]))
    sig = np.vstack(sig)
    sig = np.array(sig, dtype=np.float32)
    return sig


def shuffle_matrix(im):
    """Randomly shuffle values in a 2D matrix while maintaining shape.

    Args:
        im (np.ndarray): Input interface map matrix

    Returns:
        np.ndarray: Shuffled matrix with same shape as input
    """
    m, n = im.shape
    im = im.reshape([-1, 1])
    np.random.shuffle(im)
    im = np.reshape(im, [m, n])

    return im


def shuffle_matrix_along_axis(im, a):
    """Shuffle matrix values along specified axis.

    Args:
        im (np.ndarray): Input interface map matrix
        axis (int): 0 for column-wise shuffle(SUB), 1 for row-wise shuffle(E3)

    Returns:
        np.ndarray: Shuffled matrix with same shape as input
    """
    if a == 1:
        np.random.shuffle(im)
    else:
        im = im.T
        np.random.shuffle(im)
        im = im.T

    return im

# === EMD Calculation Functions ===
# Functions for computing Earth Mover's Distance metrics
def calculate_normalized_emd_2d(im, im_val, e3_len, sub_len, random=False, completion=False, *args):
    """Calculate normalized EMD distance between two 2D matrices.

    Args:
        im (np.ndarray): Input interface map matrix
        im_val (np.ndarray): Ground truth interface map
        e3_len (int): Length of E3 ligase sequence
        sub_len (int): Length of substrate sequence
        random (bool): Whether to apply random shuffling
        completion (bool): Whether to use matrix completion
        *args: Additional parameters for matrix completion

    Returns:
        float: Normalized 2D EMD distance between [0,1]
    """

    im[im < im.max() - 0.05] = 0

    if random:
        im = shuffle_matrix(im)

    if completion:
        order1, order2 = args
        im = im_completion(e3_len, sub_len, order1, order2, im)

    im = im / im.sum()
    im_val = im_val / im_val.sum()

    im_sig = convert_matrix_to_emd_signature(im)
    im_val_sig = convert_matrix_to_emd_signature(im_val)
    dist, _, flow = EMD(im_sig, im_val_sig, cv2.DIST_L2)

    max_dist = ((e3_len-1)**2 + (sub_len-1)**2)**(1/2)

    return dist/max_dist


def calculate_normalized_emd_1d(im, im_val, a, random=False, completion=False, *args):
    """Calculate 1D EMD distance along specified axis.

    Args:
        im (np.ndarray): Input interface map matrix
        im_val (np.ndarray): Ground truth interface map
        axis (int): 0 for column-wise (SUB), 1 for row-wise (E3) calculation
        random (bool): Whether to apply random shuffling
        completion (bool): Whether to use matrix completion
        *args: Additional parameters for matrix completion

    Returns:
        float: Normalized 1D EMD distance between [0,1]
    """

    im[im < im.max() - 0.05] = 0

    if random:
        im = shuffle_matrix_along_axis(im, a)

    if completion:
        e3_len, sub_len, order1, order2 = args
        im = im_completion(e3_len, sub_len, order1, order2, im)

    im = np.max(im, a)
    im = np.array(im, dtype=np.float32)
    im_val = np.max(im_val, a)
    im_val = np.array(im_val, dtype=np.float32)

    im = im / im.sum()
    im_val = im_val / im_val.sum()

    dists = [i for i in range(len(im))]
    dist = wasserstein_distance(dists, dists, im, im_val)/(len(im)-1)

    return dist