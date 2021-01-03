from scipy.io import loadmat
from skimage.util import view_as_blocks, view_as_windows
from patchify import patchify, unpatchify
import numpy as np
import torch


def load_mat_file(path='resources/Flood_UiT_HCD_California_2017_Luppino.mat'):
    mat = loadmat(path)
    roi = mat['ROI']
    t1_image = mat['t1_L8_clipped']
    t2_image = mat['logt2_clipped']

    return roi, t1_image, t2_image


def patching_image(image: np.ndarray, patch_size=100, window_step=1) -> torch.tensor:
    channels = image.shape[2]
    patches_array = torch.as_tensor(
        view_as_windows(image, (patch_size, patch_size, channels), step=window_step).squeeze())

    return patches_array


def compute_affinity_matrix(batch_patches: torch.tensor) -> torch.tensor:
    """
    Function that computes the affinity matrix for every patch in a batch
    :param batch_patches: tensor with shape (batch_size, patch_width, patch_height, no_of_channels)
    :return:
    """
    _, h, w, c = batch_patches.shape
    x1: torch.tensor = batch_patches.reshape(-1, h * w, c).unsqueeze(1)
    x2: torch.tensor = batch_patches.reshape(-1, h * w, c).unsqueeze(2)
    diff = x2 - x1
    print(diff.shape)
    affinity_matrix = torch.linalg.norm(x2 - x1, dim=-1)
    print(affinity_matrix.shape)
    # TODO continue computing affinity matrix
    # HINT -> find a way to determine 'h', gaussian kernel
    return affinity_matrix
