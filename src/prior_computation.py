from scipy.io import loadmat
from skimage.util import view_as_blocks, view_as_windows
from patchify import patchify, unpatchify
import numpy as np
import torch
import tensorflow as tf


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
    affinity_matrix = torch.linalg.norm(x2 - x1, dim=-1)

    # TODO #ask_about that ***** kernel, RBF
    # Kernel determined by the same method used in the git repo
    # HINT -> find a way to determine 'h', gaussian kernel

    kernel = torch.topk(affinity_matrix, h * w).values
    kernel = torch.mean(kernel[:, :, (h * w) // 4], dim=1)
    kernel = torch.reshape(kernel, (-1, 1, 1))
    affinity_matrix = torch.exp(-(torch.divide(affinity_matrix, kernel) ** 2))
    return affinity_matrix


def alpha(x, y):
    """
    Compute the alpha prior starting from corresponding patches of data organized
    in batches. It first computes the affinity matrices of the two batches and then
    it computes the mean over the rows of the element-wise difference between the
    affinity matrices.
        Input:
            x - float, array of [batch_size, patch_size, patch_size, num_channels_x],
                Batch of patches from data domain x.
            y - float, array of [batch_size, patch_size, patch_size, num_channels_y],
                Batch of patches from data domain y.
        Output:
            alpha - float, array of [batch_size, patch_size, patch_size], prior
    """
    Ax = compute_affinity_matrix(x)
    Ay = compute_affinity_matrix(y)
    ps = int(Ax.shape[1] ** (0.5))
    alpha = tf.reshape(tf.reduce_mean(tf.abs(Ax - Ay), axis=-1), [-1, ps, ps])
    return alpha
