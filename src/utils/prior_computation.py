from patchify import unpatchify, patchify
from scipy.io import loadmat
from skimage.util import view_as_windows
import numpy as np
import torch

# import tensorflow as tf
from torchvision.utils import save_image
from tqdm import trange


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
    :return affinity_matrix: torch.tensor containing the affinity Matrix for every patch in the batch
    """
    _, h, w, c = batch_patches.shape
    x1: torch.tensor = batch_patches.reshape(-1, h * w, c).unsqueeze(1)
    x2: torch.tensor = batch_patches.reshape(-1, h * w, c).unsqueeze(2)
    affinity_matrix = torch.linalg.norm(x2 - x1, dim=-1)

    kernel = torch.topk(affinity_matrix, h * w).values
    kernel = torch.mean(kernel[:, :, (h * w) // 4], dim=1)
    kernel = torch.reshape(kernel, (-1, 1, 1))
    affinity_matrix = torch.exp(-(torch.divide(affinity_matrix, kernel) ** 2))
    return affinity_matrix


def alpha_prior(x_batch, y_batch):
    # TODO: alpha for overlapping patches
    ax = compute_affinity_matrix(x_batch)
    ay = compute_affinity_matrix(y_batch)
    ps = int(ax.shape[1] ** 0.5)
    d_absolute = torch.abs(ax - ay)
    alpha_prior_info: torch.tensor = torch.mean(d_absolute, -1)
    alpha_prior_info = torch.reshape(alpha_prior_info, [-1, ps, ps])
    # alpha = tf.reshape(tf.reduce_mean(tf.abs(ax - ay), axis=-1), [-1, ps, ps])
    return alpha_prior_info


# noinspection DuplicatedCode
def save_priorinfo_image(patch_size=10, window_step=10, batch_size=5):
    cuda = torch.device('cuda')
    roi, t1_landsat, t2_sentinel = load_mat_file(
        path=r'resources/Flood_UiT_HCD_California_2017_Luppino.mat')

    t1_patches = patchify(t1_landsat, (patch_size, patch_size, 11), window_step)
    h, w, _, _, _, _ = t1_patches.shape
    t1_patches = torch.tensor(np.reshape(t1_patches, (-1, patch_size, patch_size, 11)))

    t2_patches = patchify(t2_sentinel, (patch_size, patch_size, 3), window_step)
    t2_patches = torch.tensor(np.reshape(t2_patches, (-1, patch_size, patch_size, 3)))

    assert t1_patches.shape[0] == t2_patches.shape[0]

    full_image_alpha_prior = torch.zeros([t1_patches.shape[0], patch_size, patch_size]).to(cuda)

    for patch_idx in trange(0, t1_patches.shape[0], batch_size):
        var = alpha_prior(t1_patches[patch_idx:patch_idx + batch_size].to(cuda),
                          t2_patches[patch_idx:patch_idx + batch_size].to(cuda))
        full_image_alpha_prior[patch_idx:patch_idx + batch_size] = var

    full_image_alpha_prior = full_image_alpha_prior.cpu().detach().numpy()
    full_image_alpha_prior = unpatchify(np.reshape(full_image_alpha_prior,
                                                   (h, w,
                                                    patch_size, patch_size)),
                                        (3500, 2000))
    save_image(torch.tensor(full_image_alpha_prior), 'resources/alpha_prior_image.png')
