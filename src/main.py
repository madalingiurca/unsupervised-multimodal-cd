from src.dataPrepare import process_image
from matplotlib import pyplot as plt
from neuralNet.XNet import XNet
import torch
import prior_computation
import numpy as np

if __name__ == '__main__':
    xnet = XNet()

    #TODO #Ask_about Patch size and window_step
    patch_size = 10
    window_step = 50

    cuda = torch.device('cuda')

    roi, t1_landsat, t2_sentinel = prior_computation.load_mat_file()
    t1_patches: torch.tensor = prior_computation.patching_image(t1_landsat, patch_size, window_step)
    t1_patches = torch.reshape(t1_patches, (-1, patch_size, patch_size, 11))

    sample_t1_batch = t1_patches[0:2]
    del t1_patches

    t2_patches: torch.tensor = prior_computation.patching_image(t2_sentinel, patch_size, window_step)
    t2_patches = torch.reshape(t2_patches, (-1, patch_size, patch_size, 3))

    sample_t2_batch = t2_patches[0:2]
    del t2_patches

    torch.cuda.empty_cache()


    affinity_matrix_t1 = prior_computation.compute_affinity_matrix(sample_t1_batch)[0]
    affinity_matrix_t2 = prior_computation.compute_affinity_matrix(sample_t2_batch)[0]
    plt.imshow(affinity_matrix_t1), plt.show()
    plt.imshow(affinity_matrix_t2), plt.show()

    test = prior_computation.alpha(sample_t1_batch, sample_t2_batch);
    plt.imshow(), plt.show()

    # D = affinity_matrix_t2 - affinity_matrix_t1
    # plt.imshow(D), plt.show()

    # affinity_matrix_t2 = prior_computation.compute_affinity_matrix(sample_t2_batch)
    #
    # for i in range(2):
    #     plt.imshow(affinity_matrix_t1.cpu()[i]), plt.show()
    #     plt.imshow(affinity_matrix_t2.cpu()[i]), plt.show()
