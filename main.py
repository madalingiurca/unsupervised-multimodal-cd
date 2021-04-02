import numpy as np
import torch
from matplotlib import pyplot as plt
from patchify import unpatchify, patchify

from src.utils import prior_computation

if __name__ == '__main__':
    # TODO #Ask_about Patch size and window_step
    patch_size = 10
    window_step = 10

    cuda = torch.device('cuda')

    roi, t1_landsat, t2_sentinel = prior_computation.load_mat_file()
    # eval_prior('test', t1_landsat, t2_sentinel)
    # exit(1)

    # t1_patches: torch.tensor = prior_computation.patching_image(t1_landsat, patch_size, window_step).to(cuda)
    # t1_patches = torch.reshape(t1_patches, (-1, patch_size, patch_size, 11))
    #
    # sample_t1_batch = t1_patches[100:102]
    # del t1_patches
    #
    # t2_patches: torch.tensor = prior_computation.patching_image(t2_sentinel, patch_size, window_step).to(cuda)
    # t2_patches = torch.reshape(t2_patches, (-1, patch_size, patch_size, 3))

    t1_patches = patchify(t1_landsat, (10, 10, 11), 10)
    t1_patches = torch.tensor(np.reshape(t1_patches, (-1, 10, 10, 11))).to(cuda)

    t2_patches = patchify(t2_sentinel, (10, 10, 3), 10)
    t2_patches = torch.tensor(np.reshape(t2_patches, (-1, 10, 10, 3))).to(cuda)

    t2_patches = t2_patches.detach().numpy().astype(np.float32)
    t2_image = np.reshape(t2_patches, (350, 200, 1, 10, 10, 3))

    t2_image = unpatchify(t2_image, (3500, 2000, 3))

    plt.imshow(t2_sentinel), plt.show()
    plt.imshow(t2_image), plt.show()
    exit(1)

    sample_t2_batch = t2_patches[100:102]
    del t2_patches

    torch.cuda.empty_cache()

    # affinity_matrix_t1 = prior_computation.compute_affinity_matrix(sample_t1_batch)[0]
    # affinity_matrix_t2 = prior_computation.compute_affinity_matrix(sample_t2_batch)[0]
    # plt.imshow(affinity_matrix_t1), plt.show()
    # plt.imshow(affinity_matrix_t2), plt.show()

    plt.imshow(sample_t1_batch[0][:, :, 4]), plt.show()
    plt.imshow(sample_t2_batch[0][:, :, :]), plt.show()

    test = prior_computation.alpha_prior(sample_t1_batch, sample_t2_batch)
    plt.imshow(test[0]), plt.show()

    # D = affinity_matrix_t2 - affinity_matrix_t1
    # plt.imshow(D), plt.show()

    # affinity_matrix_t2 = prior_computation.compute_affinity_matrix(sample_t2_batch)
    #
    # for i in range(2):
    #     plt.imshow(affinity_matrix_t1.cpu()[i]), plt.show()
    #     plt.imshow(affinity_matrix_t2.cpu()[i]), plt.show()
