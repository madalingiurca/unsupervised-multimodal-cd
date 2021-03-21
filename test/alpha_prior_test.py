import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import trange
from patchify import unpatchify, patchify

from src.utils import prior_computation

if __name__ == '__main__':
    cuda = torch.device('cuda')

    patch_size = 10
    window_step = 10
    batch_size = 5

    roi, t1_landsat, t2_sentinel = prior_computation.load_mat_file(
        path=r'resources/Flood_UiT_HCD_California_2017_Luppino.mat')
    # t1_patches: torch.tensor = prior_computation.patching_image(t1_landsat, patch_size, window_step).to(cuda)
    # t1_patches = torch.reshape(t1_patches, (-1, patch_size, patch_size, 11))
    # t2_patches: torch.tensor = prior_computation.patching_image(t2_sentinel, patch_size, window_step).to(cuda)
    # t2_patches = torch.reshape(t2_patches, (-1, patch_size, patch_size, 3))

    t1_patches = patchify(t1_landsat, (patch_size, patch_size, 11), window_step)
    h, w, _, _, _, _ = t1_patches.shape
    t1_patches = torch.tensor(np.reshape(t1_patches, (-1, patch_size, patch_size, 11)))

    t2_patches = patchify(t2_sentinel, (patch_size, patch_size, 3), window_step)
    t2_patched_size = t2_patches.shape
    t2_patches = torch.tensor(np.reshape(t2_patches, (-1, patch_size, patch_size, 3)))

    assert t1_patches.shape[0] == t2_patches.shape[0]

    full_image_alpha_prior = torch.zeros([t1_patches.shape[0], patch_size, patch_size]).to(cuda)

    for patch_idx in trange(0, t1_patches.shape[0], batch_size):
        var = prior_computation.alpha_prior(t1_patches[patch_idx:patch_idx + batch_size].to(cuda),
                                            t2_patches[patch_idx:patch_idx + batch_size].to(cuda))
        # if patch_idx % 1000 == 0:
        #     for patch in range(batch_size):
        #         plt.imshow(var[patch].cpu()), plt.show()

        full_image_alpha_prior[patch_idx:patch_idx + batch_size] = var

    full_image_alpha_prior = full_image_alpha_prior.cpu().detach().numpy()
    full_image_alpha_prior = unpatchify(np.reshape(full_image_alpha_prior,
                                                   (h, w,
                                                    patch_size, patch_size)),
                                        (3500, 2000))
    # plt.imshow(full_image_alpha_prior), plt.show()
    save_image(torch.tensor(full_image_alpha_prior), 'alpha_prior_test.png')
