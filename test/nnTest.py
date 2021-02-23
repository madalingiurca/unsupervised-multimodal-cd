import numpy as np
from src.neuralNet.NeuralNet import NeuralNetwork
from src import prior_computation
from patchify import patchify, unpatchify
import torch

if __name__ == '__main__':

    patch_size = 10
    window_step = 4
    batch_size = 5

    roi, t1_landsat, t2_sentinel = prior_computation.load_mat_file(path=r'D:\UnsupervisedMultimodalCD\resources'
                                                                        r'\Flood_UiT_HCD_California_2017_Luppino.mat')
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

    neuralNetwork = NeuralNetwork()
    neuralNetwork.configure_optimizers()
    inputs = (t1_patches[0].unsqueeze(0), t2_patches[0].unsqueeze(0))
    x_hat: torch.Tensor = neuralNetwork.forward(inputs)
    x_hat = x_hat.detach().to_numpy()[0]

