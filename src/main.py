from src.dataPrepare import process_image
from matplotlib import pyplot as plt
from neuralNet.XNet import XNet
import torch
import prior_computation
import numpy as np

if __name__ == '__main__':
    xnet = XNet()

    patch_size = 10
    window_step = 2

    cuda = torch.device('cuda')

    roi, t1, t2 = prior_computation.load_mat_file()
    patches: torch.tensor = prior_computation.patching_image(t2, patch_size, window_step)
    patches = torch.reshape(patches, (-1, patch_size, patch_size, 3))

    sample_batch = patches[0:10]
    affinity_matrix = prior_computation.compute_affinity_matrix(sample_batch)

    for i in range(10):
        plt.imshow(affinity_matrix[i]), plt.show()
