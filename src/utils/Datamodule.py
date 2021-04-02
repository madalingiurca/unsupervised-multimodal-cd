import os

import cv2
import numpy as np
from patchify import patchify
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.neuralNet.hyperparams import BATCH_SIZE
from src.utils import prior_computation
from src.utils.Dataset import CaliforniaFloodDataset


class CaliforniaFloodDataModule(LightningDataModule):
    def __init__(self, data_path="resources/Flood_UiT_HCD_California_2017_Luppino.mat",
                 batch_size=BATCH_SIZE, patch_size=20, window_step=10):
        super().__init__()
        self.trainDataset = None
        self.batch_size = batch_size
        self.data_path = data_path
        self.patch_size = patch_size
        self.window_step = window_step
        self.data = {}

    def prepare_data(self, *args, **kwargs):
        roi, t1_landsat, t2_sentinel = prior_computation.load_mat_file(path=self.data_path)
        self.data['landsat'] = np.reshape(
            patchify(t1_landsat, (self.patch_size, self.patch_size, 11), self.window_step),
            (-1, self.patch_size, self.patch_size, 11))

        self.data['sentinel'] = np.reshape(
            patchify(t2_sentinel, (self.patch_size, self.patch_size, 3), self.window_step),
            (-1, self.patch_size, self.patch_size, 3))

        if not os.path.exists('resources/alpha_prior_image.png'):
            prior_computation.save_priorinfo_image(patch_size=self.patch_size, window_step=self.window_step)

        prior_information_image = cv2.imread('resources/alpha_prior_image.png', cv2.IMREAD_GRAYSCALE)

        self.data['prior_info'] = patchify(prior_information_image,
                                           (self.patch_size, self.patch_size), self.window_step)

        self.data['prior_info'] = np.reshape(
            patchify(prior_information_image, (self.patch_size, self.patch_size), self.window_step),
            (-1, self.patch_size, self.patch_size)
        )

        self.trainDataset = CaliforniaFloodDataset(self.data)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.trainDataset, batch_size=self.batch_size)
