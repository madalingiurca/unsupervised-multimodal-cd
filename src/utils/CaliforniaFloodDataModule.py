import os
from typing import Union, List

from pytorch_lightning import LightningDataModule
from scipy.io import loadmat
from torch.utils.data import DataLoader
from skimage.util import view_as_blocks

from src.utils import prior_computation
from src.utils.Dataset import CaliforniaFloodDataset, CaliforniaTestDataset


class CaliforniaFloodDataModule(LightningDataModule):
    def __init__(self, data_path="resources/Flood_UiT_HCD_California_2017_Luppino.mat",
                 batch_size=16, patch_size=20, window_step=10):
        super().__init__()
        self.trainDataset = None
        self.testDataset = None
        self.batch_size = batch_size
        self.data_path = data_path
        self.patch_size = patch_size
        self.window_step = window_step
        self.data = {}

    def prepare_data(self, *args, **kwargs):
        if not os.path.exists('resources/alpha_prior_image.png'):
            prior_computation.save_priorinfo_image(patch_size=self.patch_size, window_step=self.window_step)

    def setup(self, stage=None):
        matFile = loadmat(self.data_path)
        data = dict()

        data['landsat'] = matFile['t1_L8_clipped']
        # data['landsat'] = data['landsat'].reshape((-1, self.patch_size, self.patch_size, 11))
        data['landsat'] = view_as_blocks(data['landsat'], (self.patch_size, self.patch_size, 11)).reshape((-1, self.patch_size, self.patch_size, 11))
        data['sentinel'] = matFile['logt2_clipped']
        # data['sentinel'] = data['sentinel'].view().reshape((-1, self.patch_size, self.patch_size, 3))
        data['sentinel'] = view_as_blocks(data['sentinel'], (self.patch_size, self.patch_size, 3)).reshape((-1, self.patch_size, self.patch_size, 3))

        prior_information_image = loadmat(r'resources/alpha_prior_test.mat')['prior']
        prior_information_image = prior_information_image / prior_information_image.max()

        data['prior_info'] = prior_information_image
        # data['prior_info'] = data['prior_info'].view().reshape((-1, self.patch_size, self.patch_size))
        data['prior_info'] = view_as_blocks(data['prior_info'], (self.patch_size, self.patch_size)).reshape((-1, self.patch_size, self.patch_size))
        self.trainDataset = CaliforniaFloodDataset(data)

        data['roi'] = matFile['ROI']
        # data['roi'] = data['roi'].view().reshape((-1, self.patch_size, self.patch_size))
        data['roi'] = view_as_blocks(data['roi'], (self.patch_size, self.patch_size)).reshape((-1, self.patch_size, self.patch_size))

        self.testDataset = CaliforniaTestDataset(data)

        return data

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.trainDataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.testDataset, batch_size=self.batch_size)
