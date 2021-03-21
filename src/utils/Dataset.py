from torch.utils.data import Dataset
from torch import tensor

import numpy as np


class CaliforniaFloodDataset(Dataset):
    def __init__(self, data: dict):
        assert data['landsat'].shape[0] == data['sentinel'].shape[0] == data['prior_info'].shape[0]

        self.prior_information_patches = data['prior_info']
        self.landsat_patches = data['landsat']
        self.sentinel_patches = data['sentinel']

        self.no_samples = data['landsat'].shape[0]

    def __getitem__(self, item):
        return self.landsat_patches[item], self.sentinel_patches[item], self.prior_information_patches[item]

    def __len__(self):
        return self.no_samples
