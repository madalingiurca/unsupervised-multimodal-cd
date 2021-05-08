from torch.utils.data import Dataset


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


class CaliforniaTestDataset(Dataset):
    def __init__(self, data: dict):
        assert data['landsat'].shape[0] == data['sentinel'].shape[0] == data['roi'].shape[0]

        self.ground_truth = data['roi']
        self.landsat_patches = data['landsat']
        self.sentinel_patches = data['sentinel']

        self.no_samples = data['landsat'].shape[0]

    def __getitem__(self, item):
        return self.landsat_patches[item], self.sentinel_patches[item], self.ground_truth[item]

    def __len__(self):
        return self.no_samples
