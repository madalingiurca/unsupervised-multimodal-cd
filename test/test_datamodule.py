from src.utils.CaliforniaFloodDataModule import CaliforniaFloodDataModule
from torch.utils.data import DataLoader

if __name__ == '__main__':
    dm = CaliforniaFloodDataModule()
    dm.prepare_data()
    dl: DataLoader = dm.train_dataloader()
    for i in dl:
        print(i)
