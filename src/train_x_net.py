from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer

from src.neuralNetwork.x_net import XNet
from utils.CaliforniaFloodDataModule import CaliforniaFloodDataModule

if __name__ == '__main__':
    datamodule = CaliforniaFloodDataModule()
    model = XNet()

    trainer = Trainer(gpus=1, max_epochs=10,
                      logger=TensorBoardLogger(save_dir='lightning_logs/X-NET', default_hp_metric=False),
                      # auto_lr_find=True,
                      # auto_scale_batch_size=True,
                      log_every_n_steps=10,
                      flush_logs_every_n_steps=50
                      )

    trainer.fit(model, datamodule=datamodule)
