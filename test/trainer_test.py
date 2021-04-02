from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer

from src.neuralNet.NeuralNet import NeuralNetwork
from src.utils.Datamodule import CaliforniaFloodDataModule

if __name__ == '__main__':
    datamodule = CaliforniaFloodDataModule()
    model = NeuralNetwork()

    trainer = Trainer(gpus=1, max_epochs=100,
                      logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False),
                      # auto_lr_find=True,
                      # auto_scale_batch_size=True
                      )

    trainer.fit(model, datamodule=datamodule)
