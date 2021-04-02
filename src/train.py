from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer

from neuralNet.NeuralNet import NeuralNetwork
from utils.Datamodule import CaliforniaFloodDataModule

if __name__ == '__main__':
    datamodule = CaliforniaFloodDataModule()
    model = NeuralNetwork()

    trainer = Trainer(gpus=1, max_epochs=100,
                      logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False),
                      )

    trainer.fit(model, datamodule=datamodule)
