from pytorch_lightning.trainer import Trainer

from src.utils.Datamodule import CaliforniaFloodDataModule
from src.neuralNet.NeuralNet import NeuralNetwork

if __name__ == '__main__':
    datamodule = CaliforniaFloodDataModule(batch_size=8)
    model = NeuralNetwork()

    trainer = Trainer(gpus=1, max_epochs=100)

    trainer.fit(model, datamodule=datamodule)
