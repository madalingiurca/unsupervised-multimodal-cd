# XNet Model with Pytorch Lighting Structure
from typing import Any

import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy


class XNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.learning_rate = 5e-4
        self.batch_size = None

        self.CNNLayerX = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=6, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 18, kernel_size=6, padding=1),
            nn.ReLU(),
            nn.Conv2d(18, 11, kernel_size=6, padding=1)
        )

        self.CNNLayerY = nn.Sequential(
            nn.Conv2d(11, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=6, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=6, padding=1),
        )

    def forward(self, x):
        return self.CNNLayerY(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.75)
        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
