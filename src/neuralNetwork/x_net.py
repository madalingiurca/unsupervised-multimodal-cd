import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import mse_loss

import src.neuralNetwork.hyperparams as hp
from src.neuralNetwork.hyperparams import *


class XNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.learning_rate = LEARNING_RATE
        # self.automatic_optimization = False
        self.batch_size = BATCH_SIZE
        # self.mse = tf.keras.losses.MeanSquaredError()
        # TODO: Add dropout layers
        # [batch_size, h, w, no_channels_landsat] -> A

        self.Rx = nn.Sequential(
            nn.Conv2d(11, 100, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(100, 50, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(50, 20, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(20, 3, kernel_size=3, padding=(4, 4))
        )

        self.Py = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(100, 50, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(50, 20, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(20, 11, kernel_size=3, padding=(4, 4))
        )

    def forward(self, input):
        x, y = input
        x_trans = self.Rx(x)
        y_trans = self.Py(y)

        return x_trans, y_trans

    def training_step(self, test_batch, batch_idx):
        x, y, prior_information = test_batch
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        x_trans, y_trans = self((x, y))

        x_cycled = self.Py(y_trans)
        y_cycled = self.Rx(x_trans)

        prior_loss = self.mse_loss_weighted(x, x_trans, 1 - prior_information) + \
                     self.mse_loss_weighted(y, y_trans, 1 - prior_information)

        cycle_loss = mse_loss(x, x_cycled) + mse_loss(y, y_cycled)

        total_loss = hp.W_HAT * prior_loss + hp.W_CYCLE * cycle_loss
        self.log("total loss", total_loss)

        return hp.W_HAT * prior_loss + hp.W_CYCLE * cycle_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=hp.W_REG)
        return optimizer

    @staticmethod
    def mse_loss_weighted(x, x_hat, weights):
        L2_Norm = torch.linalg.norm(x_hat - x, dim=1) ** 2
        # L2_Norm = ((x_hat - x) ** 2).sum(1)
        weighted_L2_norm: torch.Tensor = L2_Norm * weights
        loss = weighted_L2_norm.mean()
        return loss
