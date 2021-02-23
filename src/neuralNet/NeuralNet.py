# XNet Model with Pytorch Lighting Structure

import pytorch_lightning as pl
import torch
from torch import nn


class NeuralNetwork(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.learning_rate = 5e-4
        self.batch_size = None
        # TODO: Add dropout layers
        # [batch_size, h, w, no_channels_landsat] -> A

        self.Rx = nn.Sequential(
            nn.Conv2d(11, 100, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(100, 50, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(50, 20, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3)
        )

        self.Py = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(100, 50, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(50, 20, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3)
        )

        self.Sz = nn.Sequential(
            nn.ConvTranspose2d(20, 100, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(100, 200, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(200, 3, kernel_size=3),
        )

        self.Qz = nn.Sequential(
            nn.ConvTranspose2d(20, 100, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(100, 200, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(200, 11, kernel_size=3),
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.Flatten(),
            nn.Linear(1600, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs):
        x, y = inputs
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        code_x = self.Rx(x)
        code_y = self.Py(y)
        x_hat = self.Qz(code_y)
        y_hat = self.Sz(code_x)

        return x_hat, y_hat

    # TODO: custom loss function
    # set automatic_optimization=False in Trainer
    # https://pytorch-lightning.readthedocs.io/en/stable/optimizers.html
    def training_step(self, test_batch, batch_idx, optimizer_idx):
        x, y, prior_information = test_batch
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        if optimizer_idx == 0:
            pass
        if optimizer_idx == 1:
            pass
        # if optimizer_idx == 2:
        #     pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        encoders_params = list(self.Rx.parameters()) + list(self.Py.parameters())
        decoders_params = list(self.Sz.parameters()) + list(self.Qz.parameters())

        optimizer_AE = torch.optim.Adam(encoders_params + decoders_params, lr=self.learning_rate)
        # optimizer_E = torch.optim.Adam(encoders_params, lr=self.learning_rate)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        # return [optimizer_AE, optimizer_disc, optimizer_E]
        return [optimizer_AE, optimizer_disc]
