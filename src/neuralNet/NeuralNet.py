# XNet Model with Pytorch Lighting Structure

import pytorch_lightning as pl
import torch
from torch import nn

import neuralNet.hyperparams as hp


class NeuralNetwork(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.learning_rate = hp.LEARNING_RATE
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
            nn.Conv2d(20, 64, kernel_size=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(64, 32, kernel_size=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(32, 16, kernel_size=2),
            nn.Flatten(),
            nn.Linear(1936, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs):
        # TODO: Compute the absolute difference between traslated images
        x, y = inputs
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        code_x = self.Rx(x)
        code_y = self.Py(y)
        x_hat = self.Qz(code_y)
        y_hat = self.Sz(code_x)

        return x_hat, y_hat

    # set automatic_optimization=False in Trainer if iterable optimizers doesnt work
    # this implies explicit care of backpropagation, zero_grad, etc
    # https://pytorch-lightning.readthedocs.io/en/stable/optimizers.html
    # noinspection DuplicatedCode
    def training_step(self, test_batch, batch_idx, optimizer_idx):
        x, y, prior_information = test_batch
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        # optimizer 0, optimize with respect to AE params based on returned loss
        if optimizer_idx == 0:
            x_hat = self.Qz(self.Py(y))
            y_hat = self.Sz(self.Rx(x))
            x_tilda = self.Qz(self.Rx(x))
            y_tilda = self.Sz(self.Py(y))
            x_cycled = self.Qz(self.Py(y_hat))
            y_cycled = self.Sz(self.Rx(x_hat))

            cycle_loss = self.mse_loss(x, x_cycled, y, y_cycled)
            prior_loss = self.prior_loss(x, x_hat, y, y_hat, (1 - prior_information))
            recon_loss = self.mse_loss(x, x_tilda, y, y_tilda)
            # TODO: recheck regularisation term | norma params ^ 2?
            self.log('Cycle Loss', cycle_loss)
            self.log('Prior Loss', prior_loss)
            self.log('Reconstruction Loss', recon_loss)

            total_loss = hp.W_CYCLE * cycle_loss + hp.W_HAT * prior_loss + hp.W_RECON * recon_loss
            self.log('Non-discriminator total loss', total_loss)

            return total_loss
        # optimizer 1, optimize with respect to the Discriminator params
        if optimizer_idx == 1:
            x_disc_code = self.Rx(x)
            x_disc_code = self.discriminator(x_disc_code)
            # x_disc_code = self.discriminator(self.Rx(x))
            y_disc_code = self.discriminator(self.Py(y))
            ones = torch.ones_like(x_disc_code)

            # TODO: check if correct pg8 - ARXIV unit tests*
            x_part = torch.sum((x_disc_code - ones) ** 2) / torch.numel(x_disc_code)
            y_part = torch.sum(y_disc_code ** 2) / torch.numel(y_disc_code)
            disc_loss = hp.W_D * (x_part + y_part)

            self.log("Discrimator loss", disc_loss)

            return hp.W_D * (x_part + y_part)
        # optimizer 2, optimize with respect to both encoders params, generator part opposite to optimizer 1
        if optimizer_idx == 2:
            x_disc_code = self.discriminator(self.Rx(x))
            y_disc_code = self.discriminator(self.Py(y))
            ones = torch.ones_like(x_disc_code)

            # TODO: check if correct pg8 - ARXIV unit tests
            x_part = torch.sum(x_disc_code ** 2) / torch.numel(x_disc_code)
            y_part = torch.sum((y_disc_code - ones) ** 2) / torch.numel(y_disc_code)
            code_loss = hp.W_D * (x_part + y_part)

            self.log("Code loss", code_loss)
            return hp.W_D * (x_part + y_part)

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        encoders_params = list(self.Rx.parameters()) + list(self.Py.parameters())
        decoders_params = list(self.Sz.parameters()) + list(self.Qz.parameters())

        optimizer_ae = torch.optim.Adam(encoders_params + decoders_params, lr=self.learning_rate)
        optimizer_e = torch.optim.Adam(encoders_params, lr=self.learning_rate)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        # return [optimizer_ae, optimizer_disc, optimizer_E]
        return [optimizer_ae, optimizer_disc, optimizer_e]

    @staticmethod
    def mse_loss(x, x_mod, y, y_mod):
        x_mse = torch.sum((torch.linalg.norm(x_mod - x, dim=1) ** 2)) / (torch.numel(x))
        y_mse = torch.sum((torch.linalg.norm(y_mod - y, dim=1) ** 2)) / (torch.numel(y))

        return x_mse + y_mse

    @staticmethod
    def prior_loss(x, x_hat, y, y_hat, weights):
        a = torch.linalg.norm(y_hat - y, dim=1) ** 2
        a = a ** 2
        a = a * weights
        a = torch.sum(a) / torch.numel(a)
        x_weighted_mse = torch.sum((torch.linalg.norm(x_hat - x, dim=1) ** 2) * weights) / (torch.numel(x))
        y_weighted_mse = torch.sum((torch.linalg.norm(y_hat - y, dim=1) ** 2) * weights) / (torch.numel(y))

        return y_weighted_mse + x_weighted_mse
