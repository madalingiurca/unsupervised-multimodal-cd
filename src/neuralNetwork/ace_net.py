# XNet Model with Pytorch Lighting Structure

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import mse_loss

from src.neuralNetwork.hyperparams import *


# ACE-NET Model with Pytorch Lighting Structure
class AceNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.learning_rate = LEARNING_RATE
        self.automatic_optimization = False
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
            # nn.Linear(5184, 1),
            nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, y):
        code_x = self.Rx(x)
        code_y = self.Py(y)
        x_hat = self.Qz(code_y)
        y_hat = self.Sz(code_x)

        return x_hat, y_hat

    def training_step(self, test_batch, batch_idx, optimizer_idx):
        x, y, prior_information = test_batch
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        opt_encoder_decoder, opt_encoders, opt_disc = self.configure_optimizers()

        opt_encoder_decoder.zero_grad()
        # Network outputs
        # X data flow
        x_code = self.Rx(x)
        y_hat = self.Sz(x_code)
        x_cycle = self.Qz(self.Py(y_hat))
        x_tilda = self.Qz(x_code)

        # Y data flow
        y_code = self.Py(y)
        x_hat = self.Qz(y_code)
        y_cycle = self.Sz(self.Rx(x_hat))
        y_tilda = self.Sz(y_code)

        # x_disc = self.discriminator(x_code)
        # y_disc = self.discriminator(y_code)

        # X related losses
        x_recon_loss = mse_loss(x, x_tilda)
        x_cycle_loss = mse_loss(x, x_cycle)
        y_hat_loss = self.mse_loss_weighted(x, x_hat, 1 - prior_information)
        # x_disc_loss = torch.linalg.norm((torch.zeros_like(x_disc) - x_disc) ** 2)
        # x_disc_loss = mse_loss(torch.zeros_like(x_disc), x_disc)

        # Y related losses
        y_recon_loss = mse_loss(y, y_tilda)
        y_cycle_loss = mse_loss(y, y_cycle)
        x_hat_loss = self.mse_loss_weighted(y, y_hat, 1 - prior_information)
        # y_hat_loss = self.mse(y.detach().numpy(), y_hat.detach().numpy(), 1 - prior_information.detach().numpy)
        # y_disc_loss = torch.linalg.norm((torch.ones_like(y_disc) - y_disc) ** 2)
        # y_disc_loss = mse_loss(torch.ones_like(y_disc), y_disc)

        disc_out = self.discriminator(torch.cat((x_code, y_code)))
        x_disc, y_disc = torch.tensor_split(disc_out, 2, dim=0)

        x_disc_loss = mse_loss(torch.zeros_like(x_disc), x_disc)
        y_disc_loss = mse_loss(torch.ones_like(y_disc), y_disc)

        total_ae_loss = (
                W_D * (x_disc_loss + y_disc_loss) +
                W_RECON * (x_recon_loss + y_recon_loss) +
                W_HAT * (x_hat_loss + y_hat_loss) +
                W_CYCLE * (x_cycle_loss + y_cycle_loss)
        )

        self.log("Discriminator code loss", W_D * (x_disc_loss + y_disc_loss))
        self.log("Reconstruction loss", W_RECON * (x_recon_loss + y_recon_loss))
        self.log("Prior information loss", W_HAT * (x_hat_loss + y_hat_loss))
        self.log("Total AutoEncoders loss", total_ae_loss, prog_bar=True)

        self.manual_backward(total_ae_loss, opt_encoder_decoder)
        opt_encoder_decoder.step()

        # opt_encoders.zero_grad()

        # x_code = self.Rx(x)
        # y_code = self.Py(y)

        # disc_out = self.discriminator(torch.cat((x_code, y_code)))
        # x_disc, y_disc = torch.tensor_split(disc_out, 2, dim=0)

        # x_disc_loss = mse_loss(torch.zeros_like(x_disc), x_disc)
        # y_disc_loss = mse_loss(torch.ones_like(y_disc), y_disc)

        # disc_code_loss = W_D * (x_disc_loss + y_disc_loss)
        # self.log("Discriminator code loss", disc_code_loss)
        # self.manual_backward(disc_code_loss, opt_encoders)
        # opt_encoders.step()

        opt_disc.zero_grad()

        x_code = self.Rx(x)
        y_code = self.Py(y)

        disc_out = self.discriminator(torch.cat((x_code, y_code)))
        x_disc, y_disc = torch.tensor_split(disc_out, 2, dim=0)

        disc_loss = mse_loss(torch.ones_like(x_disc), x_disc) + mse_loss(torch.zeros_like(y_disc), y_disc)

        self.log("Discriminator loss", disc_loss)

        self.manual_backward(disc_loss, opt_disc)
        opt_disc.step()

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        encoders_params = list(self.Rx.parameters()) + list(self.Py.parameters())
        decoders_params = list(self.Sz.parameters()) + list(self.Qz.parameters())

        optimizer_ae = torch.optim.Adam(encoders_params + decoders_params, lr=self.learning_rate, weight_decay=W_REG)
        optimizer_e = torch.optim.Adam(encoders_params, lr=self.learning_rate, weight_decay=W_REG)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=W_REG)

        return [optimizer_ae, optimizer_e, optimizer_disc]

    @staticmethod
    def mse_loss_weighted(x, x_hat, weights):
        L2_Norm = torch.linalg.norm(x_hat - x, dim=1) ** 2
        # L2_Norm = ((x_hat - x) ** 2).sum(1)
        weighted_L2_norm: torch.Tensor = L2_Norm * weights
        loss = weighted_L2_norm.mean()
        return loss
