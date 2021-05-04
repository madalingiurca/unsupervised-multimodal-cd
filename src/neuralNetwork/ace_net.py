from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import mse_loss

from src.neuralNetwork.hyperparams import *


class AceNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE

        self.Rx = nn.Sequential(
            nn.Conv2d(11, 100, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(100, 50, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(50, 20, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3)
        )

        self.Py = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(100, 50, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(50, 20, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3)
        )

        self.Sz = nn.Sequential(
            nn.ConvTranspose2d(20, 100, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(100, 200, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(200, 3, kernel_size=(3, 3)),
        )

        self.Qz = nn.Sequential(
            nn.ConvTranspose2d(20, 100, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(100, 200, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(200, 11, kernel_size=(3, 3)),
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(64, 32, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(32, 16, kernel_size=(3, 3)),
            nn.Flatten(),
            nn.Linear(123904, 1),
            # nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def training_step(self, test_batch, batch_idx, optimizer_idx):
        x, y, prior_info = test_batch
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        opt_encoders_decoders, opt_encoders, opt_disc = self.configure_optimizers()

        opt_encoders_decoders.zero_grad()

        y_hat = self.Sz(self.Rx(x))
        x_translation_loss = self.mse_loss_weighted(y, y_hat, 1 - prior_info)
        x_cycled = self.Qz(self.Py(y_hat))
        x_cycle_loss = mse_loss(x, x_cycled)
        x_reconstructed = self.Qz(self.Rx(x))
        x_recon_loss = mse_loss(x, x_reconstructed)

        # Y data flow
        x_hat = self.Qz(self.Py(y))
        y_translation_loss = self.mse_loss_weighted(x, x_hat, 1 - prior_info)
        y_cycled = self.Sz(self.Rx(x_hat))
        y_cycle_loss = mse_loss(y, y_cycled)
        x_hat = self.Sz(self.Py(y))
        y_recon_loss = mse_loss(y, x_hat)

        total_AE_loss = (
                W_RECON * (x_recon_loss + y_recon_loss) +
                W_HAT * (x_translation_loss + y_translation_loss) +
                W_CYCLE * (x_cycle_loss + y_cycle_loss)
        )

        self.log("Reconstruction loss", W_RECON * (x_recon_loss + y_recon_loss))
        self.log("Prior information loss", W_HAT * (x_translation_loss + y_translation_loss))
        self.log("Total AutoEncoders loss", total_AE_loss, prog_bar=True)

        self.manual_backward(total_AE_loss, opt_encoders_decoders)
        opt_encoders_decoders.step()

        opt_encoders.zero_grad()

        generator_code = self.discriminator(torch.cat((self.Rx(x), self.Py(y))))
        x_disc, y_disc = torch.tensor_split(generator_code, 2, dim=0)
        disc_code_loss = W_D * (mse_loss(torch.zeros_like(x_disc), x_disc) + mse_loss(torch.ones_like(y_disc), y_disc))
        self.log("Discriminator code loss", disc_code_loss)

        self.manual_backward(disc_code_loss, opt_encoders)
        opt_encoders.step()

        opt_disc.zero_grad()

        disc_out = self.discriminator(torch.cat((self.Rx(x), self.Py(y))))
        x_disc, y_disc = torch.tensor_split(disc_out, 2, dim=0)

        disc_loss = W_D * (mse_loss(torch.ones_like(x_disc), x_disc) + mse_loss(torch.zeros_like(y_disc), y_disc))

        self.log("Discriminator loss", disc_loss)

        self.manual_backward(disc_loss, opt_disc)
        opt_disc.step()

    def configure_optimizers(self) -> List[torch.optim.Adam]:
        encoders_params = list(self.Rx.parameters()) + list(self.Py.parameters())
        decoders_params = list(self.Sz.parameters()) + list(self.Qz.parameters())

        optimizer_ae = torch.optim.Adam(encoders_params + decoders_params, lr=self.learning_rate,
                                        weight_decay=W_REG)
        optimizer_e = torch.optim.Adam(encoders_params, lr=self.learning_rate, weight_decay=W_REG)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,
                                          weight_decay=W_REG)

        return [optimizer_ae, optimizer_e, optimizer_disc]

    @staticmethod
    def mse_loss_weighted(x, x_hat, weights):
        L2_Norm = torch.linalg.norm(x_hat - x, dim=1) ** 2
        weighted_L2_norm: torch.Tensor = L2_Norm * weights
        loss = weighted_L2_norm.mean()
        return loss
