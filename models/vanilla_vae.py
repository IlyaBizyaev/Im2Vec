from typing import List

import torch
from torch.nn.functional import binary_cross_entropy, mse_loss

from models import BaseVAE, interpolate_vectors, reparameterize


def calc_kld_loss(mu, log_var):
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 loss_fn: str = 'MSE',
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        print('Beta:', beta, 'loss_fn:', loss_fn)
        self.beta = beta
        self.only_auxiliary_training = kwargs['only_auxiliary_training']

        if loss_fn == 'BCE':
            self.loss_fn_ = binary_cross_entropy
        else:
            self.loss_fn_ = mse_loss

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ),
                    # nn.BatchNorm2d(h_dim),
                    torch.nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder_ = torch.nn.Sequential(*modules)
        out_size = kwargs['img_size'] // (2 ** 5)
        self.fc_mu_ = torch.nn.Linear(hidden_dims[-1] * out_size * out_size, latent_dim)
        self.fc_var_ = torch.nn.Linear(hidden_dims[-1] * out_size * out_size, latent_dim)

        # Build Decoder
        self.decoder_input_ = torch.nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    torch.nn.LeakyReLU()
                )
            )
        self.decoder_ = torch.nn.Sequential(*modules)

        self.final_layer_ = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            # nn.BatchNorm2d(hidden_dims[-1]),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                hidden_dims[-1],
                out_channels=3,
                kernel_size=3,
                padding=1
            ),
            torch.nn.Sigmoid()
        )

    def encode(self, inp: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param inp: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_(inp)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu_(result)
        log_var = self.fc_var_(result)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input_(z)
        result = result.view(-1, 512, 2, 2)  # TODO: why 512?
        result = self.decoder_(result)
        result = self.final_layer_(result)
        return result

    def forward(self, inp: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(inp)
        z = reparameterize(mu, log_var)
        return [self.decode(z), inp, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log \\frac{1}{\\sigma} + \\frac{\\sigma^2 + \\mu^2}{2} - \\frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons, inp, mu, log_var = args[:4]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = self.loss_fn_(recons, inp)
        kld_loss = calc_kld_loss(mu, log_var)
        loss = recons_loss + self.beta * kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    def interpolate(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var_ = self.encode(x)
        return [self.decode(interpolate_vectors(mu[2], mu[i], 10)) for i in range(mu.shape[0])]
