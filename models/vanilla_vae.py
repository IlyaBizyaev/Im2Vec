from typing import List

import torch
from torch.nn.functional import binary_cross_entropy, mse_loss

from models import BaseVAE, interpolate_vectors


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 loss_fn: str = 'MSE',
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        print('Beta:', beta, ' loss_fn: ', loss_fn)
        self.beta = beta
        self.scale_factor = kwargs['scale_factor']
        self.learn_sampling = kwargs['learn_sampling']
        self.only_auxiliary_training = kwargs['only_auxiliary_training']
        self.memory_leak_training = kwargs['memory_leak_training']
        self.other_losses_weight = 0
        if loss_fn == 'BCE':
            self.output_transform = lambda x: x
            self.loss_fn = binary_cross_entropy
        else:
            self.output_transform = lambda x: x
            self.loss_fn = mse_loss
        self.latent_dim = latent_dim
        self.memory_leak_epochs = 105
        if 'memory_leak_epochs' in kwargs.keys():
            self.memory_leak_epochs = kwargs['memory_leak_epochs']

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
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

        self.encoder = torch.nn.Sequential(*modules)
        imsize = kwargs['imsize']
        outsize = int(imsize / (2 ** 5))
        self.fc_mu = torch.nn.Linear(hidden_dims[-1] * outsize * outsize, latent_dim)
        self.fc_var = torch.nn.Linear(hidden_dims[-1] * outsize * outsize, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = torch.nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

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
            # modules.append(
            #     nn.Sequential(
            #         nn.ConvTranspose2d(hidden_dims[i+1],
            #                            hidden_dims[i+1],
            #                            kernel_size=3,
            #                            stride = 2,
            #                            padding=1,
            #                            output_padding=1),
            #         # nn.BatchNorm2d(hidden_dims[i + 1]),
            #         nn.LeakyReLU())
            # )
            #

        self.decoder = torch.nn.Sequential(*modules)

        self.final_layer = torch.nn.Sequential(
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
            torch.nn.Sigmoid())

    def encode(self, inp: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param inp: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(inp)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, inp: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(inp)
        z = reparameterize(mu, log_var)
        return [self.decode(z), inp, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log \\frac{1}{\\sigma} + \\frac{\\sigma^2 + \\mu^2}{2} - \\frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        inp = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        # recons_loss =F.mse_loss(recons, input)
        recons = self.output_transform(recons)
        inp = self.output_transform(inp)
        recons_loss = self.loss_fn(recons, inp)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.beta * kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
        # .type(torch.FloatTensor).to(device)

    def interpolate(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = interpolate_vectors(mu[2], mu[i], 10)
            all_interpolations.append(self.decode(z))
        return all_interpolations
