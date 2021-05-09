from math import pi
import random
from typing import List

import numpy as np
import torch
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    mse_loss, tanh, relu
)

import matplotlib.pyplot as plt

import kornia
import pydiffvg

from models import BaseVAE, interpolate_vectors, reparameterize


OPAQUE_BLACK = (0, 0, 0, 1)
HIGH = np.array((0.565, 0.392, 0.173, 1))
LOW = np.array((0.094, 0.310, 0.635, 1))

dsample = kornia.transform.PyrDown()


def fig2data(fig) -> np.array:
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    x = np.array(fig.canvas.renderer.buffer_rgba())
    return x[:, :, :3]


def bilinear_downsample(tensor: torch.Tensor, size: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(tensor, size, mode='bilinear')


def sample_circle(r: int, angles: torch.Tensor, sample_rate: int = 10):
    pos = []
    for i in range(1, sample_rate + 1):
        x = (torch.cos(angles * (sample_rate / i)) * r)  # + r
        y = (torch.sin(angles * (sample_rate / i)) * r)  # + r
        pos.append(x)
        pos.append(y)
    return torch.stack(pos, dim=-1)


def decode_transform(x):
    return x.permute(0, 2, 1)


def gaussian_pyramid_loss(recons, inp, loss_fn):
    recon_loss = loss_fn(recons, inp, reduction='none').mean(dim=[1, 2, 3])
    for j in range(2, 5):
        recons = dsample(recons)
        inp = dsample(inp)
        recon_loss += loss_fn(recons, inp, reduction='none').mean(dim=[1, 2, 3]) / j
    return recon_loss


def raster_verbose(curves, points) -> ([pydiffvg.Path], [pydiffvg.ShapeGroup]):
    np.random.seed(0)
    colors = np.random.rand(curves, 4)
    colors[:, 3] = 1
    diff = (HIGH - LOW) / curves

    shapes = []
    shape_groups = []
    for i in range(curves):
        scale = diff * i
        color = LOW + scale
        color[3] = 1
        color = torch.tensor(color)
        num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
        if i * 3 + 4 > curves * 3:
            curve_points = torch.stack([points[i * 3], points[i * 3 + 1], points[i * 3 + 2], points[0]])
        else:
            curve_points = points[i * 3:i * 3 + 4]
        path = pydiffvg.Path(
            num_control_points=num_ctrl_pts, points=curve_points,
            is_closed=False, stroke_width=torch.tensor(4))
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=color)
        shapes.append(path)
        shape_groups.append(path_group)

    return shapes, shape_groups


class VectorVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List[int] = None,
                 loss_fn: str = 'MSE',
                 img_size: int = 128,
                 paths: int = 4,
                 **kwargs) -> None:
        super(VectorVAE, self).__init__()

        self.latent_dim_ = latent_dim  # Used by VectorVAEnLayers
        self.img_size_ = img_size
        self.should_reparameterize_ = kwargs.get('reparameterize', False)
        self.other_losses_weight_ = kwargs.get('other_losses_weight', 0)
        self.curves_ = paths
        self.scale_factor_ = kwargs['scale_factor']
        self.learn_sampling_ = kwargs['learn_sampling']

        self.beta = kwargs['beta']
        self.only_auxiliary_training = kwargs['only_auxiliary_training']

        if loss_fn == 'BCE':
            self.loss_fn_ = binary_cross_entropy_with_logits
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
                    torch.nn.ReLU())
            )
            in_channels = h_dim
        self.encoder_ = torch.nn.Sequential(*modules)

        out_size = img_size // (2 ** 5)
        self.fc_mu_ = torch.nn.Linear(hidden_dims[-1] * out_size * out_size, latent_dim)
        self.fc_var_ = torch.nn.Linear(hidden_dims[-1] * out_size * out_size, latent_dim)

        self.circle_rad_ = kwargs['radius']
        self.number_of_points_ = self.curves_ * 3

        angles = torch.arange(0, self.number_of_points_, dtype=torch.float32) * pi * 2 / self.number_of_points_
        sample_rate = 1
        self.id_circle_ = sample_circle(self.circle_rad_, angles, sample_rate)[:, :]
        base_control_features = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.float32)
        self.register_buffer('base_control_features', base_control_features)
        self.angles_ = angles

        def get_computational_unit(in_chan, out_chan, unit_type):
            if unit_type == 'conv':
                return torch.nn.Conv1d(
                    in_chan,
                    out_chan,
                    kernel_size=3,
                    padding=2,
                    padding_mode='circular',
                    stride=1,
                    dilation=1
                )
            else:
                return torch.nn.Linear(in_chan, out_chan)

        # Build Decoder
        num_one_hot = base_control_features.shape[1]
        fused_latent_dim = latent_dim + num_one_hot + (sample_rate * 2)

        unit = 'conv'
        self.decoder_input_ = get_computational_unit(fused_latent_dim, fused_latent_dim * 2, unit)

        self.point_predictor_ = torch.nn.ModuleList([
            get_computational_unit(fused_latent_dim * 2, fused_latent_dim * 2, unit),
            get_computational_unit(fused_latent_dim * 2, fused_latent_dim * 2, unit),
            get_computational_unit(fused_latent_dim * 2, fused_latent_dim * 2, unit),
            get_computational_unit(fused_latent_dim * 2, fused_latent_dim * 2, unit),
            get_computational_unit(fused_latent_dim * 2, 2, unit),
            # nn.Sigmoid()  # bound spatial extent
        ])
        if self.learn_sampling_:
            self.sample_deformation_ = torch.nn.Sequential(
                get_computational_unit(latent_dim + 2 + (sample_rate * 2), latent_dim * 2, unit),
                torch.nn.ReLU(),
                get_computational_unit(latent_dim * 2, latent_dim * 2, unit),
                torch.nn.ReLU(),
                get_computational_unit(latent_dim * 2, 1, unit),
            )

        unit = 'mlp'
        self.aux_network_ = torch.nn.Sequential(
            get_computational_unit(latent_dim, latent_dim * 2, unit),
            torch.nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, unit),
            torch.nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, unit),
            torch.nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, 3, unit),
        )
        self.latent_lossvpath_ = {}
        self.save_lossvspath = False
        if self.only_auxiliary_training:
            self.save_lossvspath = True
            for name, param in self.named_parameters():
                if 'aux_network' in name:
                    print(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def redo_features(self, n):
        self.curves_ = n
        self.number_of_points_ = self.curves_ * 3
        self.angles_ = (torch.arange(0, self.number_of_points_, dtype=torch.float32) * pi * 2 / self.number_of_points_)
        self.id_circle_ = sample_circle(self.circle_rad_, self.angles_, sample_rate=1)[:, :]

    def encode(self, inp: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param inp: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_(inp)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_(result)
        log_var = self.fc_var_(result)

        return mu, log_var

    def raster_(self, all_points, color=OPAQUE_BLACK, verbose=False, white_background=True) -> torch.Tensor:
        assert len(color) == 4
        render_size = self.img_size_
        if verbose:
            render_size *= 2
        all_points = all_points * render_size

        num_ctrl_pts = torch.zeros(self.curves_, dtype=torch.int32).to(all_points.device) + 2
        color = torch.tensor(color).to(all_points.device)
        batch_size = all_points.shape[0]
        outputs = []
        for k in range(batch_size):
            # Get point parameters from network
            render = pydiffvg.RenderFunction.apply
            points = all_points[k].contiguous()  # [self.sort_idx[k]] # .cpu()

            if verbose:
                shapes, shape_groups = raster_verbose(self.curves_, points)
            else:
                shapes = [pydiffvg.Path(num_control_points=num_ctrl_pts, points=points, is_closed=True)]
                shape_groups = [
                    pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([len(shapes) - 1]),
                        fill_color=color,
                        stroke_color=color
                    )
                ]

            scene_args = pydiffvg.RenderFunction.serialize_scene(render_size, render_size, shapes, shape_groups)
            out = render(render_size,  # width
                         render_size,  # height
                         3,  # num_samples_x
                         3,  # num_samples_y
                         102,  # seed
                         None,
                         *scene_args)
            out = out.permute(2, 0, 1).view(4, render_size, render_size)  # [:3]#.mean(0, keepdim=True)
            outputs.append(out)

        output = torch.stack(outputs).to(all_points.device)

        # map to [-1, 1]
        if white_background:
            alpha = output[:, 3:4, :, :]
            output_white_bg = output[:, :3, :, :] * alpha + (1 - alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)

        del num_ctrl_pts, color
        return output

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        self.id_circle_ = self.id_circle_.to(z.device)

        batch_size = z.shape[0]
        z = z[:, None, :].repeat([1, self.curves_ * 3, 1])
        base_control_features = self.base_control_features[None, :, :].repeat(batch_size, self.curves_, 1)
        z_base = torch.cat([z, base_control_features], dim=-1)
        if self.learn_sampling_:
            self.angles_ = self.angles_.to(z.device)
            angles = self.angles_[None, :, None].repeat(batch_size, 1, 1)
            x = torch.cos(angles)  # + r
            y = torch.sin(angles)  # + r
            z_angles = torch.cat([z_base, x, y], dim=-1)

            angles_delta = self.sample_deformation_(decode_transform(z_angles))
            angles_delta = tanh(angles_delta / 50) * pi / 2
            angles_delta = decode_transform(angles_delta)

            new_angles = angles + angles_delta
            x = (torch.cos(new_angles) * self.circle_rad_)  # + r
            y = (torch.sin(new_angles) * self.circle_rad_)  # + r
            z = torch.cat([z_base, x, y], dim=-1)
        else:
            id_circle = self.id_circle_[None, :, :].repeat(batch_size, 1, 1)
            z = torch.cat([z_base, id_circle], dim=-1)

        all_points = self.decoder_input_(decode_transform(z))
        for compute_block in self.point_predictor_:
            all_points = relu(all_points)
            all_points = compute_block(all_points)
        all_points = decode_transform(torch.sigmoid(all_points / self.scale_factor_))

        return all_points

    def reparameterize_(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return reparameterize(mu, log_var) if self.should_reparameterize_ else mu

    def forward(self, inp: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(inp)
        z = self.reparameterize_(mu, log_var)
        all_points = self.decode(z)
        if not self.only_auxiliary_training or self.save_lossvspath:
            output = self.raster_(all_points, white_background=True)
        else:
            output = torch.zeros([1, 3, 64, 64])
        return [output, inp, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log \\frac{1}{\\sigma} + \\frac{\\sigma^2 + \\mu^2}{2} - \\frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons, inp, mu, log_var = args[:4]
        recons = recons[:, :3, :, :]
        other_losses = args[4] if len(args) == 5 else 0
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        if not self.only_auxiliary_training or self.save_lossvspath:
            recon_loss = gaussian_pyramid_loss(recons, inp, self.loss_fn_)
        else:
            recon_loss = torch.zeros([1])

        if self.only_auxiliary_training:
            recon_loss_non_reduced = recon_loss[:, None].clone().detach()
            spacing = self.aux_network_(mu.clone().detach())
            latents = mu.cpu().numpy()
            num_latents = latents.shape[0]

            if self.save_lossvspath:
                recon_loss_non_reduced_cpu = recon_loss_non_reduced.cpu().numpy()
                keys = self.latent_lossvpath_.keys()
                for i in range(num_latents):
                    if np.array2string(latents[i]) in keys:
                        pair = torch.tensor([self.curves_, recon_loss_non_reduced_cpu[i, 0], ])[None, :].to(mu.device)
                        self.latent_lossvpath_[np.array2string(latents[i])] \
                            = torch.cat([self.latent_lossvpath_[np.array2string(latents[i])], pair], dim=0)
                    else:
                        self.latent_lossvpath_[np.array2string(latents[i])] = torch.tensor(
                            [[self.curves_, recon_loss_non_reduced_cpu[i, 0]], ]).to(mu.device)
                num = torch.ones_like(spacing[:, 0]) * self.curves_
                est_loss = spacing[:, 2] + 1 / torch.exp(num * spacing[:, 0] - spacing[:, 1])
                aux_loss = torch.abs(num * (est_loss - recon_loss_non_reduced)).mean() * 10
            else:
                aux_loss = 0
                for i in range(num_latents):
                    pair = self.latent_lossvpath_[np.array2string(latents[i])]
                    est_loss = spacing[i, 2] + 1 / torch.exp(pair[:, 0] * spacing[i, 0] - spacing[i, 1])
                    aux_loss += torch.abs(pair[:, 0] * (est_loss - pair[:, 1])).mean()
            logs = {'Reconstruction_Loss': recon_loss.mean(), 'aux_loss': aux_loss}

            return {'loss': aux_loss, 'progress_bar': logs}

        recon_loss = recon_loss.mean()
        kld_loss = 0
        if self.beta > 0:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)\
                       * self.beta * kld_weight
        recon_loss = recon_loss * 10
        loss = recon_loss + kld_loss + other_losses * self.other_losses_weight_
        logs = {
            'Reconstruction_Loss': recon_loss.detach(),
            'KLD': -kld_loss,
            'other losses': other_losses.detach() * self.other_losses_weight_
        }

        return {'loss': loss, 'progress_bar': logs}

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize_(mu, log_var)
        return self.raster_(self.decode(z), verbose=random.choice([True, False]))

    def save(self, x, save_dir, name):
        z, log_var = self.encode(x)
        all_points = self.decode(z)

        # Get point parameters from network
        points = all_points[0].cpu()  # [self.sort_idx[k]]

        color = torch.cat([torch.tensor([0, 0, 0, 1]), ])
        num_ctrl_pts = torch.zeros(self.curves_, dtype=torch.int32) + 2

        shapes = [
            pydiffvg.Path(num_control_points=num_ctrl_pts, points=points, is_closed=True)
        ]
        shape_groups = [
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=color,
                stroke_color=color
            )
        ]
        pydiffvg.save_svg(f"{save_dir}{name}/{name}.svg", self.img_size_, self.img_size_, shapes, shape_groups)

    # TODO: interpolation functions seem hardcoded
    def interpolate(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = interpolate_vectors(mu[2], mu[i], 10)
            all_points = self.decode(z)
            all_interpolations.append(self.raster_(all_points, verbose=kwargs['verbose']))
        return all_interpolations

    def interpolate_2d(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        all_interpolations = []
        y_axis = interpolate_vectors(mu[7], mu[6], 10)
        for i in range(10):
            z = interpolate_vectors(y_axis[i], mu[3], 10)
            all_points = self.decode(z)
            all_interpolations.append(self.raster_(all_points, verbose=kwargs['verbose']))
        return all_interpolations

    def naive_vector_interpolate(self, x: torch.Tensor, **kwargs) -> [torch.Tensor]:
        mu, log_var = self.encode(x)
        all_points = self.decode(mu)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = interpolate_vectors(all_points[2], all_points[i], 10)
            all_interpolations.append(self.raster_(z, verbose=kwargs['verbose']))
        return all_interpolations

    def visualize_sampling(self, x: torch.Tensor, **kwargs) -> [torch.Tensor]:
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(5, 27):
            self.redo_features(i)
            all_points = self.decode(mu)
            all_interpolations.append(self.raster_(all_points, verbose=kwargs['verbose']))
        return all_interpolations

    def sampling_error(self, x: torch.Tensor) -> torch.Tensor:
        error = []
        figure = plt.figure(figsize=(6, 6))
        batch_size = x.shape[0]
        for i in range(7, 25):
            self.redo_features(i)
            results = self.forward(x)
            recons = results[0][:, :3, :, :]
            input_batch = results[1]
            recon_loss = gaussian_pyramid_loss(recons, input_batch, self.loss_fn_)
            error.append(recon_loss)
        etn = torch.stack(error, dim=1).numpy()

        np.savetxt('sample_error.csv', etn, delimiter=',')
        y = np.arange(7, 25)
        for i in range(batch_size):
            plt.plot(y, etn[i, :], label=str(i + 1))
        plt.legend(loc='upper right')
        img = fig2data(figure)
        return img

    def visualize_aux_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        batch_size = mu.shape[0]
        all_spacing = []
        figure = plt.figure(figsize=(6, 6))

        for i in np.arange(7, 25):
            spacing = self.aux_network_(mu.clone().detach())
            num = torch.ones_like(spacing[:, 0]) * i
            est_loss = spacing[:, 2] + (spacing[:, 0] / num)
            all_spacing.append(est_loss)
        all_spacing = torch.stack(all_spacing, dim=1).detach().cpu().numpy()

        y = np.arange(7, 25)
        for i in range(batch_size):
            plt.plot(y, all_spacing[i, :], label=str(i + 1))
        plt.legend(loc='upper right')
        img = fig2data(figure)
        return img
