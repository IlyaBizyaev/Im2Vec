import random
from typing import List

import torch

import pydiffvg

from models import VectorVAE, interpolate_vectors, raster_verbose, HIGH, LOW


# TODO: this hardcodes colors for the emoji dataset!!
COLORS = [
    [252/255, 194/255, 27/255, 1.0],  # emoji yellow...
    [1.0, 0, 0, 1.0],  # red
    [0, 1.0, 0, 1.0],  # green
    [0, 0, 1.0, 1.0],  # blue
]


def soft_composite(layers, z_layers=None):
    # TODO: how is this supposed to work w/o z_layers?
    n = len(layers)

    inv_mask = (1 - layers[0][:, 3:4, :, :])
    for i in range(1, n):
        inv_mask = inv_mask * (1 - layers[i][:, 3:4, :, :])

    sum_alpha = layers[0][:, 3:4, :, :] * z_layers[0]
    for i in range(1, n):
        sum_alpha += layers[i][:, 3:4, :, :] * z_layers[i]
    sum_alpha += inv_mask

    inv_mask = inv_mask / sum_alpha

    rgb = layers[0][:, :3] * layers[0][:, 3:4, :, :] * z_layers[0] / sum_alpha
    for i in range(1, n):
        rgb += layers[i][:, :3] * layers[i][:, 3:4, :, :] * z_layers[i] / sum_alpha
    rgb = rgb * (1 - inv_mask) + inv_mask
    return rgb


class VectorVAEnLayers(VectorVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 loss_fn: str = 'MSE',
                 img_size: int = 128,
                 paths: int = 4,
                 **kwargs) -> None:
        super(VectorVAEnLayers, self).__init__(in_channels,
                                               latent_dim,
                                               hidden_dims,
                                               loss_fn,
                                               img_size,
                                               paths,
                                               **kwargs)

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

        self.rnn_ = torch.nn.LSTM(latent_dim, latent_dim, 2, bidirectional=True)
        # TODO: this is literally just ReLU
        self.divide_shape_ = torch.nn.Sequential(
            torch.nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # torch.nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # torch.nn.ReLU(),  # bound spatial extent
        )
        self.final_shape_latent_ = torch.nn.Sequential(
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            torch.nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            torch.nn.ReLU(),  # bound spatial extent
        )
        self.z_order_ = torch.nn.Sequential(
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # torch.nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # torch.nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, 1, 'mlp'),
        )
        layer_id = torch.eye(3)
        self.register_buffer('layer_id', layer_id)

    def forward(self, inp: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(inp)
        z = self.reparameterize_(mu, log_var)
        output, control_loss = self.decode_and_composite_(z, verbose=False, return_overlap_loss=True)
        return [output, inp, mu, log_var, control_loss]

    def decode_and_composite_(self, z: torch.Tensor, return_overlap_loss=False, **kwargs):
        layers = []
        n = len(COLORS)
        loss = 0
        z_rnn_input = z[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim]
        outputs, hidden = self.rnn_(z_rnn_input)
        outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
        outputs = outputs[:, :, :self.latent_dim_] + outputs[:, :, self.latent_dim_:]
        z_layers = []
        # TODO: the number of layers is equal to the number of hardcoded colors above!
        for i in range(n):
            shape_output = self.divide_shape_(outputs[:, i, :])
            shape_latent = self.final_shape_latent_(shape_output)
            all_points = self.decode(shape_latent)
            layer = self.raster_(all_points, COLORS[i], verbose=kwargs['verbose'], white_background=False)
            z_pred = self.z_order_(shape_output)
            layers.append(layer)
            z_layers.append(torch.exp(z_pred[:, :, None, None]))
            if return_overlap_loss:
                loss += self.control_polygon_distance_(all_points)

        output = soft_composite(layers, z_layers)
        if return_overlap_loss:
            return output, loss
        return output

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize_(mu, log_var)
        output = self.decode_and_composite_(z, verbose=random.choice([True, False]))
        return output

    def control_polygon_distance_(self, all_points):
        def distance(vec1, vec2):
            return ((vec1 - vec2) ** 2).mean()

        loss = 0
        for idx in range(self.number_of_points_):
            c_0 = all_points[:, idx - 1, :]
            c_1 = all_points[:, idx, :]
            loss += distance(c_0, c_1)
        return loss

    # TODO: how similar are these to the 1 layer case? Can the classes be merged in favor of this one?
    def interpolate(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = interpolate_vectors(mu[2], mu[i], 10)
            output = self.decode_and_composite_(z, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations

    def interpolate_mini(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = interpolate_vectors(mu[0], mu[1], 10)
        output = self.decode_and_composite_(z, verbose=kwargs['verbose'])
        return output

    def interpolate_2d(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        all_interpolations = []
        y_axis = interpolate_vectors(mu[7], mu[6], 10)
        for i in range(10):
            z = interpolate_vectors(y_axis[i], mu[3], 10)
            output = self.decode_and_composite_(z, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations

    def naive_vector_interpolate(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        all_interpolations = []
        batch_size = mu.shape[0]
        n = len(COLORS)
        for j in range(batch_size):
            layers = []
            z_rnn_input = mu[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim]
            outputs, hidden = self.rnn_(z_rnn_input)
            outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
            outputs = outputs[:, :, :self.latent_dim_] + outputs[:, :, self.latent_dim_:]
            for i in range(n):
                shape_latent = self.divide_shape_(outputs[:, i, :])
                all_points = self.decode(shape_latent)
                all_points_interpolate = interpolate_vectors(all_points[2], all_points[j], 10)
                layer = self.raster_(all_points_interpolate, COLORS[i], verbose=kwargs['verbose'])
                layers.append(layer)

            output = soft_composite(layers=layers)
            all_interpolations.append(output)
        return all_interpolations

    def visualize_sampling(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(7, 25):
            self.redo_features(i)
            output = self.decode_and_composite_(mu, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations

    # TODO: this feels largely duplicated and hardcoded
    def save(self, all_points, save_dir, name, verbose=False, white_background=True):
        # note that this if for a single shape and batch_size dimension should have multiple curves
        render_size = self.img_size_
        batch_size = all_points.shape[0]
        if verbose:
            render_size *= 2
        all_points = all_points * render_size
        num_ctrl_pts = torch.zeros(self.curves_, dtype=torch.int32) + 2

        for k in range(batch_size):
            # Get point parameters from network
            points = all_points[k].cpu().contiguous()  # [self.sort_idx[k]]

            if verbose:
                shapes, shape_groups = raster_verbose(self.curves_, points)
                diff = (HIGH - LOW) / self.curves_
                for i in range(self.curves_ * 3):
                    scale = diff * (i // 3)
                    color = LOW + scale
                    color[3] = 1
                    color = torch.tensor(color)

                    if i % 3 == 0:
                        shape = pydiffvg.Rect(p_min=points[i] - 8, p_max=points[i] + 8)
                        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves_ + i]), fill_color=color)
                    else:
                        shape = pydiffvg.Circle(radius=torch.tensor(8.0), center=points[i])
                        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves_ + i]), fill_color=color)
                    shapes.append(shape)
                    shape_groups.append(group)
            else:
                color = torch.tensor(COLORS[k])

                shapes = [pydiffvg.Path(num_control_points=num_ctrl_pts, points=points, is_closed=True)]
                shape_groups = [
                    pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([len(shapes) - 1]),
                        fill_color=color,
                        stroke_color=color
                    )
                ]
            pydiffvg.save_svg(f"{save_dir}{name}/{name}.svg", self.img_size_, self.img_size_, shapes, shape_groups)
