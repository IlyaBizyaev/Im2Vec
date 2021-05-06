import argparse
import yaml

import numpy as np
import torch


def request_and_read_config():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    x = np.array(fig.canvas.renderer.buffer_rgba())
    return x[:, :, :3]


def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x


def hard_composite(**kwargs):
    layers = kwargs['layers']
    n = len(layers)
    alpha = (1 - layers[n - 1][:, 3:4, :, :])
    rgb = layers[n - 1][:, :3] * layers[n - 1][:, 3:4, :, :]
    for i in reversed(range(n - 1)):
        rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * alpha
        alpha = (1 - layers[i][:, 3:4, :, :]) * alpha
    rgb = rgb + alpha
    return rgb
