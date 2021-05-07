import argparse
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pytorch_lightning.loggers import TestTubeLogger


def enable_reproducibility(config):
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False


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


def make_test_tube_logger(config):
    return TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
        version=config['logging_params']['version'],
    )


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
