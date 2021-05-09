import argparse
import os
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


def get_last_weight_path(model_save_path: str) -> str:
    weights = [os.path.join(model_save_path, x) for x in os.listdir(model_save_path) if '.ckpt' in x]
    weights.sort(key=lambda x: os.path.getmtime(x))
    return weights[-1] if weights else None


def make_test_tube_logger(config):
    return TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
        version=config['logging_params']['version'],
    )


def make_model_save_path(config, tt_logger=None):
    version = 'best' if tt_logger is None else tt_logger.version
    return '{}/{}/version_{}'.format(
        config['logging_params']['save_dir'],
        config['logging_params']['name'],
        version
    )


def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x
