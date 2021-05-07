import os

from experiment import VAEExperiment
from models import *
from utils import enable_reproducibility, request_and_read_config


config = request_and_read_config()

enable_reproducibility(config)


def main():
    model_save_path = os.getcwd()
    parent = '/'.join(model_save_path.split('/')[:-3])
    config['logging_params']['save_dir'] = os.path.join(parent, config['logging_params']['save_dir'])
    config['exp_params']['data_path'] = os.path.join(parent, config['exp_params']['data_path'])
    print(parent, config['exp_params']['data_path'])

    model = vae_models[config['model_params']['name']](
        imsize=config['exp_params']['img_size'],
        **config['model_params']
    )
    experiment = VAEExperiment(model, config['exp_params'])

    weights = [x for x in os.listdir(model_save_path) if '.ckpt' in x]
    weights.sort(key=lambda x: os.path.getmtime(x))
    load_weight = weights[-1]
    print('loading: ', load_weight)

    checkpoint = torch.load(load_weight)
    experiment.load_state_dict(checkpoint['state_dict'])
    _ = experiment.train_dataloader()

    experiment.eval()
    experiment.freeze()
    experiment.sample_interpolate(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        version=config['logging_params']['version'],
        save_svg=True,
        other_interpolations=config['logging_params']['other_interpolations']
    )


if __name__ == '__main__':
    main()
