import os

from experiment import VAEExperiment
from models import *
from utils import enable_reproducibility, request_and_read_config, make_test_tube_logger


config = request_and_read_config()

enable_reproducibility(config)


def main():
    tt_logger = make_test_tube_logger(config)

    model = VAE_MODELS[config['model_params']['name']](
        imsize=config['exp_params']['img_size'],
        **config['model_params']
    )
    model_save_path = '{}/{}/version_{}'.format(
        config['logging_params']['save_dir'],
        config['logging_params']['name'],
        tt_logger.version
    )

    if config['logging_params']['resume'] is None:
        weights = [os.path.join(model_save_path, x) for x in os.listdir(model_save_path) if '.ckpt' in x]
        weights.sort(key=lambda x: os.path.getmtime(x))
        model_path = weights[-1]
        print('loading: ', model_path)
    else:
        model_path = '{}/{}'.format(model_save_path, config['logging_params']['resume'])
    experiment = VAEExperiment.load_from_checkpoint(model_path, vae_model=model, params=config['exp_params'])

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
