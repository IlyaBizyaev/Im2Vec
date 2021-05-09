import torch

from experiment import VAEExperiment
from models import make_model
from utils import (
    enable_reproducibility,
    get_last_weight_path,
    make_model_save_path,
    request_and_read_config
)


config = request_and_read_config()

enable_reproducibility(config)


def main():
    model = make_model(config)

    model_save_path = make_model_save_path(config)
    load_weight = get_last_weight_path(model_save_path)
    print('loading: ', load_weight)

    # Alternatively: experiment = VAEExperiment.load_from_checkpoint(
    #     load_weight, vae_model=model, params=config['exp_params']
    # )
    experiment = VAEExperiment(model, config['exp_params'])
    checkpoint = torch.load(load_weight)
    experiment.load_state_dict(checkpoint['state_dict'])

    experiment.val_dataloader()

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
