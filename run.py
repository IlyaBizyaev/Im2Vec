import os
from shutil import copytree, ignore_patterns, rmtree

import click

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from experiment import VAEExperiment
from models import *
from utils import (
    enable_reproducibility,
    get_last_weight_path,
    request_and_read_config,
    make_test_tube_logger
)


config = request_and_read_config()
torch.autograd.set_detect_anomaly(True)

IGNORE_PATTERNS = ignore_patterns('*.pyc', '*.md', 'tmp*', 'logs*', 'data*', '.git*')


def main():
    root_dir = os.getcwd()
    resume = False
    tt_logger = make_test_tube_logger(config)
    model_save_path = '{}/{}/version_{}/'.format(
        config['logging_params']['save_dir'],
        config['logging_params']['name'],
        tt_logger.version
    )
    print(model_save_path)
    # Copying the folder
    if os.path.exists(model_save_path):
        if config['model_params']['only_auxiliary_training'] or config['model_params']['memory_leak_training']:
            print('Training Auxiliary Network or Memory Leak')
        elif click.confirm('Folder exists, override?', default=True):
            rmtree(model_save_path)
            copytree(root_dir, model_save_path, ignore=IGNORE_PATTERNS)
        else:
            resume = True
    else:
        copytree(root_dir, model_save_path, ignore=IGNORE_PATTERNS)

    enable_reproducibility(config)

    print(f"Model params: {config['model_params']}")
    model = make_model(config)
    experiment = VAEExperiment(model, config['exp_params'])

    model_path = None
    if config['model_params']['only_auxiliary_training'] or config['model_params']['memory_leak_training'] or resume:
        model_path = get_last_weight_path(model_save_path)
        print('loading: ', model_path)
        if config['model_params']['only_auxiliary_training']:
            checkpoint = torch.load(model_path)
            experiment.load_state_dict(checkpoint['state_dict'])
            model_path = None

    checkpoint_callback = ModelCheckpoint(model_save_path, verbose=True, save_last=True)

    print(f"Experiment params: {config['exp_params']}")
    runner = Trainer(
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=model_path,
        logger=tt_logger,
        weights_summary='full',
        **config['trainer_params']
    )

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)


if __name__ == '__main__':
    main()
