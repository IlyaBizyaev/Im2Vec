import gc
import os
import random

import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from torchvision.utils import save_image

from pytorch_lightning import LightningModule
from torch_optimizer import AdamP
from ranger import Ranger

from models import BaseVAE


class VAEExperiment(LightningModule):

    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAEExperiment, self).__init__()

        self.model = vae_model
        self.params = params

        self.cur_device = None
        self.first_epoch = True
        self.beta_scale = 2.0  # 1.4

        self.num_train_imgs = None
        self.val_dataloader_ = None
        self.num_val_imgs = None

    def forward(self, inp: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(inp, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.cur_device = real_img.device
        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)['loss']

        if self.model.only_auxiliary_training:
            path = self.current_epoch + 6
            if path > 30:
                path = random.randint(7, 25)
                self.model.save_lossvspath = False
        else:
            path = random.randint(7, 25)
        if self.params['grow']:
            self.model.redo_features(path)

        self.log('loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.cur_device = real_img.device

        results = self.forward(real_img, labels=labels)
        loss = self.model.loss_function(*results,
                                        M_N=self.params['batch_size'] / self.num_val_imgs,
                                        optimizer_idx=optimizer_idx,
                                        batch_idx=batch_idx)['loss']
        self.log('val_loss', loss)
        return loss

    def on_load_checkpoint(self, checkpoint):
        load_epoch = checkpoint['epoch']
        new_beta = self.model.beta * (self.beta_scale ** (load_epoch // 25))
        self.model.beta = min(new_beta, 4)
        print('loaded beta: ', self.model.beta)

    def training_epoch_end(self, training_step_outputs):
        super(VAEExperiment, self).training_epoch_end(training_step_outputs)
        self.sample_images_()
        if (self.current_epoch + 1) % self.model.memory_leak_epochs == 0 \
                and self.model.memory_leak_training \
                and not self.first_epoch:
            quit()
        self.first_epoch = False
        print('beta: ', self.model.beta)
        if self.current_epoch % 25 == 0:
            self.model.beta = min(self.model.beta * self.beta_scale, 4)
        gc.collect()
        torch.cuda.empty_cache()

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        if training_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
            self.log("loss", avg_loss)
            tensorboard_logs = {'avg_train_loss': avg_loss, 'learning_rate': lr}
            self.log("log", tensorboard_logs)

    def sample_images_(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.val_dataloader_))
        test_input = test_input.to(self.cur_device)
        recons = self.model.generate(test_input, labels=test_label)

        prefix = f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        suffix = f"{self.logger.name}_{self.current_epoch:04d}.png"

        save_image(recons.data, f"{prefix}recons_{suffix}", normalize=False, nrow=12)
        save_image(test_input.data, f"{prefix}real_img_{suffix}", normalize=False, nrow=12)

        del test_input, recons

    def sample_interpolate(self, save_dir, name, version, save_svg=False, other_interpolations=False):
        test_input, test_label = next(iter(self.val_dataloader_))
        test_input = test_input.to(self.cur_device)

        prefix = f"{save_dir}{name}/version_{version}/{name}"

        methods = [
            ("interpolate", 10)
        ]

        if other_interpolations:
            methods += [
                ("interpolate_2d", 10),
                ("visualize_sampling", self.params['val_batch_size']),
                ("naive_vector_interpolate", 10)
            ]

        for method_name, nrow in methods:
            method = getattr(self.model, method_name)
            for verbose in [False, True]:
                suffix = "vector" if verbose else "image"

                interpolate_samples = method(test_input, verbose=verbose)
                interpolate_samples = torch.cat(interpolate_samples, dim=0)
                save_image(interpolate_samples.cpu().data,
                           f"{prefix}_{method_name}_{suffix}.png",
                           normalize=False,
                           nrow=nrow)

        if other_interpolations:
            sampling_graph = self.model.sampling_error(test_input)
            plt.imsave(f"{prefix}_recons_graph.png", sampling_graph)
            if self.model.only_auxiliary_training:
                graph = self.model.visualize_aux_error(test_input, verbose=True)
                plt.imsave(f"{prefix}_aux_graph.png", graph)

        recons = self.model.generate(test_input, labels=test_label)
        save_image(recons.cpu().data, f"{prefix}_recons.png", normalize=False, nrow=10)
        save_image(test_input.cpu().data, f"{prefix}_input.png", normalize=False, nrow=10)
        if save_svg:
            self.model.save(test_input, save_dir, name)

    def configure_optimizers(self):
        if self.model.only_auxiliary_training:
            print('Learning Rate changed for auxiliary training')
            self.params['LR'] = 0.00001
        optimizer = Ranger(self.model.parameters(),
                           lr=self.params['LR'],
                           weight_decay=self.params['weight_decay'])
        optimizers = [optimizer]
        # Check if more than 1 optimizer is required (Used for adversarial training)
        if 'LR_2' in self.params:
            optimizer2 = AdamP(getattr(self.model, self.params['submodel']).parameters(),
                               lr=self.params['LR_2'])
            optimizers.append(optimizer2)

        scheduler = ReduceLROnPlateau(
            optimizers[0], 'min', verbose=True,
            factor=self.params['scheduler_gamma'],
            min_lr=0.0001,
            patience=int(self.model.memory_leak_epochs / 7)
        )
        schedulers = [{
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
        }]
        # Check if another scheduler is required for the second optimizer
        if 'scheduler_gamma_2' in self.params:
            scheduler2 = ExponentialLR(optimizers[1], gamma=self.params['scheduler_gamma_2'])
            schedulers.append(scheduler2)

        return optimizers, schedulers

    def train_dataloader(self) -> DataLoader:
        transform = self.data_transforms_()

        train_dataset = ImageFolder(self.params['data_path'], transform=transform)
        self.num_train_imgs = len(train_dataset)

        return DataLoader(train_dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=False, num_workers=1)

    def val_dataloader(self):
        transform = self.data_transforms_()

        val_dataset_path = self.params['data_path'].replace('train', 'test')
        if not os.path.exists(val_dataset_path):
            val_dataset_path = self.params['data_path']

        val_dataset = ImageFolder(val_dataset_path, transform=transform)
        self.num_val_imgs = len(val_dataset)
        dataloader = DataLoader(val_dataset,
                                batch_size=self.params['val_batch_size'],
                                shuffle=self.params['val_shuffle'],
                                drop_last=False)
        self.val_dataloader_ = dataloader

        return dataloader

    def data_transforms_(self):
        return Compose([
            Resize(self.params['img_size']),
            CenterCrop(self.params['img_size']),
            ToTensor(),
        ])
