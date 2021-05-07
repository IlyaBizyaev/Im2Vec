from abc import abstractmethod
from typing import Any, List

import torch


def interpolate_vectors(v1: torch.Tensor, v2: torch.Tensor, n: int) -> torch.Tensor:
    step = (v2 - v1) / (n - 1)
    return torch.stack([v1 + i * step for i in range(n)], dim=0)


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


class BaseVAE(torch.nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, inp: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError

    def decode(self, inp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        raise RuntimeWarning()

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> dict:
        pass
