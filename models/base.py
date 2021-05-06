from typing import Any, List
import torch
from abc import abstractmethod


def interpolate_vectors(v1, v2, n):
    step = (v2 - v1) / (n - 1)
    vectors = []
    for i in range(n):
        vectors.append(v1 + i * step)
    return torch.stack(vectors, dim=0)


class BaseVAE(torch.nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, inp: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, inp: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        raise RuntimeWarning()

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass
