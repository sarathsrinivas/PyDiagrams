import torch as tc
from torch import Tensor
from typing import Sequence

tc.set_default_tensor_type(tc.DoubleTensor)


class Sampler:
    """
     Base class for sampling random samples
    """

    def __init__(self, seed: int):
        self.seed: int = seed

    def sample(
        self, n: int, mins: Sequence[float], maxs: Sequence[float]
    ) -> Tensor:
        raise NotImplementedError


class Uniform(Sampler):
    """
     Samples from uniform distribution
    """

    def sample(
        self, n: int, mins: Sequence[float], maxs: Sequence[float]
    ) -> Tensor:
        tc.manual_seed(self.seed)
        dim = len(mins)
        mint = tc.tensor(mins)
        maxt = tc.tensor(maxs)
        x = mint + tc.rand(n, dim).mul_(maxt - mint)
        return x
