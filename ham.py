import torch as tc
from .bases import Basis
from torch import Tensor
from typing import Callable


tc.set_default_tensor_type(tc.DoubleTensor)

POT_2B = Callable[..., Tensor]
POT_1B = Callable[..., Tensor]


class Hamiltonian:
    """
     Base class for Hamiltonians
    """

    def __init__(self, basis: Basis) -> None:
        self.basis: Basis = basis
        self.E: float = NotImplemented
        self.F: Tensor = NotImplemented
        self.V: Tensor = NotImplemented
        return None

    def eval(self, base: Basis) -> None:
        self.E = self.get_0b()
        self.F = self.get_1b(base.k_1b)
        self.V = self.get_2b(base.k_2b)
        return None

    def get_0b(self) -> float:
        raise NotImplementedError

    def get_1b(self, k: Tensor) -> Tensor:
        raise NotImplementedError

    def get_2b(self, k: Tensor) -> Tensor:
        raise NotImplementedError

    def get_1b_loop(self, kl: Tensor, q: Tensor) -> Tensor:
        k = kl[:, None].add(q[None, :])
        F = self.get_1b(k.view(-1))
        return F.view(*k.shape)

    def get_2b_loop(self, kl: Tensor, q: Tensor) -> Tensor:
        k = kl[:, None, :].add(q[None, :, :])
        V = self.get_2b(k.view(-1, kl.shape[-1]))
        return V.view(k.shape[-3], k.shape[-2])


class Free_Space(Hamiltonian):
    """
     Free space Hamiltonian from single particle
     and two particle interaction potentials.
    """

    def __init__(self, basis: Basis, pot_1b: POT_1B, pot_2b: POT_2B) -> None:
        super().__init__(basis)
        self.pot_1b = pot_1b
        self.pot_2b = pot_2b
        return None

    def get_0b(self) -> float:
        return 0

    def get_1b(self, k: Tensor) -> Tensor:
        return self.pot_1b(k)

    def get_2b(self, k: Tensor) -> Tensor:
        invar = self.basis.get_invariants(k)
        return self.pot_2b(invar)
