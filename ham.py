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

    def __init__(self) -> None:
        self.E: float = NotImplemented
        self.F: Tensor = NotImplemented
        self.V: Tensor = NotImplemented
        return None

    def eval(self, base: Basis) -> None:
        self.E = self.get_E(base)
        self.F = self.get_F(base)
        self.V = self.get_V(base)
        return None

    def get_E(self, base: Basis) -> float:
        raise NotImplementedError

    def get_F(self, base: Basis) -> Tensor:
        raise NotImplementedError

    def get_V(self, base: Basis) -> Tensor:
        raise NotImplementedError


class Free_Space(Hamiltonian):
    """
     Free space Hamiltonian from single particle
     and two particle interaction potentials.
    """

    def __init__(self, pot_1b: POT_1B, pot_2b: POT_2B) -> None:
        super().__init__()
        self.pot_1b = pot_1b
        self.pot_2b = pot_2b
        return None

    def get_E(self, base: Basis) -> float:
        return 0

    def get_F(self, base: Basis) -> Tensor:
        return self.pot_1b(base.k_1b)

    def get_V(self, base: Basis) -> Tensor:
        invar = base.get_invariants()
        return self.pot_2b(invar)
