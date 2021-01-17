import torch as tc
from torch import Tensor
from typing import List
from .ham import Hamiltonian


class Diagram:
    """
     Base class for diagrammatic terms.
    """

    def __init__(self, H: Hamiltonian, Integ: Integrator) -> None:
        self.H: Hamiltonian = H
        self.Integ: Integrator = Integ
        self.FL: List[Tensor] = NotImplemented
        self.VL: List[Tensor] = NotImplemented
        self.DL: List[Tensor] = NotImplemented
        self.IL: Tensor = NotImplemented
        self.limits: List = NotImplemented
        return None

    def arg_kl_1b(self) -> List[Tensor]:
        raise NotImplementedError

    def arg_q_1b(self) -> List[Tensor]:
        raise NotImplementedError

    def arg_kl_2b(self) -> List[Tensor]:
        raise NotImplementedError

    def arg_q_2b(self) -> List[Tensor]:
        raise NotImplementedError

    def get_D(self) -> Tensor:
        raise NotImplementedError

    def get_loop_vtx(self) -> None:
        kl_1b = self.arg_kl_1b()
        q_1b = self.arg_kl_1b()
        assert len(kl_1b) == len(q_1b)
        self.FL = [self.H.get_1b_loop(kl, q) for kl, q in zip(kl_1b, q_1b)]

        kl_2b = self.arg_kl_2b()
        q_2b = self.arg_kl_2b()
        assert len(kl_2b) == len(q_2b)
        self.VL = [self.H.get_2b_loop(kl, q) for kl, q in zip(kl_2b, q_2b)]

        self.DL = self.get_D()

        return None

    def get_integrand(self) -> Tensor:
        return NotImplementedError

    def eval(self) -> Tensor:
        raise NotImplementedError
