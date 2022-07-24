from torch import Tensor
from typing import List
from .ham import Hamiltonian
from PyQuadrature import Integrator


class Diagram:
    """
     Base class for diagrammatic terms.
    """

    def __init__(self, integ: Integrator) -> None:
        self.integ: Integrator = integ
        self.x = NotImplemented
        self.wt = NotImplemented
        self.FL: List[Tensor] = NotImplemented
        self.VL: List[Tensor] = NotImplemented
        self.DL: Tensor = NotImplemented
        self.IL: Tensor = NotImplemented
        self.limits: dict = NotImplemented
        return None

    def arg_kl_1b(self, ke: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def arg_q_1b(self) -> List[Tensor]:
        raise NotImplementedError

    def arg_kl_2b(self, ke: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def arg_q_2b(self) -> List[Tensor]:
        raise NotImplementedError

    def get_D(self, ke: Tensor) -> Tensor:
        raise NotImplementedError

    def get_loop_vtx(self, ke: Tensor, H: Hamiltonian) -> None:
        kl_1b = self.arg_kl_1b(ke)
        q_1b = self.arg_q_1b()
        assert len(kl_1b) == len(q_1b)
        self.FL = [H.get_1b_loop(kl, q) for kl, q in zip(kl_1b, q_1b)]

        kl_2b = self.arg_kl_2b(ke)
        q_2b = self.arg_q_2b()
        assert len(kl_2b) == len(q_2b)
        self.VL = [H.get_2b_loop(kl, q) for kl, q in zip(kl_2b, q_2b)]

        self.DL = self.get_D(ke)

        return None

    def get_integrand(self) -> Tensor:
        raise NotImplementedError

    def eval(self, ke: Tensor, H: Hamiltonian) -> Tensor:
        self.get_loop_vtx(ke, H)
        Ikq = self.get_integrand()
        return self.integ.integrate(Ikq, self.wt)
