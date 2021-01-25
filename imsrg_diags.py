import torch as tc
import numpy as np
from torch import Tensor
from .diag import Diagram
from typing import List
from lib_quadrature.integrator import Integrator

tc.set_default_tensor_type(tc.DoubleTensor)


class ZS(Diagram):
    """
     ZS diagram for 2b vertex
    """

    def __init__(self, integ: Integrator, kf: float, sgn: int) -> None:
        super().__init__(integ)
        self.sgn = sgn
        self.limits = {"rmax": kf}
        self.integ.get_quadrature(self.limits)
        self.n_1b = 6
        self.n_2b = 2
        return None

    def arg_kl_1b(self, ke: Tensor) -> List[Tensor]:
        zr = tc.zeros(ke.shape[-2])
        dl = tc.stack([ke[:, 0], zr, zr], 1)
        dlp = ke[:, 1:4]
        P = ke[:, 4:]

        kl = [
            -1 * dl + self.sgn * dl,
            1 * dl + self.sgn * dl,
            P + dlp + dl,
            P - dlp - dl,
            P + dlp - dl,
            P - dlp + dl,
        ]

        assert len(kl) == self.n_1b

        return kl

    def arg_q_1b(self) -> List[Tensor]:
        q = self.integ.x
        zr = tc.zeros_like(q)

        ql = [q, q, zr, zr, zr, zr]

        assert len(ql) == self.n_1b

        return ql

    def arg_kl_2b(self, ke: Tensor) -> List[Tensor]:

        dl = ke[:, 0].view(-1, 1)
        dlp = ke[:, 1:4]
        P = ke[:, 4:]

        kl1 = tc.cat(
            [
                1 * dl,
                0.5 * (P + dlp - self.sgn * dl),
                0.5 * (P + dlp + self.sgn * dl),
            ],
            dim=-1,
        )

        kl2 = tc.cat(
            [
                1 * dl,
                0.5 * (P - dlp - self.sgn * dl),
                0.5 * (P - dlp + self.sgn * dl),
            ],
            dim=-1,
        )

        kl = [kl1, kl2]

        assert len(kl) == self.n_2b

        return kl

    def arg_q_2b(self) -> List[Tensor]:
        q = self.integ.x
        zr = tc.zeros(q.shape[0], 1)

        ql1 = tc.cat([zr, -0.5 * q, 0.5 * q], dim=-1)
        ql2 = tc.cat([zr, -0.5 * q, 0.5 * q], dim=-1)

        ql = [ql1, ql2]

        assert len(ql) == self.n_2b

        return ql

    def get_D(self, ke: Tensor) -> Tensor:
        return 1.0

    def get_integrand(self) -> Tensor:
        f = self.FL
        v = self.VL

        VI = v[0].mul(v[1])
        FI = 2 * (f[0] - f[1]) + f[2] - f[3] - f[4] + f[5]

        prefac = 1 / (8 * np.pi ** 3)

        self.IL = VI.mul_(FI).mul_(prefac)

        return self.IL
