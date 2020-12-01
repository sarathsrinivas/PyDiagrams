import torch as tc
from torch import Tensor
from .sampler import Sampler
import numpy as np

tc.set_default_tensor_type(tc.DoubleTensor)


class Basis:
    """
     Base class for Hamiltonian basis
    """

    def __init__(self):
        self.dim: int = NotImplemented
        self.dim_1b: int = NotImplemented
        self.dim_2b: int = NotImplemented

        self.k_1b: Tensor = NotImplemented
        self.k_2b: Tensor = NotImplemented

    def get_invariants(self) -> Tensor:
        raise NotImplementedError

    def from_invariants(self, inv: Tensor) -> None:
        raise NotImplementedError


def time_reversal(base: Basis) -> Basis:
    pass


def parity_reversal(base: Basis) -> Basis:
    pass


def particle_exchange(base: Basis) -> Basis:
    invar = base.get_invariants()

    dl, dlp, P, dl_dlp, dl_P, dlp_P = tc.split(invar, 1, dim=-1)

    invar_ex = tc.cat([dlp, dl, P, dl_dlp, dlp_P, dl_P], dim=-1)

    base_ex = base.__class__()
    base_ex.from_invariants(invar_ex)

    return base_ex


class Exch_Stoch_Cart(Basis):
    """
     Exchage transfer basis (dl, dlp, P)
    """

    def __init__(self) -> None:
        super().__init__()
        self.k_2b_dlp: Tensor = NotImplemented
        self.k_2b_P: Tensor = NotImplemented
        return None

    def sample(
        self, n_1b: int, n_2b: int, kmax: float, sampler: Sampler
    ) -> None:
        self.k_1b = sampler.sample(n_1b, [0], [kmax])

        mins = [0.0, -kmax, -kmax, -kmax, -kmax, -kmax, -kmax]
        maxs = [kmax, kmax, kmax, kmax, kmax, kmax, kmax]

        self.k_2b = sampler.sample(n_2b, mins, maxs)

    def get_invariants(self) -> Tensor:
        k_2b = self.k_2b

        dlz = k_2b[:, 0]
        dlp = k_2b[:, 1:4]
        P = k_2b[:, 4:]
        dl = tc.zeros_like(dlp)
        dl[:, 2] = dlz

        mod_dl = dlz
        mod_dlp = dlp.square().sum(-1).sqrt_()
        mod_P = P.square().sum(-1).sqrt_()

        dl_dlp = dl.mul(dlp).sum(-1)
        dl_P = dl.mul(P).sum(-1)
        dlp_P = dlp.mul(P).sum(-1)

        invar = tc.stack([mod_dl, mod_dlp, mod_P, dl_dlp, dl_P, dlp_P], dim=-1)

        return invar

    def from_invariants(self, invar: Tensor) -> None:
        self.k_2b = invar_to_cart(invar)
        return None


def invar_to_cart(invar: Tensor, phi: float = 0.3) -> Tensor:
    """
     Convert invariant (a, b, c, a.b, a.c, b.c)
     to (az, bx, by, bz, cx, cy, cz) where atan(by/bx) = phi.
    """

    a = invar[:, 0]
    b = invar[:, 1]
    c = invar[:, 2]

    ab = invar[:, 3]
    ac = invar[:, 4]
    bc = invar[:, 5]

    ab_ang = ab.div(a).div_(b).acos_()
    ac_ang = ac.div(a).div_(c).acos_()
    bc_ang = bc.div(b).div_(c).acos_()

    az = a
    bx = b.mul(np.cos(phi)).mul_(ab_ang.sin())
    by = b.mul(np.sin(phi)).mul_(ab_ang.sin())
    bz = b.mul(ab_ang.cos())

    phi_c = (
        tc.acos(
            (tc.cos(bc_ang) - tc.cos(ab_ang) * tc.cos(ac_ang))
            / (tc.sin(ab_ang) * tc.sin(ac_ang))
        )
        + phi
    )

    # phi_c = (
    #    bc_ang.cos()
    #    .sub_(ab_ang.cos().mul_(ac_ang.cos()))
    #    .div_(ab_ang.sin().mul_(ac_ang.sin()))
    # )

    cx = c.mul(phi_c.cos()).mul_(ac_ang.sin())
    cy = c.mul(phi_c.sin()).mul_(ac_ang.sin())
    cz = c.mul(ac_ang.cos())

    k_2b = tc.stack([az, bx, by, bz, cx, cy, cz], dim=-1)

    return k_2b
