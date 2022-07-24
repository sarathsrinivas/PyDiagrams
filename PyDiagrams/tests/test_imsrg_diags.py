import torch as tc
import numpy as np
from PyQuadrature import Ball_lebedev_gauss_cart
from PyDiagrams import Exch_Stoch_Cart, Free_Space, ZS, Uniform
from itertools import product
import pytest as pyt

tc.set_default_tensor_type(tc.DoubleTensor)

nk = (10,)
nleb = (13, 15, 17)
nr = (10, 15)
kmax = (2.5, 3.0)
kf = (0.5, 1.0, 1.2)
g = (1.0, 2.0)
seed = (22,)

tparam = list(product(nk, nleb, nr, kmax, kf, g, seed))


@pyt.mark.parametrize("nk, nleb, nr, kmax, kf, g, seed", tparam)
def test_zs_contact(
    nk: int, nleb: int, nr: int, kmax: float, kf: float, g: float, seed: int,
) -> None:
    def pot1b(k):
        return 0.5 * k ** 2

    def pot2b(k):
        return g * tc.ones(k.shape[0])

    basis = Exch_Stoch_Cart()
    smplr = Uniform(seed)
    basis.sample(10, nk, kmax, smplr)

    H = Free_Space(basis, pot1b, pot2b)

    ball_quad = Ball_lebedev_gauss_cart(nr=15)

    diag_p = ZS(ball_quad, kf, sgn=1)
    diag_m = ZS(ball_quad, kf, sgn=-1)

    ke = basis.k_2b

    zsp = diag_p.eval(ke, H)
    zsm = diag_m.eval(ke, H)

    zs = zsp - zsm

    zs_comp = (-4.0 / (3 * np.pi ** 2)) * g ** 2 * kf ** 3 * ke[:, 0] ** 2

    assert tc.allclose(zs, zs_comp)
