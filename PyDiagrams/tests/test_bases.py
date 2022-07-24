import torch as tc
import pytest as pyt
from itertools import product
from PyDiagrams import Uniform, Exch_Stoch_Cart, particle_exchange


n = (100, 1000)
seed = (1, 45)
kmax = (1.0, 3.0, 5.0)

tparams = list(product(n, kmax, seed))


@pyt.mark.parametrize("n,kmax,seed", tparams)
def test_invar(n: int, kmax: float, seed: int) -> None:
    uni = Uniform(seed)
    A = Exch_Stoch_Cart()
    A.sample(10, n, kmax, uni)

    invar = A.get_invariants()

    B = Exch_Stoch_Cart()

    B.from_invariants(invar)

    assert tc.allclose(invar, B.get_invariants())


@pyt.mark.parametrize("n,kmax,seed", tparams)
def test_particle_exchange(n: int, kmax: float, seed: int) -> None:
    uni = Uniform(seed)
    A = Exch_Stoch_Cart()
    A.sample(10, n, kmax, uni)

    B = particle_exchange(A)

    C = particle_exchange(B)

    assert tc.allclose(A.get_invariants(), C.get_invariants())
