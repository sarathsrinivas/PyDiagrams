import lib_gpr.sampler as smp
import torch as tc
import numpy as np
import pytest as pyt
from itertools import product
from .bases import *
import sys
sys.path.append('..')

n = (100, 1000)
seed = (1, 45)
kmax = (1.0, 3.0, 5.0)

tparams = list(product(n, kmax, seed))


@pyt.mark.parametrize("n,kmax,seed", tparams)
def test_rotate_dlp(n, kmax, seed):

    sampler = smp.UNIFORM(seed)

    B = GPR_EX_CART(20, n, kmax, sampler)

    kd = B.invariants(B.k_2b)
    ke = B.invariants(B.k_2b_ex)

    kd = tc.stack((kd[:, 0], kd[:, 1], kd[:, 2], kd[:, 3], kd[:, 4], kd[:, 5]))
    ke = tc.stack((ke[:, 1], ke[:, 0], ke[:, 2], ke[:, 3], ke[:, 5], ke[:, 4]))

    assert tc.allclose(kd, ke)
