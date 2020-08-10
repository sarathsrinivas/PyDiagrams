import lib_gpr.gpr as gp
import lib_gpr.covar as cv
import torch as tc
import pytest as pyt
import opt_einsum as oen
from itertools import product
from .gpr_split_covar import GPR
import sys
sys.path.append('..')

tc.set_default_tensor_type(tc.DoubleTensor)

ns = (5, 10, 100)
nq = (4, 9, 50)
dim = (3, 7)
cov = (cv.sq_exp, cv.sq_exp_noise)

tparam = list(product(ns, nq, dim, cov))


@pyt.mark.parametrize("ns,nq,dim,cov", tparam)
def test_split_covars(ns, nq, dim, cov):
    xs = tc.rand(ns, dim)
    xq = tc.rand(nq, dim)
    xe = tc.rand(ns, dim)

    y = tc.exp(xs.sum(-1))

    GP = GPR(xs, y, cov)

    GP.get_split_covars(xe, xq)

    krn_split = oen.contract('eq,es,sq->eqs', GP.A,
                             GP.B, GP.C, backend='torch')

    xeq = xe[:, None, :].add(xq[None, :, :])

    hp = cov(xs, xs=xeq)
    krn = cov(xs, xs=xeq, hp=hp)

    assert tc.allclose(krn, krn_split)


@pyt.mark.parametrize("ns,nq,dim,cov", tparam)
def test_interpolate(ns, nq, dim, cov):
    xs = tc.rand(ns, dim)
    xq = tc.rand(nq, dim)
    xe = tc.rand(ns, dim)

    y = tc.exp(xs.sum(-1))

    xeq = xe[:, None, :].add(xq[None, :, :])

    GP = gp.GPR(xs, y, cov)

    yeq, var_eq = GP.interpolate(xeq.view(-1, dim))

    GP_split = GPR(xs, y, cov)

    yeq_split, var_eq_split = GP_split.interpolate(xe, xq)

    assert tc.allclose(yeq, yeq_split.view(-1))
