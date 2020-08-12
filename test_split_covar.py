import lib_gpr.gpr as gp
import lib_gpr.gr_bcm as grbcm
import lib_gpr.covar as cv
import torch as tc
import pytest as pyt
import opt_einsum as oen
from itertools import product
from .gpr_split_covar import GPR
from .grbcm_split_covar import GRBCM
import sys
sys.path.append('..')

tc.set_default_tensor_type(tc.DoubleTensor)

ns = (10, 50, 100)
nq = (4, 9, 50)
dim = (3, 7)
cov = (cv.sq_exp, cv.sq_exp_noise)

tparam = list(product(ns, nq, dim, cov))


@pyt.mark.parametrize("ns,nq,dim,cov", tparam)
def test_split_covars_gpr(ns, nq, dim, cov):
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
def test_interpolate_gpr(ns, nq, dim, cov):
    xs = tc.rand(ns, dim)
    xq = tc.rand(nq, dim)
    xe = tc.rand(ns, dim)

    y = tc.exp(xs.sum(-1))

    xeq = xe[:, None, :].add(xq[None, :, :])

    GP = gp.GPR(xs, y, cov)

    yeq = GP.interpolate(xeq.view(-1, dim), skip_var=True)

    GP_split = GPR(xs, y, cov)

    yeq_split, var_eq_split = GP_split.interpolate(xe, xq)

    assert tc.allclose(yeq, yeq_split.view(-1))


@pyt.mark.parametrize("ns,nq,dim,cov", tparam)
def test_pred_covar_gpr(ns, nq, dim, cov):
    x = tc.rand(dim)
    xs = tc.rand(ns, dim)
    xq = tc.rand(nq, dim)

    y = tc.exp(xs.sum(-1))

    xe = tc.empty(ns, dim).copy_(x)

    xxq = xq[:, :].add(x[None, :])

    GP = gp.GPR(xs, y, cov)

    yeq, var_eq = GP.interpolate(xxq)

    GP_split = GPR(xs, y, cov)

    yeq_split, var_eq_split = GP_split.interpolate(xe, xq)

    assert tc.allclose(var_eq_split, var_eq)


ng = (10,)
nc = (3, 5)
ns = (5, 10, 20)
nq = (4, 9, 50)
dim = (3, 7)
cov = (cv.sq_exp, cv.sq_exp_noise)

tparam = list(product(ng, nc, ns, nq, dim, cov))


@ pyt.mark.parametrize("ng,nc,ns,nq,dim,cov", tparam)
def test_split_covars_grbcm(ng, nc, ns, nq, dim, cov):
    xl = tc.rand(nc, ns, dim)
    xq = tc.rand(nq, dim)
    xe = tc.rand(nc, ns, dim)
    xg = tc.rand(ng, dim)

    yl = tc.exp(xl.sum(-1))
    yg = tc.exp(xg.sum(-1))

    GP = GRBCM(xl, yl, xg, yg, cov)

    GP.get_split_covars_local(xe, xq)

    krn_split = oen.contract('ceq,ces,csq->ceqs', GP.Al,
                             GP.Bl, GP.Cl, backend='torch')

    xceq = xe[:, :, None, :].add(xq[None, None, :, :]).reshape(-1, dim)

    hp = cov(GP.x, xs=xceq)
    krn = cov(GP.x, xs=xceq, hp=hp).reshape_as(krn_split)

    assert tc.allclose(krn, krn_split)
