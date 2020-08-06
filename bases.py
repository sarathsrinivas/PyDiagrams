import lib_quadrature.ball as quad
import torch as tc
import numpy as np
import opt_einsum as oen
import sys
sys.path.append('..')

tc.set_default_tensor_type(tc.DoubleTensor)


class GPR_EX_CART(object):
    "Stochastic Basis with exchange parametrization."

    def __init__(self, n_1b, n_2b, kmax, sampler):

        self.dim_1b = 1
        self.dim_2b = 7
        self.dim_invar = 6

        self.sampler = sampler
        self.k_1b = self.sampler.sample(n_1b, tc.tensor([0.0]),
                                        tc.tensor([1.0]))
        self.k_1b.squeeze_()

        self.kmax = kmax

        self.nq = 20
        self.nleb = 15

        mins = tc.full([self.dim_2b], -kmax)
        mins[0] = 0.0
        maxs = tc.full([self.dim_2b], kmax)

        self.k_2b = self.sampler.sample(n_2b, mins, maxs)

        self.invars = self.invariants(self.k_2b)

        self.k_2b_ex = self.rotate_to_dlp(self.k_2b)

    def invariants(self, k_2b):

        k_2b_l = k_2b.view(-1, self.dim_2b)

        dlz = k_2b_l[:, 0]
        dlpx = k_2b_l[:, 1]
        dlpy = k_2b_l[:, 2]
        dlpz = k_2b_l[:, 3]
        Px = k_2b_l[:, 4]
        Py = k_2b_l[:, 5]
        Pz = k_2b_l[:, 6]

        dl = dlz
        dlp = k_2b_l[:, 1:4].square().sum(1).sqrt_()
        P = k_2b_l[:, 4:].square().sum(1).sqrt_()

        dl_dlp = dlz.mul(dlpz)
        P_dl = dlz.mul(Pz)
        P_dlp = dlpx.mul(Px).add_(dlpy.mul(Py)).add_(dlpz.mul(Pz))

        invar_mom = tc.stack((dl, dlp, P, dl_dlp, P_dl, P_dlp), dim=-1)

        return invar_mom.view(*k_2b.shape[:-1], self.dim_invar)

    def rotate_to_dlp(self, k_2b, phi_P=0.4):

        invar_mom = self.invariants(k_2b)
        invar_mom_l = invar_mom.view(-1, self.dim_invar)

        dl = invar_mom_l[:, 0]
        dlp = invar_mom_l[:, 1]
        P = invar_mom_l[:, 2]
        dl_dlp = invar_mom_l[:, 3]
        P_dl = invar_mom_l[:, 4]
        P_dlp = invar_mom_l[:, 5]

        dl_dlp_ang = dl_dlp.div(dl).div(dlp).acos_()
        P_dl_ang = P_dl.div(P).div(dl).acos_()
        P_dlp_ang = P_dlp.div(P).div(dlp).acos_()

        phi_dl = tc.acos((tc.cos(P_dl_ang) - tc.cos(P_dlp_ang) * tc.cos(dl_dlp_ang)) /
                         (tc.sin(P_dlp_ang) * tc.sin(dl_dlp_ang))) + phi_P

        k_2b_dlp = tc.empty(dl.shape[0], self.dim_2b)

        k_2b_dlp[:, 0] = dlp
        k_2b_dlp[:, 1] = dl.mul(phi_dl.cos()).mul_(dl_dlp_ang.sin())
        k_2b_dlp[:, 2] = dl.mul(phi_dl.sin()).mul_(dl_dlp_ang.sin())
        k_2b_dlp[:, 3] = dl.mul(dl_dlp_ang.cos())
        k_2b_dlp[:, 4] = P.mul(np.cos(phi_P)).mul_(P_dlp_ang.sin())
        k_2b_dlp[:, 5] = P.mul(np.sin(phi_P)).mul_(P_dlp_ang.sin())
        k_2b_dlp[:, 6] = P.mul(P_dlp_ang.cos())

        return k_2b_dlp.view(k_2b.shape)

    def loop_normal_order_1b(self, k_1b, kf):

        q, th_q, phi_q, wt = quad.ball_quad(
            rmax=kf, nr=self.nq, nleb=self.nleb)

        qx, qy, qz = quad.sph_to_cart(q, th_q, phi_q)

        tmp = tc.zeros_like(k_1b)
        dlz = tc.zeros(k_1b.shape[0], qx.shape[0])
        dlpx = (tmp[:, None] - qx[None, :]).mul_(0.5)
        dlpy = (tmp[:, None] - qy[None, :]).mul_(0.5)
        dlpz = (k_1b[:, None] - qz[None, :]).mul_(0.5)
        Px = (tmp[:, None] + qx[None, :]).mul_(0.5)
        Py = (tmp[:, None] + qy[None, :]).mul_(0.5)
        Pz = (k_1b[:, None] + qz[None, :]).mul_(0.5)

        kq_2b = tc.stack((dlz, dlpx, dlpy, dlpz, Px, Py, Pz), -1)

        return kq_2b, wt

    def loop_normal_order_0b(self, kf):

        q, th_q, phi_q, wt_2b = quad.ball_quad(
            rmax=kf, nr=self.nq, nleb=self.nleb)

        qx, qy, qz = quad.sph_to_cart(q, th_q, phi_q)

        n = wt_2b.shape[0]
        dlz = tc.zeros(n, n)

        dlpx = (qx[:, None] - qx[None, :]).mul_(0.5)
        dlpy = (qy[:, None] - qy[None, :]).mul_(0.5)
        dlpz = (qz[:, None] - qz[None, :]).mul_(0.5)

        Px = (qx[:, None] + qx[None, :]).mul_(0.5)
        Py = (qy[:, None] + qy[None, :]).mul_(0.5)
        Pz = (qz[:, None] + qz[None, :]).mul_(0.5)

        kq_2b = tc.stack((dlz, dlpx, dlpy, dlpz, Px, Py, Pz), -1)

        kq_1b, wt_1b = quad.gauss_legendre(self.nq, a=0, b=kf)

        return kq_1b, wt_1b, kq_2b, wt_2b
