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
        self.sampler = sampler
        self.k_1b = self.sampler.sample(n_1b, tc.tensor([0.0]),
                                        tc.tensor([1.0]))
        self.k_1b.squeeze_()

        self.dim_2b = 7

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

        dlz = k_2b[:, 0]
        dlpx = k_2b[:, 1]
        dlpy = k_2b[:, 2]
        dlpz = k_2b[:, 3]
        Px = k_2b[:, 4]
        Py = k_2b[:, 5]
        Pz = k_2b[:, 6]

        dl = k_2b[:, 0]
        dlp = k_2b[:, 1:4].square().sum(1).sqrt_()
        P = k_2b[:, 4:].square().sum(1).sqrt_()

        dl_dlp = dlz.mul(dlpz).div_(dl).div_(dlp).acos_()
        P_dl = dlz.mul(Pz).div_(dl).div_(P).acos_()
        P_dlp = dlpx.mul(Px).add_(dlpy.mul(Py)).add_(
            dlpz.mul(Pz)).div_(dlp).div_(P).acos_()

        return dl, dlp, P, dl_dlp, P_dl, P_dlp

    def rotate_to_dlp(self, k_2b, phi_P=0.4):

        dl, dlp, P, dl_dlp, P_dl, P_dlp = self.invariants(k_2b)

        phi_dl = tc.acos((tc.cos(P_dl) - tc.cos(P_dlp) * tc.cos(dl_dlp)) /
                         (tc.sin(P_dlp) * tc.sin(dl_dlp))) + phi_P

        k_2b_dlp = tc.empty_like(k_2b)

        k_2b_dlp[:, 0] = dlp
        k_2b_dlp[:, 1] = dl.mul(phi_dl.cos()).mul_(dl_dlp.sin())
        k_2b_dlp[:, 2] = dl.mul(phi_dl.sin()).mul_(dl_dlp.sin())
        k_2b_dlp[:, 3] = dl.mul(dl_dlp.cos())
        k_2b_dlp[:, 4] = P.mul(np.cos(phi_P)).mul_(P_dlp.sin())
        k_2b_dlp[:, 5] = P.mul(np.sin(phi_P)).mul_(P_dlp.sin())
        k_2b_dlp[:, 6] = P.mul(P_dlp.cos())

        return k_2b_dlp

    def loop_normal_order_1b(self, k_1b, kf):

        q, th_q, phi_q, wt = quad.ball_quad(
            rmax=kf, nr=self.nq, nleb=self.nleb)

        qx, qy, qz = quad.sph_to_cart(q, th_q, phi_q)

        tmp = tc.zeros_like(self.k_1b)
        dlz = tc.zeros(self.k_2b.shape[0], qx.shape[0])
        dlpx = (tmp - qx).mul_(0.5)
        dlpy = (tmp - qy).mul_(0.5)
        dlpz = (k_1b - qz).mul_(0.5)
        Px = (tmp + qx).mul_(0.5)
        Py = (tmp + qy).mul_(0.5)
        Pz = (k_1b + qz).mul_(0.5)

        kq_2b = tc.stack((dlz, dlpx, dlpy, dlpz, Px, Py, Pz), -1)

        kq_2b = kq_2b.reshape(-1, self.dim_2b)

        invar = self.invariants(kq_2b)

        return invar, wt

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

        loop_2b = self.invariants(kq_2b.reshape(-1, self.dim_2b))

        loop_1b, wt_1b = quad.gauss_legendre(self.nq, a=0, b=kf)

        return loop_1b, wt_1b, loop_2b, wt_2b
