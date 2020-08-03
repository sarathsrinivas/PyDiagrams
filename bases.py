import torch as tc
import numpy as np
import opt_einsum as oen
import sys
sys.path.append('..')
import lib_gpr.sampler as smp
import lib_pots.pot as pots

tc.set_default_tensor_type(tc.DoubleTensor)


class GPR_EX_CART(object):
    "Stochastic Basis with exchange parametrization."

    def __init__(self, n, kmax, seed):

        self.dim_1b = 1
        self.k_1b, min_dist = smp.nsample_repulsion(n, tc.tensor([0.0]),
                                                    tc.tensor([1.0]))
        self.k_1b.squeeze_()

        self.dim_2b = 6

        self.kmax = kmax

        mins = tc.full(self.dim_2b, -kmax)
        mins[0] = 0.0
        maxs = tc.full(self.dim_2b, kmax)

        self.k_2b, min_dist = smp.nsample_repulsion(n, mins, maxs)

        self.dl, self.dlp, self.dl_dlp, slef.P_dl, self.P_dlp = self.invariants(
            self.k_2b)

        self.k_2b_ex = self.rotate_to_dlp()

    def invariants(self, k_2b):

        dlz = k_2b[:, 0]
        dlpx = k_2b[:, 1]
        dlpy = k_2b[:, 2]
        dlpz = k_2b[:, 3]
        Px = k_2b[:, 4]
        Py = k_2b[:, 5]
        Pz = k_2b[:, 6]

        dl = k_2b[:, 0]
        dlp = k_2b[:, 1:3].square().sum(1).sqrt_()
        P = k_2b[:, 3:].square().sum(1).sqrt_()

        dl_dlp = dlz.mul(dlpz).div_(dl).div_(dlp).acos_()
        P_dl = dlz.mul(Pz).div_(dl).div_(P).acos_()
        P_dlp = dlpx.mul(Px).add_(dlpy.mul(Py)).add_(dlpz.mul(Pz)).acos_()

        return dl, dlp, P, dl_dlp, P_dl, P_dlp

    def rotate_to_dlp(self, phi_P=0.4):
        phi_dl = tc.acos(
            (tc.cos(self.P_dl) - tc.cos(self.P_dlp) * tc.cos(self.dl_dlp)) /
            (tc.sin(self.P_dlp) * tc.sin(self.dl_dlp))) + phi_P

        k_2b_dlp = tc.empty_like(self.k_2b)

        k_2b_dlp[:, 0] = self.dl.mul(phi_dl.cos()).mul_(self.dl_dlp.sin())
        k_2b_dlp[:, 1] = self.dl.mul(phi_dl.cos()).mul_(self.dl_dlp.sin())
        k_2b_dlp[:, 2] = self.dl.mul(dl_dlp.cos())
        k_2b_dlp[:, 3] = self.P.mul(np.cos(phi_P)).mul_(self.P_dlp.sin())
        k_2b_dlp[:, 4] = self.P.mul(np.sin(phi_P)).mul_(self.P_dlp.sin())
        k_2b_dlp[:, 5] = self.P.mul(self.P_dlp.cos())

        return k_2b_dlp

    def get_1b_op(self, k_1b, op=None, **kwargs):
        return op(k_1b, **kwargs)

    def get_2b_op(self, k_2b, pot=None, **kwargs):

        invar = self.invariants(k_2b)

        v = pot(*invar, self.kmax, **kwargs)

        return v

    def normal_order_1b(self, k_1b, kf, op_1b=None, op_2b=None, nq=20,
                        nleb=15):

        q, th_q, phi_q, w = quad.ball_quad(rmax=kf, nr=nq, nleb=nleb)

        qx, qy, qz = quad.sph_to_cart(q, th_q, phi_q)

        tmp = tc.zeros_like(self.k_1b)
        dlz = tc.zeros(n, nq)
        dlpx = (tmp - qx).mul_(0.5)
        dlpy = (tmp - qy).mul_(0.5)
        dlpz = (k_1b - qz).mul_(0.5)
        Px = (tmp + qx).mul_(0.5)
        Py = (tmp + qy).mul_(0.5)
        Pz = (k_1b + qz).mul_(0.5)

        kq_2b = tc.stack((dlz, dlpx, dlpy, dlpz, Px, Py, Pz), -1)

        invar = self.invariants(kq_2b)

        f = op_1b(k_1b, **kwargs)
        v = op_2b(*invar, self.kmax, **kwargs)

        f_no = f.add_(v.mul_(wt).sum(-1))

        return f_no

    def normal_order_0b(self, op_1b=None, op_2b=None, **kwargs):
        q, th_q, phi_q, wt = quad.ball_quad(rmax=kf, nr=nq, nleb=nleb)
        qx, qy, qz = quad.sph_to_cart(q, th_q, phi_q)

        n = wt.shape[0]
        dlz = tc.zeros(n, n)

        dlpx = (qx[:, None] - qx[None, :]).mul_(0.5)
        dlpy = (qy[:, None] - qy[None, :]).mul_(0.5)
        dlpz = (qz[:, None] - qz[None, :]).mul_(0.5)

        Px = (qx[:, None] + qx[None, :]).mul_(0.5)
        Py = (qy[:, None] + qy[None, :]).mul_(0.5)
        Pz = (qz[:, None] + qz[None, :]).mul_(0.5)

        kq_2b = tc.stack((dlz, dlpx, dlpy, dlpz, Px, Py, Pz), -1)

        invar = self.invariants(kq_2b)

        k_1b, wt_1b = quad.gauss_legendre(nq, a=0, b=kf)

        f = op_1b(k_1b, **kwargs)
        v = op_2b(*invar, self.kmax, **kwargs)

        v_int = oen.contract('i,j,ij->', wt, wt, v)

        e_no = tc.dot(f, wt_1b) + 0.5 * v_int

        return e_no
