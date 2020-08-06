import lib_gpr as gp
import torch as tc
import numpy as np
import opt_einsum as oen
import sys
sys.path.append('..')

tc.set_default_tensor_type(tc.DoubleTensor)


class HAMILTONIAN(object):
    def __init__(self, basis, pot=None, op_1b=None, **op_1b_args):

        self.basis = basis
        self.pot = pot
        self.E = 0
        if op_1b is None:
            self.F = None
            self.V = None
        else:
            self.op_1b = op_1b
            self.op_1b_args = op_1b_args

            self.F = self.op_1b(basis.k_1b, **self.op_1b_args)
            self.V = self.pot.eval(*basis.invars)

    def normal_order(self, kf):

        loop_2b, wt_2b = self.basis.loop_normal_order_1b(self.basis.k_1b, kf)

        n = wt_2b.shape[0]

        f = self.op_1b(self.basis.k_1b, **self.op_1b_args)
        v = self.pot.eval(*loop_2b)

        self.F = f.add_(v.mul_(wt_2b).sum(-1))

        loop_1b, wt_1b, loop_2b, wt_2b = self.basis.loop_normal_order_0b(kf)

        f = self.op_1b(loop_1b, **self.op_1b_args)
        v = self.pot.eval(*loop_2b)

        v_fold = oen.contract('i,j,ij->', wt_2b, wt_2b,
                              v.reshape(n, n), backend='torch')

        self.E = tc.dot(f, wt_1b) + 0.5 * v_fold


class HAM_GPR(HAMILTONIAN):
    def __init__(self,
                 basis,
                 kf,
                 pot,
                 cov=None,
                 cov_args=None,
                 op_1b=None,
                 op_1b_args=None):

        super().__init__(basis, pot=pot, op_1b=op_1b, **op_1b_args)

        super().normal_order(kf)

        self.F_gp = gp.gpr.GPR(self.basis.k_1b, self.F, cov, **cov_args)
        self.V_gp = gp.gpr.GPR(self.basis.k_2b, self.V, cov, **cov_args)

    def train(self, method='CG', jac=True):
        res_F = self.F_gp.train(method=method, jac=jac)
        res_V = self.V_gp.train(method=method, jac=jac)

        return res_F, res_V

    def interpolate(self, basis):
        H_pred = HAMILTONIAN(basis)
        H_pred.F, H_pred.var_F = self.F_gp.interpolate(basis.k_1b)
        H_pred.V, H_pred.var_V = self.V_gp.interpolate(basis.k_2b)

        return H_pred


class HAM_GRBCM(HAM_GPR):
    def __init__(self,
                 basis,
                 kf,
                 pot,
                 cov=None,
                 cov_args=None,
                 op_1b=None,
                 op_1b_args=None):

        HAMILTONIAN.__init__(self, basis, pot=pot, op_1b=op_1b, **op_1b_args)

        self.Fg = self.op_1b(basis.k_1b_g, **self.op_1b_args)
        self.Vg = self.pot.eval(*basis.invar_g)

        super().normal_order(kf)

        self.F_gp = gp.gr_bcm.GRBCM(self.basis.k_1b, self.F, self.basis.k_1b_g,
                                    self.Fg, cov, **cov_args)
        self.V_gp = gp.gr_bcm.GRBCM(self.basis.k_2b, self.V, self.basis.k_2b_g,
                                    self.Vg, cov, **cov_args)
