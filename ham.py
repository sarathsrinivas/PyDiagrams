import torch as tc
import numpy as np

tc.set_default_tensor_type(tc.DoubleTensor)


class HAMILTONIAN(object):
    def __init__(self, basis, op_1b=None, pot=None, **pot_args):

        self.basis = basis
        self.E = 0
        if op_1b is None:
            self.F = None
            self.V = None
        else:
            self.F = basis.get_1b_op(basis.k_1b, op=op_1b, **pot_args)
            self.V = basis.get_2b_op(basis.k_2b, pot=pot, **pot_args)
            self.pot_args = pot_args

    def normal_order(self, kf):
        self.E = basis.normal_order_0b(op_1b=op_1b, op_2b=pot, **self.pot_args)
        self.f = basis.normal_order_1b(self.basis.k_1b,
                                       kf,
                                       op_1b=op_1b,
                                       op_2b=pot,
                                       **self.pot_args)


class HAM_GPR(HAMILTONIAN):
    def __init__(self,
                 basis,
                 kf,
                 cov=None,
                 cov_args=None,
                 op_1b=None,
                 pot=None,
                 **pot_args):

        super().__init__(basis, op_1b=op_1b, pot=pot, **pot_args)

        super().normal_order(kf)

        self.F_gp = gp.GPR(self.basis.k_1b, self.F, cov, **cov_args)
        self.V_gp = gp.GPR(self.basis.k_2b, self.V, cov, **cov_args)

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
                 cov=None,
                 cov_args=None,
                 op_1b=None,
                 pot=None,
                 **pot_args):

        HAMILTONIAN.__init__(basis, op_1b=op_1b, pot=pot, **pot_args)

        super().normal_order(kf)

        self.F_gp = gp.GRBCM(self.basis.k_1b, self.F, self.basis.k_1b_g,
                             self.Fg, cov, **cov_args)
        self.V_gp = gp.GRBCM(self.basis.k_2b, self.V, self.basis.k_2b_g,
                             self.Vg, cov, **cov_args)
