import lib_gpr.gpr as gp
import torch as tc
import sys
sys.path.append('..')

tc.set_default_tensor_type(tc.DoubleTensor)


class GPR(gp.GPR):
    def __init__(self, x, y, cov, hp=None, **kargs):
        super().__init__(x, y, cov, hp=None, **kargs)

        self.split_covars = NotImplemented

    def get_split_covars(self, xe, xq):

        sig = self.hp[0]
        dim = xe.shape[-1]
        ls = self.hp[None, None, -dim:]

        Aeq = xe[:, None, :].mul(xq[None, :, :]).mul_(2.0).add_(
            xq.square()[None, :, :]).mul_(ls).sum(-1)

        Bes = xe[:, None, :].sub(self.x[None, :, :]).square_().mul_(
            ls).sum(-1)

        Csq = self.x[:, None, :].mul(
            xq[None, :, :]).mul_(-2.0).mul_(ls).sum(-1)

        Aeq.mul_(-1.0).exp_()
        Bes.mul_(-1.0).exp_()
        Csq.mul_(-1.0).exp_().mul_(sig)

        self.A = Aeq
        self.B = Bes
        self.C = Csq

    def interpolate(self, xe, xq):
        if self.need_upd:
            self.krn = self.cov(self.x, hp=self.hp, **self.args)
            self.krnchd = tc.cholesky(self.krn)
            self.wt = tc.squeeze(
                tc.cholesky_solve(self.y.reshape(-1, 1), self.krnchd))
            self.get_split_covars(xe, xq)
            self.need_upd = False

        ys = oen.contract('s,eq,es,sq->eq', wt, self.A,
                          self.B, self.C, backend='torch')

        covars = NotImplemented

        return ys, covars
