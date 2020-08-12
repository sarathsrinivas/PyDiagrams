import lib_gpr.gpr as gp
import lib_gpr.gr_bcm as gr
import torch as tc
import opt_einsum as oen
import sys
sys.path.append('..')

tc.set_default_tensor_type(tc.DoubleTensor)


class GPR(gp.GPR):
    def __init__(self, x, y, cov, hp=None, **kargs):
        super().__init__(x, y, cov, hp=None, **kargs)

        self.split_covars = NotImplemented
        self.krns = NotImplemented
        self.var_samples = 10

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
        Csq.mul_(-1.0).exp_().mul_(sig**2)

        self.A = Aeq
        self.B = Bes
        self.C = Csq

    def get_pred_covar(self, xe, xq, samples=10, hp=None):
        if hp is None:
            hp = self.hp

        assert samples <= xe.shape[0]

        self.var_samples = samples

        xlp = xe[:samples, None, :].add(xq[None, :, :])

        if self.krns is NotImplemented:
            self.krns = self.cov(self.x, xs=xlp, hp=hp)

        if self.need_upd:
            self.krns = self.cov(self.x, xs=xlp, hp=hp)

        covars_avg = tc.zeros(xq.shape[0], xq.shape[0])

        for s in range(0, samples):
            covars = super().get_pred_covar(
                xlp[s, :, :], self.krns[s, :, :], hp=hp)
            covars_avg.add_(covars)

        covars_avg.div_(samples)

        return covars_avg

    def interpolate(self, xe, xq, skip_var=False):
        if self.need_upd:
            self.krn = self.cov(self.x, hp=self.hp, **self.args)
            self.krnchd = tc.cholesky(self.krn)
            self.wt = tc.squeeze(
                tc.cholesky_solve(self.y.reshape(-1, 1), self.krnchd))
            self.get_split_covars(xe, xq)
            self.need_upd = False

        # ys = oen.contract('s,eq,es,sq->eq', self.wt, self.A,
        #                  self.B, self.C, backend='torch')

        ys = tc.mm(self.B, self.C.mul(self.wt[:, None])).mul_(self.A)

        if not skip_var:
            covars = self.get_pred_covar(
                xe, xq, samples=self.var_samples, hp=self.hp)
            return ys, covars
        else:
            return ys
