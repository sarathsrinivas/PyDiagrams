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

    def interpolate(self, xe, xq):
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

        covars = NotImplemented

        return ys, covars


class GRBCM(gr.GRBCM):
    def __init__(self, xl, yl, xg, yg, cov, hp=None, **kargs):
        super().__init__(xl, yl, xg, yg, cov, hp=None, **kargs)

        self.split_covars_local = NotImplemented
        self.split_covars_global = NotImplemented

    def get_split_covars_global(self, xel, xq):

        sig = self.hpg[0]
        dim = xel.shape[-1]
        ls = self.hpg[None, None, -dim:]
        xe = xel.reshape(-1, dim)

        Aeq = xe[:, None, :].mul(xq[None, :, :]).mul_(2.0).add_(
            xq.square()[None, :, :]).mul_(ls).sum(-1)

        Bes = xe[:, None, :].sub(self.xg[None, :, :]).square_().mul_(
            ls).sum(-1)

        Csq = self.xg[:, None, :].mul(
            xq[None, :, :]).mul_(-2.0).mul_(ls).sum(-1)

        Aeq.mul_(-1.0).exp_()
        Bes.mul_(-1.0).exp_()
        Csq.mul_(-1.0).exp_().mul_(sig**2)

        self.Ag = Aeq
        self.Bg = Bes
        self.Cg = Csq

    def get_split_covars_local(self, xel, xq):

        sig = self.hp[:, None, None, 0]
        dim = xel.shape[-1]
        ls = self.hp[:, -dim:]
        xe = xel.reshape(-1, dim)

        Aceq = xe[None, :, None, :].mul(xq[None, None, :, :]).mul_(2.0).add_(
            xq.square()[None, None, :, :]).mul(ls[:, None, None, :]).sum(-1)

        Bces = xe[None, :, None, :].sub(self.x[:, None, :, :]).square_().mul_(
            ls[:, None, None, :]).sum(-1)

        Ccsq = self.x[:, :, None, :].mul(
            xq[None, None, :, :]).mul_(-2.0).mul_(ls[:, None, None, :]).sum(-1)

        Aceq.mul_(-1.0).exp_()
        Bces.mul_(-1.0).exp_()
        Ccsq.mul_(-1.0).exp_().mul_(sig**2)

        self.Al = Aceq
        self.Bl = Bces
        self.Cl = Ccsq

    def interpolate_local(self, xel, xq):
        if self.need_upd:
            self.krn = self.cov(self.x, hp=self.hp, **self.args)
            self.krnchd = tc.cholesky(self.krn)
            y = self.y.view(-1, self.y.shape[-1], 1)
            self.wt = tc.cholesky_solve(y, self.krnchd)
            self.get_split_covars_local(xel, xq)
            self.need_upd = False

        # ys = oen.contract('cs,ceq,ces,csq->ceq', self.wt.squeeze(-1), self.Al,
        #                  self.Bl, self.Cl, backend='torch')

        ys = tc.bmm(self.Bl, self.Cl.mul(self.wt[:, :, None])).mul_(self.Al)

        covars = NotImplemented

        return ys, covars

    def interpolate_global(self, xel, xq):
        if self.need_upd_g:
            self.krng = self.cov(self.xg, hp=self.hpg, **self.args)
            self.krnchdg = tc.cholesky(self.krng)
            self.wtg = tc.squeeze(
                tc.cholesky_solve(self.yg.reshape(-1, 1), self.krnchdg))
            self.get_split_covars_global(xel, xq)
            self.need_upd_g = False

        # ys = oen.contract('s,eq,es,sq->eq', self.wtg, self.Ag,
        #                  self.Bg, self.Cg, backend='torch')

        ys = tc.mm(self.Bg, self.Cg.mul(self.wt[:, None])).mul_(self.Ag)

        covars = NotImplemented

        return ys, covars

    def aggregate(self, ys_g, covars_g, ys_l, covars_l, diag_only=True):
        ysg = ys_g.view(-1)
        ysl = ys_l.view(self.nc, -1)
        ys, covars = super().aggregate(ysg, covars_g, ysl, covars_l, diag_only=diag_only)

        return ys.reshape_as(ys_g), covars.reshape(*ysg.shape, *ysg.shape)

    def interpolate(self, xel, xq, diag_only=True):
        ys_g, covars_g = self.interpolate_global(xel, xq)
        ys_l, covars_l = self.interpolate_local(xel, xq)

        return self.aggregate(ys_g, covars_g, ys_l, covars_l, diag_only=diag_only)
