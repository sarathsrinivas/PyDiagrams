import torch as tc
from typing import Sequence
from torch import Tensor
from lib_gpr.gpr import Exact_GP


class GPR_SPLIT_COVAR(Exact_GP):
    """
     GPR with prediction done using splt covariances.
    """

    def get_split_covar(self, xe: Tensor, xq: Tensor) -> Sequence[Tensor]:

        sig = self.params[0]
        dim = xe.shape[-1]
        ls = self.params[-dim:]

        xel = xe.mul(ls[None, :])
        xql = xq.mul(ls[None, :])
        xsl = self.x.mul(ls[None, :])

        Aeq = (
            xel[:, None, :]
            .mul(xql[None, :, :])
            .mul_(2.0)
            .add_(xql.square()[None, :, :])
            .sum(-1)
        )

        Bes = xel[:, None, :].sub(xsl[None, :, :]).square_().sum(-1)

        Csq = xsl[:, None, :].mul(xql[None, :, :]).mul_(-2.0).sum(-1)

        Aeq.mul_(-1.0).exp_()
        Bes.mul_(-1.0).exp_()
        Csq.mul_(-1.0).exp_().mul_(sig ** 2)

        return Aeq, Bes, Csq

    def predict_split(
        self, xe: Tensor, xq: Tensor, var: str = "FULL"
    ) -> Sequence[Tensor]:

        Aeq, Bes, Csq = self.get_split_covar(xe, xq)

        self.update()

        ys = tc.mm(Bes, Csq.mul(self.wt[:, None])).mul_(Aeq)

        covars = NotImplemented

        return ys, covars
