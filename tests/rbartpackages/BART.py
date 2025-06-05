from jaxtyping import Float64, Int64
from numpy import ndarray
from rpy2.rlike.container import OrdDict

from ._base import RObjectABC


class mc_gbart(RObjectABC):
    """
    Additional notes:
    - using the `x_test` argument may create problems, try not passing it
    - the object has an additional undocumented attribute `varprob`
    """  # noqa: D205, D400

    _rfuncname = 'BART::mc.gbart'
    _methods = ['predict']

    yhat_train: Float64[ndarray, 'ndpost n']
    yhat_test: Float64[ndarray, 'ndpost m']
    offset: float
    varcount: Int64[ndarray, 'ndpost p']
    varcount_mean: Float64[ndarray, ' p']
    treedraws: OrdDict

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        # fix type of ndpost attribute
        ndpost = self.ndpost.astype(int).item()
        self.ndpost = ndpost

        # modify sigma attribute to have the documented format
        if hasattr(self, 'sigma'):  # only if type='wbart' (default)
            if self.sigma.ndim == 1:
                self.sigma = self.sigma[:, None]
            nchains = self.sigma.shape[-1]
            assert ndpost % nchains == 0
            sigma = self.sigma[-ndpost // nchains :, :].flatten()
            first_sigma = self.sigma[: -ndpost // nchains, :].flatten()
            assert sigma.shape == (ndpost,)
            self.sigma = sigma
            self.first_sigma = first_sigma

    @property
    def type(self):
        return self._robject.rclass[0]


class bartModelMatrix(RObjectABC):
    _rfuncname = 'BART::bartModelMatrix'
