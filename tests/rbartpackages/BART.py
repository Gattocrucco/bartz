# bartz/tests/rbartpackages/BART.py
#
# Copyright (c) 2024-2025, Giacomo Petrillo
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from jaxtyping import Float64, Int64
from numpy import ndarray
from rpy2.rlike.container import OrdDict

from ._base import RObjectBase


class mc_gbart(RObjectBase):
    """
    Additional notes:
    - using the `x_test` argument may create problems, try not passing it
    - the object has an additional undocumented attribute `varprob`
    """  # noqa: D205, D400

    _rfuncname = 'BART::mc.gbart'
    _methods = ('predict',)

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


class bartModelMatrix(RObjectBase):
    _rfuncname = 'BART::bartModelMatrix'
