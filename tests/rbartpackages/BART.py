from . import _base

class mc_gbart(_base.RObjectABC):

    _rfuncname = 'BART::mc.gbart'
    _methods = ['predict']
    
    def __init__(self, *args, **kw):
        """
        Notes:
        - using the `x_test` argument may create problems, try not passing it
        - the object has an additional undocumented attribute `varprob`
        """
        super().__init__(*args, **kw)

        # fix type of ndpost attribute
        ndpost = self.ndpost.astype(int).item()
        self.ndpost = ndpost

        # modify sigma attribute to have the documented format
        if hasattr(self, 'sigma'): # only if type='wbart' (default)
            if self.sigma.ndim == 1:
                self.sigma = self.sigma[:, None]
            nchains = self.sigma.shape[-1]
            assert ndpost % nchains == 0
            sigma = self.sigma[-ndpost // nchains:, :].flatten()
            first_sigma = self.sigma[:-ndpost // nchains, :].flatten()
            assert sigma.shape == (ndpost,)
            self.sigma = sigma
            self.first_sigma = first_sigma

    @property
    def type(self):
        return self._robject.rclass[0]
