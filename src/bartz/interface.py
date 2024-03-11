# bartz/src/bartz/interface.py
#
# Copyright (c) 2024, Giacomo Petrillo
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

import jax.numpy as jnp

from . import jaxext
from . import prepcovars
from . import mcmcstep
from . import mcmcloop

class BART:
    """
    Nonparametric regression with Bayesian Additive Regression Trees (BART).

    Regress `y_train` on `x_train` with a latent mean function represented as
    a sum of decision trees. The inference is carried out by estimating the
    posterior distribution of the tree ensemble with an MCMC.

    Parameters
    ----------
    x_train : array (n, p) or DataFrame
        The training predictors.
    y_train : array (n,), Series, DataFrame with one column
        The training responses.
    x_test : array (m, p) or DataFrame, optional
        The test predictors.
    sigest : float, optional
        An estimate of the residual standard deviation on `y_train`, used to
        set `lamda`. If not specified, it is estimated by linear regression.
        Ignored if `lamda` is specified.
    sigdf : int, default 3
        The degrees of freedom of the scaled inverse-chisquared prior on the
        noise variance.
    sigquant : float, default 0.9
        The quantile of the prior on the noise variance that shall match
        `sigest` to set the scale of the prior. Ignored if `lamda` is specified.
    k : float, default 2
        The inverse scale of the prior standard deviation on the latent mean
        function, relative to half the observed range of `y_train`.
    power : float, default 2
    base : float, default 0.95
        Parameters of the prior on tree node generation. The probability that a
        node at depth `d` (0-based) is non-terminal is ``base / (1 + d) **
        power``.
    maxdepth : int, default 6
        The maximum depth of the trees. This is 1-based, so with the default
        ``maxdepth=6``, the depths of the levels range from 0 to 5.
    lamda : float, optional
        The scale of the prior on the noise variance. If ``lambda==1``, the
        prior is an inverse chi-squared scaled to have harmonic mean 1. If
        not specified, it is set based on `sigest` and `sigquant`.
    offset : float, optional
        The prior mean of the latent mean function. If not specified, it is set
        to the mean of `y_train`.
    ntree : int, default 200
        The number of trees used to represent the latent mean function.
    numcut : int, default 255
        The maximum number of cutpoints to use for binning the predictors. Each
        covariate is binned such that its distribution in `x_train` is
        approximately uniform across bins. The number of bins is at most the
        number of unique values appearing in `x_train`, or ``numcut + 1``.
        Before running the algorithm, the predictors are compressed to th
        smallest integer type that fits the bin indices, so `numcut` is best set
        to the maximum value of an unsigned integer type.
    ndpost : int, default 1000
        The number of MCMC samples to save, after burn-in.
    nskip : int, default 100
        The number of initial MCMC samples to discard as burn-in.
    keepevery : int, default 1
        The thinning factor for the MCMC samples, after burn-in.
    printevery : int, default 100
        The number of iterations (including skipped ones) between each log.
    seed : int, default 0
        The seed for the random number generator.

    Attributes
    ----------
    offset : float
        The prior mean of the latent mean function.
    scale : float
        The prior standard deviation of the latent mean function.
    yhat_train : array (ndpost, n)
        The conditional posterior mean at `x_train` for each MCMC iteration.
    yhat_train_mean : array (n,)
        The marginal posterior mean at `x_train`.
    yhat_test : array (ndpost, m)
        The conditional posterior mean at `x_test` for each MCMC iteration.
    yhat_test_mean : array (m,)
        The marginal posterior mean at `x_test`.
    sigma : array (ndpost,)
        The standard deviation of the error.
    first_sigma : array (nskip,)
        The standard deviation of the error in the burn-in phase.

    Methods
    -------
    predict
    """

    def __init__(self, x_train, y_train, *, x_test=None, sigest=None, sigdf=3, sigquant=0.9, k=2, power=2, base=0.95, maxdepth=6, lamda=None, offset=None, ntree=200, numcut=255, ndpost=1000, nskip=100, keepevery=1, printevery=100, seed=0):

        x_train, x_train_fmt = self._process_covariate_input(x_train)
        if x_test is not None:
            x_test, x_test_fmt = self._process_covariate_input(x_test)
            self._check_compatible_formats(x_train_fmt, x_test_fmt)
        
        y_train, y_train_fmt = self._process_response_input(y_train)
        self._check_same_length(x_train, y_train)
        
        lamda = self._process_noise_variance_settings(x_train, y_train, sigest, sigdf, sigquant, lamda)
        offset = self._process_offset_settings(y_train, offset)
        scale = self._process_scale_settings(y_train, k)

        splits, max_split = self._determine_splits(x_train, numcut)
        x_train = self._bin_covariates(x_train, splits)
        if x_test is not None:
            x_test = self._bin_covariates(x_test, splits)

        y_train = self._transform_input(y_train, offset, scale)
        lamda = lamda / scale

        mcmc_state = self._setup_mcmc(x_train, y_train, max_split, lamda, sigdf, power, base, maxdepth, ntree)
        burnin_trace, main_trace = self._run_mcmc(mcmc_state, ndpost, nskip, keepevery, printevery, seed)

        yhat_train = self._predict(main_trace, x_train)
        if x_test is not None:
            yhat_test = self._predict(main_trace, x_test)

        yhat_train = self._transform_output(yhat_train, offset, scale)
        yhat_train_mean = yhat_train.mean(axis=-1)
        if x_test is not None:
            yhat_test = self._transform_output(yhat_test, offset, scale)
            yhat_test_mean = yhat_test.mean(axis=-1)

        sigma = self._extract_sigma(main_trace)
        first_sigma = self._extract_sigma(burnin_trace)

        self.offset = offset
        self.scale = scale
        self.yhat_train = yhat_train
        self.yhat_train_mean = yhat_train_mean
        self.yhat_test = yhat_test
        self.yhat_test_mean = yhat_test_mean
        self.sigma = sigma
        self.first_sigma = first_sigma

        self._x_train_fmt = x_train_fmt
        self._splits = splits
        self._main_trace = main_trace

    def predict(self, x_test):
        """
        Compute the posterior mean at `x_test` for each MCMC iteration.

        Parameters
        ----------
        x_test : array (m, p) or DataFrame
            The test predictors.

        Returns
        -------
        yhat_test : array (ndpost, m)
            The conditional posterior mean at `x_test` for each MCMC iteration.
        """
        x_test, x_test_fmt = self._process_covariate_input(x_test)
        self._check_compatible_formats(x_test_fmt, self._x_train_fmt)
        x_test = self._bin_covariates(x_test, self._splits)
        yhat_test = self._predict(self._main_trace, x_test)
        yhat_test = self._transform_output(yhat_test, self.offset, self.scale)
        return yhat_test

    def _process_covariate_input(self, x):
        if hasattr(x, 'columns'):
            fmt = dict(kind='dataframe', columns=x.columns)
            x = x.to_numpy()
        else:
            fmt = dict(kind='array', num_covar=x.shape[1])
        x = jnp.asarray(x.T)
        assert x.ndim == 2
        return x, fmt

    def _check_compatible_formats(self, fmt1, fmt2):
        assert fmt1 == fmt2

    def _process_response_input(self, y):
        if hasattr(y, 'columns'):
            fmt = dict(kind='dataframe', columns=y.columns)
            assert len(y.columns) == 1
            y = y.to_numpy().squeeze(0)
        elif hasattr(y, 'to_numpy'):
            fmt = dict(kind='series', name=y.name)
            y = y.to_numpy()
        else:
            fmt = dict(kind='array')
        y = jnp.asarray(y)
        assert y.ndim == 1
        return y, fmt

    def _check_same_length(self, x1, x2):
        get_length = lambda x: x.shape[-1]
        assert get_length(x1) == get_length(x2)

    def _process_noise_variance_settings(self, x_train, y_train, sigest, sigdf, sigquant, lamda):
        if lamda is None:
            if sigest is None:
                _, chisq, rank, _ = jnp.linalg.lstsq(x_train.T, y_train)
                chisq = chisq.squeeze(0)
                dof = len(y_train) - rank
                sigest2 = chisq / dof
            else:
                sigest2 = sigest ** 2
            alpha = sidgf / 2
            invchi2 = jaxext.scipy.stats.invgamma.ppf(sigquant, alpha) / 2
            invchi2rid = invchi2 * sigdf
            lamda = sigest2 / invchi2rid
        return lamda

    def _process_offset_settings(self, y_train, offset):
        if offset is None:
            offset = y_train.mean()
        return offset

    def _process_scale_settings(self, y_train, k):
        return (y_train.max() - y_train.min()) / (2 * k)

    def _determine_splits(self, x_train, numcut):
        return prepcovars.quantilized_splits_from_matrix(x_train, numcut + 1)

    def _bin_covariates(self, x, splits):
        return prepcovars.bin_covariates(x, splits)

    def _transform_input(self, y, offset, scale):
        return (y - offset) / scale

    def _setup_mcmc(self, x_train, y_train, max_split, lamda, sigdf, power, base, maxdepth, ntree):
        depth = jnp.arange(maxdepth - 1)
        p_nonterminal = base / (1 + depth).astype(float) ** power
        sigma2_alpha = sigdf / 2
        sigma2_beta = lamda * sigma2_alpha
        return mcmcstep.make_bart(x_train, y_train, max_split, ntree, p_nonterminal, sigma2_alpha, sigma2_beta, jnp.float32, jnp.float32)

    def _run_mcmc(self, mcmc_state, ndpost, nskip, keepevery, printevery, seed):
        key = jax.random.PRNGKey(seed)
        callback = mcmcloop.make_simple_print_callback(printevery)
        _, burnin_trace, main_trace = mcmcloop.run_mcmc(mcmc_state, ndpost, keepevery, nskip, callback, key)
        return burnin_trace, main_trace

    def _predict(self, trace, x):
        return mcmcloop.evaluate_trace(trace, x)

    def _transform_output(self, y, offset, scale):
        return y * scale + offset

    def _extract_sigma(self, trace):
        return jnp.sqrt(trace['sigma2'])