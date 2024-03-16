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

import jax
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
    x_train : array (p, n) or DataFrame
        The training predictors.
    y_train : array (n,) or Series
        The training responses.
    x_test : array (p, m) or DataFrame, optional
        The test predictors.
    sigest : float, optional
        An estimate of the residual standard deviation on `y_train`, used to
        set `lamda`. If not specified, it is estimated by linear regression.
        If `y_train` has less than two elements, it is set to 1. If n <= p, it
        is set to the variance of `y_train`. Ignored if `lamda` is specified.
    sigdf : int, default 3
        The degrees of freedom of the scaled inverse-chisquared prior on the
        noise variance.
    sigquant : float, default 0.9
        The quantile of the prior on the noise variance that shall match
        `sigest` to set the scale of the prior. Ignored if `lamda` is specified.
    k : float, default 2
        The inverse scale of the prior standard deviation on the latent mean
        function, relative to half the observed range of `y_train`. If `y_train`
        has less than two elements, `k` is ignored and the scale is set to 1.
    power : float, default 2
    base : float, default 0.95
        Parameters of the prior on tree node generation. The probability that a
        node at depth `d` (0-based) is non-terminal is ``base / (1 + d) **
        power``.
    maxdepth : int, default 6
        The maximum depth of the trees. This is 1-based, so with the default
        ``maxdepth=6``, the depths of the levels range from 0 to 5.
    lamda : float, optional
        The scale of the prior on the noise variance. If ``lamda==1``, the
        prior is an inverse chi-squared scaled to have harmonic mean 1. If
        not specified, it is set based on `sigest` and `sigquant`.
    offset : float, optional
        The prior mean of the latent mean function. If not specified, it is set
        to the mean of `y_train`. If `y_train` is empty, it is set to 0.
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
    seed : int or jax random key, default 0
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
        
        y_train, y_train_fmt = self._process_response_input(y_train)
        self._check_same_length(x_train, y_train)
        
        lamda = self._process_noise_variance_settings(x_train, y_train, sigest, sigdf, sigquant, lamda)
        offset = self._process_offset_settings(y_train, offset)
        scale = self._process_scale_settings(y_train, k)

        splits, max_split = self._determine_splits(x_train, numcut)
        x_train = self._bin_covariates(x_train, splits)

        y_train = self._transform_input(y_train, offset, scale)
        lamda = lamda / scale

        mcmc_state = self._setup_mcmc(x_train, y_train, max_split, lamda, sigdf, power, base, maxdepth, ntree)
        final_state, burnin_trace, main_trace = self._run_mcmc(mcmc_state, ndpost, nskip, keepevery, printevery, seed)

        yhat_train = self._predict(main_trace, x_train)
        yhat_train = self._transform_output(yhat_train, offset, scale)
        yhat_train_mean = yhat_train.mean(axis=0)

        sigma = self._extract_sigma(main_trace, scale)
        first_sigma = self._extract_sigma(burnin_trace, scale)

        self.offset = offset
        self.scale = scale
        self.yhat_train = yhat_train
        self.yhat_train_mean = yhat_train_mean
        self.sigma = sigma
        self.first_sigma = first_sigma

        self._x_train_fmt = x_train_fmt
        self._splits = splits
        self._main_trace = main_trace
        self._mcmc_state = final_state

        if x_test is not None:
            yhat_test = self.predict(x_test)
            self.yhat_test = yhat_test
            self.yhat_test_mean = yhat_test.mean(axis=0)

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
            x = x.to_numpy().T
        else:
            fmt = dict(kind='array', num_covar=x.shape[0])
        x = jnp.asarray(x)
        assert x.ndim == 2
        return x, fmt

    def _check_compatible_formats(self, fmt1, fmt2):
        assert fmt1 == fmt2

    def _process_response_input(self, y):
        if hasattr(y, 'to_numpy'):
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
        if lamda is not None:
            return lamda
        else:
            if sigest is not None:
                sigest2 = sigest * sigest
            elif y_train.size < 2:
                sigest2 = 1
            elif y_train.size <= x_train.shape[0]:
                sigest2 = jnp.var(y_train)
            else:
                _, chisq, rank, _ = jnp.linalg.lstsq(x_train.T, y_train)
                chisq = chisq.squeeze(0)
                dof = len(y_train) - rank
                sigest2 = chisq / dof
            alpha = sigdf / 2
            invchi2 = jaxext.scipy.stats.invgamma.ppf(sigquant, alpha) / 2
            invchi2rid = invchi2 * sigdf
            return sigest2 / invchi2rid

    def _process_offset_settings(self, y_train, offset):
        if offset is not None:
            return offset
        elif y_train.size < 1:
            return 0
        else:
            return y_train.mean()

    def _process_scale_settings(self, y_train, k):
        if y_train.size < 2:
            return 1
        else:
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
        return mcmcstep.make_bart(
            X=x_train,
            y=y_train,
            max_split=max_split,
            num_trees=ntree,
            p_nonterminal=p_nonterminal,
            sigma2_alpha=sigma2_alpha,
            sigma2_beta=sigma2_beta,
            small_float_dtype=jnp.float32,
            large_float_dtype=jnp.float32,
        )

    def _run_mcmc(self, mcmc_state, ndpost, nskip, keepevery, printevery, seed):
        if isinstance(seed, jax.Array) and jnp.issubdtype(seed.dtype, jax.dtypes.prng_key):
            key = seed
        else:
            key = jax.random.key(seed)
        callback = mcmcloop.make_simple_print_callback(printevery)
        return mcmcloop.run_mcmc(mcmc_state, nskip, ndpost, keepevery, callback, key)

    def _predict(self, trace, x):
        return mcmcloop.evaluate_trace(trace, x)

    def _transform_output(self, y, offset, scale):
        return offset + scale * y

    def _extract_sigma(self, trace, scale):
        return scale * jnp.sqrt(trace['sigma2'])

    
    def _predict_debug(self, x_test):
        from . import debug
        x_test, x_test_fmt = self._process_covariate_input(x_test)
        self._check_compatible_formats(x_test_fmt, self._x_train_fmt)
        x_test = self._bin_covariates(x_test, self._splits)
        return debug.trace_evaluate_trees(self._main_trace, x_test)

    def _show_tree(self, i_sample, i_tree, print_all=False):
        from . import debug
        trace = self._main_trace
        leaf_tree = trace['leaf_trees'][i_sample, i_tree]
        var_tree = trace['var_trees'][i_sample, i_tree]
        split_tree = trace['split_trees'][i_sample, i_tree]
        debug.print_tree(leaf_tree, var_tree, split_tree, print_all)

    def _sigma_harmonic_mean(self, prior=False):
        bart = self._mcmc_state
        if prior:
            alpha = bart['sigma2_alpha']
            beta = bart['sigma2_beta']
        else:
            resid = bart['resid']
            alpha = bart['sigma2_alpha'] + resid.size / 2
            norm2 = jnp.dot(resid, resid, preferred_element_type=bart['sigma2_beta'].dtype)
            beta = bart['sigma2_beta'] + norm2 / 2
        sigma2 = beta / alpha
        return jnp.sqrt(sigma2) * self.scale

    def _compare_resid(self):
        bart = self._mcmc_state
        resid1 = bart['resid']
        yhat = mcmcstep.evaluate_tree_vmap_x(bart['X'], bart['leaf_trees'], bart['var_trees'], bart['split_trees'], jnp.float32)
        resid2 = bart['y'] - yhat
        return resid1, resid2
