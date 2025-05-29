# bartz/src/bartz/BART.py
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

"""Implement a user interface that mimics the R BART package."""

import functools
import math
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from jaxtyping import Array, Bool, Float, Float32

from . import grove, jaxext, mcmcloop, mcmcstep, prepcovars

FloatLike = float | Float[Any, '']


class gbart:
    """
    Nonparametric regression with Bayesian Additive Regression Trees (BART).

    Regress `y_train` on `x_train` with a latent mean function represented as
    a sum of decision trees. The inference is carried out by sampling the
    posterior distribution of the tree ensemble with an MCMC.

    Parameters
    ----------
    x_train : array (p, n) or DataFrame
        The training predictors.
    y_train : array (n,) or Series
        The training responses.
    x_test : array (p, m) or DataFrame, optional
        The test predictors.
    type
        The type of regression. 'wbart' for continuous regression, 'pbart' for
        binary regression with probit link.
    usequants : bool, default False
        Whether to use predictors quantiles instead of a uniform grid to bin
        predictors.
    sigest : float, optional
        An estimate of the residual standard deviation on `y_train`, used to set
        `lamda`. If not specified, it is estimated by linear regression (with
        intercept, and without taking into account `w`). If `y_train` has less
        than two elements, it is set to 1. If n <= p, it is set to the standard
        deviation of `y_train`. Ignored if `lamda` is specified.
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
    lamda
        The prior harmonic mean of the error variance. (The harmonic mean of x
        is 1/mean(1/x).) If not specified, it is set based on `sigest` and
        `sigquant`.
    tau_num
        The numerator in the expression that determines the prior standard
        deviation of leaves. If not specified, default to ``(max(y_train) -
        min(y_train)) / 2`` (or 1 if `y_train` has less than two elements) for
        continuous regression, and 3 for binary regression.
    offset
        The prior mean of the latent mean function. If not specified, it is set
        to the mean of `y_train` for continuous regression, and to
        ``Phi^-1(mean(y_train))`` for binary regression. If `y_train` is empty,
        `offset` is set to 0.
    w : array (n,), optional
        Coefficients that rescale the error standard deviation on each
        datapoint. Not specifying `w` is equivalent to setting it to 1 for all
        datapoints. Note: `w` is ignored in the automatic determination of
        `sigest`, so either the weights should be O(1), or `sigest` should be
        specified by the user.
    ntree : int, default 200
        The number of trees used to represent the latent mean function.
    numcut : int, default 255
        If `usequants` is `False`: the exact number of cutpoints used to bin the
        predictors, ranging between the minimum and maximum observed values
        (excluded).

        If `usequants` is `True`: the maximum number of cutpoints to use for
        binning the predictors. Each predictor is binned such that its
        distribution in `x_train` is approximately uniform across bins. The
        number of bins is at most the number of unique values appearing in
        `x_train`, or ``numcut + 1``.

        Before running the algorithm, the predictors are compressed to the
        smallest integer type that fits the bin indices, so `numcut` is best set
        to the maximum value of an unsigned integer type.
    ndpost : int, default 1000
        The number of MCMC samples to save, after burn-in.
    nskip : int, default 100
        The number of initial MCMC samples to discard as burn-in.
    keepevery : int, default 1
        The thinning factor for the MCMC samples, after burn-in.
    printevery : int or None, default 100
        The number of iterations (including thinned-away ones) between each log
        line. Set to `None` to disable logging.

        `printevery` has a few unexpected side effects. On cpu, interrupting
        with ^C halts the MCMC only on the next log. And the total number of
        iterations is a multiple of `printevery`, so if ``nskip + keepevery *
        ndpost`` is not a multiple of `printevery`, some of the last iterations
        will not be saved.
    seed : int or jax random key, default 0
        The seed for the random number generator.
    maxdepth : int, default 6
        The maximum depth of the trees. This is 1-based, so with the default
        ``maxdepth=6``, the depths of the levels range from 0 to 5.
    init_kw : dict
        Additional arguments passed to `mcmcstep.init`.
    run_mcmc_kw : dict
        Additional arguments passed to `mcmcloop.run_mcmc`.

    Attributes
    ----------
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
    offset : float
        The prior mean of the latent mean function.
    sigest : float or None
        The estimated standard deviation of the error used to set `lamda`.

    Notes
    -----
    This interface imitates the function ``gbart`` from the R package `BART
    <https://cran.r-project.org/package=BART>`_, but with these differences:

    - If `x_train` and `x_test` are matrices, they have one predictor per row
      instead of per column.
    - If `type` is not specified, it is determined solely based on the data type
      of `y_train`, and not on whether it contains only two unique values.
    - If ``usequants=False``, R BART switches to quantiles anyway if there are
      less predictor values than the required number of bins, while bartz
      always follows the specification.
    - The error variance parameter is called `lamda` instead of `lambda`.
    - `rm_const` is always `False`.
    - The default `numcut` is 255 instead of 100.
    - A lot of functionality is missing (e.g., variable selection).
    - There are some additional attributes, and some missing.
    - The trees have a maximum depth.

    """

    def __init__(
        self,
        x_train,
        y_train,
        *,
        x_test=None,
        type: Literal['wbart', 'pbart'] = 'wbart',
        usequants=False,
        sigest=None,
        sigdf=3,
        sigquant=0.9,
        k=2,
        power=2,
        base=0.95,
        lamda: FloatLike | None = None,
        tau_num: FloatLike | None = None,
        offset: FloatLike | None = None,
        w=None,
        ntree=200,
        numcut=255,
        ndpost=1000,
        nskip=100,
        keepevery=1,
        printevery=100,
        seed=0,
        maxdepth=6,
        init_kw=None,
        run_mcmc_kw=None,
    ):
        x_train, x_train_fmt = self._process_predictor_input(x_train)
        y_train, _ = self._process_response_input(y_train)
        self._check_same_length(x_train, y_train)
        if w is not None:
            w, _ = self._process_response_input(w)
            self._check_same_length(x_train, w)

        y_train = self._process_type_settings(y_train, type, w)
        # from here onwards, the type is determined by y_train.dtype == bool
        offset = self._process_offset_settings(y_train, offset)
        sigma_mu = self._process_leaf_sdev_settings(y_train, k, ntree, tau_num)
        lamda, sigest = self._process_error_variance_settings(
            x_train, y_train, sigest, sigdf, sigquant, lamda
        )

        splits, max_split = self._determine_splits(x_train, usequants, numcut)
        x_train = self._bin_predictors(x_train, splits)

        mcmc_state = self._setup_mcmc(
            x_train,
            y_train,
            offset,
            w,
            max_split,
            lamda,
            sigma_mu,
            sigdf,
            power,
            base,
            maxdepth,
            ntree,
            init_kw,
        )
        final_state, burnin_trace, main_trace = self._run_mcmc(
            mcmc_state, ndpost, nskip, keepevery, printevery, seed, run_mcmc_kw
        )

        sigma = self._extract_sigma(main_trace)
        first_sigma = self._extract_sigma(burnin_trace)

        self.offset = final_state.offset  # from the state because of buffer donation
        self.sigest = sigest
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

    @functools.cached_property
    def yhat_train(self):
        x_train = self._mcmc_state.X
        return self._predict(self._main_trace, x_train)

    @functools.cached_property
    def yhat_train_mean(self):
        return self.yhat_train.mean(axis=0)

    def predict(self, x_test):
        """
        Compute the posterior mean at `x_test` for each MCMC iteration.

        Parameters
        ----------
        x_test : array (p, m) or DataFrame
            The test predictors.

        Returns
        -------
        yhat_test : array (ndpost, m)
            The conditional posterior mean at `x_test` for each MCMC iteration.

        Raises
        ------
        ValueError
            If `x_test` has a different format than `x_train`.
        """
        x_test, x_test_fmt = self._process_predictor_input(x_test)
        if x_test_fmt != self._x_train_fmt:
            raise ValueError(
                f'Input format mismatch: {x_test_fmt=} != x_train_fmt={self._x_train_fmt!r}'
            )
        x_test = self._bin_predictors(x_test, self._splits)
        return self._predict(self._main_trace, x_test)

    @staticmethod
    def _process_predictor_input(x):
        if hasattr(x, 'columns'):
            fmt = dict(kind='dataframe', columns=x.columns)
            x = x.to_numpy().T
        else:
            fmt = dict(kind='array', num_covar=x.shape[0])
        x = jnp.asarray(x)
        assert x.ndim == 2
        return x, fmt

    @staticmethod
    def _process_response_input(y):
        if hasattr(y, 'to_numpy'):
            fmt = dict(kind='series', name=y.name)
            y = y.to_numpy()
        else:
            fmt = dict(kind='array')
        y = jnp.asarray(y)
        assert y.ndim == 1
        return y, fmt

    @staticmethod
    def _check_same_length(x1, x2):
        get_length = lambda x: x.shape[-1]
        assert get_length(x1) == get_length(x2)

    @staticmethod
    def _process_error_variance_settings(
        x_train, y_train, sigest, sigdf, sigquant, lamda
    ) -> tuple[Float32[Array, ''] | None, ...]:
        if y_train.dtype == bool:
            if sigest is not None:
                raise ValueError('Let `sigest=None` for binary regression')
            if lamda is not None:
                raise ValueError('Let `lamda=None` for binary regression')
            return None, None
        elif lamda is not None:
            if sigest is not None:
                raise ValueError('Let `sigest=None` if `lamda` is specified')
            return lamda, None
        else:
            if sigest is not None:
                sigest2 = jnp.square(sigest)
            elif y_train.size < 2:
                sigest2 = 1
            elif y_train.size <= x_train.shape[0]:
                sigest2 = jnp.var(y_train)
            else:
                x_centered = x_train.T - x_train.mean(axis=1)
                y_centered = y_train - y_train.mean()
                # centering is equivalent to adding an intercept column
                _, chisq, rank, _ = jnp.linalg.lstsq(x_centered, y_centered)
                chisq = chisq.squeeze(0)
                dof = len(y_train) - rank
                sigest2 = chisq / dof
            alpha = sigdf / 2
            invchi2 = jaxext.scipy.stats.invgamma.ppf(sigquant, alpha) / 2
            invchi2rid = invchi2 * sigdf
            return sigest2 / invchi2rid, jnp.sqrt(sigest2)

    @staticmethod
    def _process_type_settings(y_train, type, w):
        match type:
            case 'wbart':
                if y_train.dtype != jnp.float32:
                    raise TypeError(
                        'Continuous regression requires y_train.dtype=float32,'
                        f' got {y_train.dtype=} instead.'
                    )
            case 'pbart':
                if w is not None:
                    raise ValueError(
                        'Binary regression does not support weights, set `w=None`'
                    )
                if y_train.dtype != bool:
                    raise TypeError(
                        'Binary regression requires y_train.dtype=bool,'
                        f' got {y_train.dtype=} instead.'
                    )
            case _:
                raise ValueError(f'Invalid {type=}')

        return y_train

    @staticmethod
    def _process_offset_settings(
        y_train: Float32[Array, 'n'] | Bool[Array, 'n'],
        offset: float | Float32[Any, ''] | None,
    ) -> Float32[Array, '']:
        if offset is not None:
            return jnp.asarray(offset)
        elif y_train.size < 1:
            return jnp.array(0.0)
        else:
            mean = y_train.mean()

        if y_train.dtype == bool:
            return ndtri(mean)
        else:
            return mean

    @staticmethod
    def _process_leaf_sdev_settings(
        y_train: Float32[Array, 'n'] | Bool[Array, 'n'],
        k: float,
        ntree: int,
        tau_num: FloatLike | None,
    ):
        if tau_num is None:
            if y_train.dtype == bool:
                tau_num = 3.0
            elif y_train.size < 2:
                tau_num = 1.0
            else:
                tau_num = (y_train.max() - y_train.min()) / 2

        return tau_num / (k * math.sqrt(ntree))

    @staticmethod
    def _determine_splits(x_train, usequants, numcut):
        if usequants:
            return prepcovars.quantilized_splits_from_matrix(x_train, numcut + 1)
        else:
            return prepcovars.uniform_splits_from_matrix(x_train, numcut + 1)

    @staticmethod
    def _bin_predictors(x, splits):
        return prepcovars.bin_predictors(x, splits)

    @staticmethod
    def _setup_mcmc(
        x_train,
        y_train,
        offset,
        w,
        max_split,
        lamda,
        sigma_mu,
        sigdf,
        power,
        base,
        maxdepth,
        ntree,
        init_kw,
    ):
        depth = jnp.arange(maxdepth - 1)
        p_nonterminal = base / (1 + depth).astype(float) ** power

        if y_train.dtype == bool:
            sigma2_alpha = None
            sigma2_beta = None
        else:
            sigma2_alpha = sigdf / 2
            sigma2_beta = lamda * sigma2_alpha

        kw = dict(
            X=x_train,
            # copy y_train because it's going to be donated in the mcmc loop
            y=jnp.array(y_train),
            offset=offset,
            error_scale=w,
            max_split=max_split,
            num_trees=ntree,
            p_nonterminal=p_nonterminal,
            sigma_mu2=jnp.square(sigma_mu),
            sigma2_alpha=sigma2_alpha,
            sigma2_beta=sigma2_beta,
            min_points_per_leaf=5,
        )
        if init_kw is not None:
            kw.update(init_kw)
        return mcmcstep.init(**kw)

    @staticmethod
    def _run_mcmc(mcmc_state, ndpost, nskip, keepevery, printevery, seed, run_mcmc_kw):
        if isinstance(seed, jax.Array) and jnp.issubdtype(
            seed.dtype, jax.dtypes.prng_key
        ):
            key = seed.copy()
            # copy because the inner loop in run_mcmc will donate the buffer
        else:
            key = jax.random.key(seed)

        kw = dict(
            n_burn=nskip,
            n_skip=keepevery,
            inner_loop_length=printevery,
            allow_overflow=True,
        )
        if printevery is not None:
            kw.update(mcmcloop.make_print_callbacks())
        if run_mcmc_kw is not None:
            kw.update(run_mcmc_kw)

        return mcmcloop.run_mcmc(key, mcmc_state, ndpost, **kw)

    @staticmethod
    def _extract_sigma(trace) -> Float32[Array, 'trace_length'] | None:
        if trace['sigma2'] is None:
            return None
        else:
            return jnp.sqrt(trace['sigma2'])

    @staticmethod
    def _predict(trace, x):
        return mcmcloop.evaluate_trace(trace, x)

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
            norm2 = jnp.dot(
                resid, resid, preferred_element_type=bart['sigma2_beta'].dtype
            )
            beta = bart['sigma2_beta'] + norm2 / 2
        sigma2 = beta / alpha
        return jnp.sqrt(sigma2)

    def _compare_resid(self):
        bart = self._mcmc_state
        resid1 = bart.resid

        trees = grove.evaluate_forest(
            bart.X,
            bart.forest.leaf_trees,
            bart.forest.var_trees,
            bart.forest.split_trees,
            jnp.float32,  # TODO remove these configurable dtypes around
        )

        if bart.z is not None:
            ref = bart.z
        else:
            ref = bart.y
        resid2 = ref - (trees + bart.offset)

        return resid1, resid2

    def _avg_acc(self):
        trace = self._main_trace

        def acc(prefix):
            acc = trace[f'{prefix}_acc_count']
            prop = trace[f'{prefix}_prop_count']
            return acc.sum() / prop.sum()

        return acc('grow'), acc('prune')

    def _avg_prop(self):
        trace = self._main_trace

        def prop(prefix):
            return trace[f'{prefix}_prop_count'].sum()

        pgrow = prop('grow')
        pprune = prop('prune')
        total = pgrow + pprune
        return pgrow / total, pprune / total

    def _avg_move(self):
        agrow, aprune = self._avg_acc()
        pgrow, pprune = self._avg_prop()
        return agrow * pgrow, aprune * pprune

    def _depth_distr(self):
        from . import debug

        trace = self._main_trace
        split_trees = trace['split_trees']
        return debug.trace_depth_distr(split_trees)

    def _points_per_leaf_distr(self):
        from . import debug

        return debug.trace_points_per_leaf_distr(self._main_trace, self._mcmc_state.X)

    def _check_trees(self):
        from . import debug

        return debug.check_trace(self._main_trace, self._mcmc_state)

    def _tree_goes_bad(self):
        bad = self._check_trees().astype(bool)
        bad_before = jnp.pad(bad[:-1], [(1, 0), (0, 0)])
        return bad & ~bad_before
