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

import math
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Literal, Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, Real, Shaped, UInt
from numpy import ndarray

from bartz import mcmcloop, mcmcstep, prepcovars
from bartz.jaxext.scipy.special import ndtri
from bartz.jaxext.scipy.stats import invgamma

FloatLike = float | Float[Any, '']


class DataFrame(Protocol):
    """DataFrame duck-type for `gbart`.

    Attributes
    ----------
    columns : Sequence[str]
        The names of the columns.
    """

    columns: Sequence[str]

    def to_numpy(self) -> ndarray:
        """Convert the dataframe to a 2d numpy array with columns on the second axis."""
        ...


class Series(Protocol):
    """Series duck-type for `gbart`.

    Attributes
    ----------
    name : str | None
        The name of the series.
    """

    name: str | None

    def to_numpy(self) -> ndarray:
        """Convert the series to a 1d numpy array."""
        ...


class gbart:
    """
    Nonparametric regression with Bayesian Additive Regression Trees (BART).

    Regress `y_train` on `x_train` with a latent mean function represented as
    a sum of decision trees. The inference is carried out by sampling the
    posterior distribution of the tree ensemble with an MCMC.

    Parameters
    ----------
    x_train
        The training predictors.
    y_train
        The training responses.
    x_test
        The test predictors.
    type
        The type of regression. 'wbart' for continuous regression, 'pbart' for
        binary regression with probit link.
    xinfo
        A matrix with the cutpoins to use to bin each predictor. If not
        specified, it is generated automatically according to `usequants` and
        `numcut`.

        Each row shall contain a sorted list of cutpoints for a predictor. If
        there are less cutpoints than the number of columns in the matrix,
        fill the remaining cells with NaN.

        `xinfo` shall be a matrix even if `x_train` is a dataframe.
    usequants
        Whether to use predictors quantiles instead of a uniform grid to bin
        predictors. Ignored if `xinfo` is specified.
    rm_const
        How to treat predictors with no associated decision rules (i.e., there
        are no available cutpoints for that predictor). If `True` (default),
        they are ignored. If `False`, an error is raised if there are any. If
        `None`, no check is performed, and the output of the MCMC may not make
        sense if there are predictors without cutpoints. The option `None` is
        provided only to allow jax tracing.
    sigest
        An estimate of the residual standard deviation on `y_train`, used to set
        `lamda`. If not specified, it is estimated by linear regression (with
        intercept, and without taking into account `w`). If `y_train` has less
        than two elements, it is set to 1. If n <= p, it is set to the standard
        deviation of `y_train`. Ignored if `lamda` is specified.
    sigdf
        The degrees of freedom of the scaled inverse-chisquared prior on the
        noise variance.
    sigquant
        The quantile of the prior on the noise variance that shall match
        `sigest` to set the scale of the prior. Ignored if `lamda` is specified.
    k
        The inverse scale of the prior standard deviation on the latent mean
        function, relative to half the observed range of `y_train`. If `y_train`
        has less than two elements, `k` is ignored and the scale is set to 1.
    power
    base
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
        `offset` is set to 0. With binary regression, if `y_train` is all
        `False` or `True`, it is set to ``Phi^-1(1/(n+1))`` or
        ``Phi^-1(n/(n+1))``, respectively.
    w
        Coefficients that rescale the error standard deviation on each
        datapoint. Not specifying `w` is equivalent to setting it to 1 for all
        datapoints. Note: `w` is ignored in the automatic determination of
        `sigest`, so either the weights should be O(1), or `sigest` should be
        specified by the user.
    ntree
        The number of trees used to represent the latent mean function.
    numcut
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
        to the maximum value of an unsigned integer type, like 255.

        Ignored if `xinfo` is specified.
    ndpost
        The number of MCMC samples to save, after burn-in.
    nskip
        The number of initial MCMC samples to discard as burn-in.
    keepevery
        The thinning factor for the MCMC samples, after burn-in.
    printevery
        The number of iterations (including thinned-away ones) between each log
        line. Set to `None` to disable logging.

        `printevery` has a few unexpected side effects. On cpu, interrupting
        with ^C halts the MCMC only on the next log. And the total number of
        iterations is a multiple of `printevery`, so if ``nskip + keepevery *
        ndpost`` is not a multiple of `printevery`, some of the last iterations
        will not be saved.
    seed
        The seed for the random number generator.
    maxdepth
        The maximum depth of the trees. This is 1-based, so with the default
        ``maxdepth=6``, the depths of the levels range from 0 to 5.
    init_kw
        Additional arguments passed to `mcmcstep.init`.
    run_mcmc_kw
        Additional arguments passed to `mcmcloop.run_mcmc`.

    Attributes
    ----------
    yhat_test : Float32[Array, 'ndpost m'] | None
        The conditional posterior mean at `x_test` for each MCMC iteration.
    yhat_test_mean : Float32[Array, 'm'] | None
        The marginal posterior mean at `x_test`.
    sigma : Float32[Array, 'ndpost'] | None
        The standard deviation of the error.
    first_sigma : Float32[Array, 'nskip'] | None
        The standard deviation of the error in the burn-in phase.
    offset : Float32[Array, '']
        The prior mean of the latent mean function.
    sigest : Float32[Array, ''] | None
        The estimated standard deviation of the error used to set `lamda`.

    Notes
    -----
    This interface imitates the function ``gbart`` from the R package `BART
    <https://cran.r-project.org/package=BART>`_, but with these differences:

    - If `x_train` and `x_test` are matrices, they have one predictor per row
      instead of per column.
    - If ``usequants=False``, R BART switches to quantiles anyway if there are
      less predictor values than the required number of bins, while bartz
      always follows the specification.
    - The error variance parameter is called `lamda` instead of `lambda`.
    - The default `numcut` is 255 instead of 100.
    - Some functionality is missing (e.g., variable selection).
    - There are some additional attributes, and some missing.
    - The trees have a maximum depth.
    - `rm_const` refers to predictors without decision rules instead of
      predictors that are constant in `x_train`.

    """

    yhat_test: Float32[Array, 'ndpost m'] | None = None
    yhat_test_mean: Float32[Array, ' m'] | None = None
    sigma: Float32[Array, ' ndpost'] | None
    first_sigma: Float32[Array, ' nskip'] | None
    offset: Float32[Array, '']
    sigest: Float32[Array, ''] | None

    _mcmc_state: mcmcstep.State
    _main_trace: mcmcloop.MainTrace
    _splits: Real[Array, 'p max_num_splits']

    def __init__(
        self,
        x_train: Real[Array, 'p n'] | DataFrame,
        y_train: Bool[Array, ' n'] | Float32[Array, ' n'] | Series,
        *,
        x_test: Real[Array, 'p m'] | DataFrame | None = None,
        type: Literal['wbart', 'pbart'] = 'wbart',  # noqa: A002
        xinfo: Float[Array, 'p n'] | None = None,
        usequants: bool = False,
        rm_const: bool | None = True,
        sigest: FloatLike | None = None,
        sigdf: FloatLike = 3.0,
        sigquant: FloatLike = 0.9,
        k: FloatLike = 2.0,
        power: FloatLike = 2.0,
        base: FloatLike = 0.95,
        lamda: FloatLike | None = None,
        tau_num: FloatLike | None = None,
        offset: FloatLike | None = None,
        w: Float[Array, ' n'] | None = None,
        ntree: int = 200,
        numcut: int = 255,
        ndpost: int = 1000,
        nskip: int = 100,
        keepevery: int = 1,
        printevery: int | None = 100,
        seed: int | Key[Array, ''] = 0,
        maxdepth: int = 6,
        init_kw: dict | None = None,
        run_mcmc_kw: dict | None = None,
    ):
        x_train, x_train_fmt = self._process_predictor_input(x_train)
        y_train, _ = self._process_response_input(y_train)
        self._check_same_length(x_train, y_train)
        if w is not None:
            w, _ = self._process_response_input(w)
            self._check_same_length(x_train, w)

        self._check_type_settings(y_train, type, w)
        # from here onwards, the type is determined by y_train.dtype == bool
        offset = self._process_offset_settings(y_train, offset)
        sigma_mu = self._process_leaf_sdev_settings(y_train, k, ntree, tau_num)
        lamda, sigest = self._process_error_variance_settings(
            x_train, y_train, sigest, sigdf, sigquant, lamda
        )

        splits, max_split = self._determine_splits(x_train, usequants, numcut, xinfo)
        x_train = self._bin_predictors(x_train, splits)

        initial_state = self._setup_mcmc(
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
            rm_const,
        )
        final_state, burnin_trace, main_trace = self._run_mcmc(
            initial_state, ndpost, nskip, keepevery, printevery, seed, run_mcmc_kw
        )

        self.offset = final_state.offset  # from the state because of buffer donation
        self.sigest = sigest
        self.sigma = self._extract_sigma(main_trace)
        self.first_sigma = self._extract_sigma(burnin_trace)

        self._x_train_fmt = x_train_fmt
        self._splits = splits
        self._main_trace = main_trace
        self._mcmc_state = final_state

        if x_test is not None:
            yhat_test = self.predict(x_test)
            self.yhat_test = yhat_test
            self.yhat_test_mean = yhat_test.mean(axis=0)

    @cached_property
    def yhat_train(self) -> Float32[Array, 'ndpost n']:
        """The conditional posterior mean at `x_train` for each MCMC iteration."""
        x_train = self._mcmc_state.X
        return self._predict(x_train)

    @cached_property
    def yhat_train_mean(self) -> Float32[Array, ' n']:
        """The marginal posterior mean at `x_train`."""
        return self.yhat_train.mean(axis=0)

    @cached_property
    def varcount(self) -> Int32[Array, 'ndpost p']:
        """Histogram of predictor usage for decision rules in the trees."""
        return mcmcloop.compute_varcount(
            self._mcmc_state.forest.max_split.size, self._main_trace
        )

    @cached_property
    def varcount_mean(self) -> Float32[Array, ' p']:
        """Average of `varcount` across MCMC iterations."""
        return self.varcount.mean(axis=0)

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
            msg = f'Input format mismatch: {x_test_fmt=} != x_train_fmt={self._x_train_fmt!r}'
            raise ValueError(msg)
        x_test = self._bin_predictors(x_test, self._splits)
        return self._predict(x_test)

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
    def _process_response_input(y) -> tuple[Shaped[Array, ' n'], Any]:
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
                msg = 'Let `sigest=None` for binary regression'
                raise ValueError(msg)
            if lamda is not None:
                msg = 'Let `lamda=None` for binary regression'
                raise ValueError(msg)
            return None, None
        elif lamda is not None:
            if sigest is not None:
                msg = 'Let `sigest=None` if `lamda` is specified'
                raise ValueError(msg)
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
            invchi2 = invgamma.ppf(sigquant, alpha) / 2
            invchi2rid = invchi2 * sigdf
            return sigest2 / invchi2rid, jnp.sqrt(sigest2)

    @staticmethod
    def _check_type_settings(y_train, type, w):  # noqa: A002
        match type:
            case 'wbart':
                if y_train.dtype != jnp.float32:
                    msg = (
                        'Continuous regression requires y_train.dtype=float32,'
                        f' got {y_train.dtype=} instead.'
                    )
                    raise TypeError(msg)
            case 'pbart':
                if w is not None:
                    msg = 'Binary regression does not support weights, set `w=None`'
                    raise ValueError(msg)
                if y_train.dtype != bool:
                    msg = (
                        'Binary regression requires y_train.dtype=bool,'
                        f' got {y_train.dtype=} instead.'
                    )
                    raise TypeError(msg)
            case _:
                msg = f'Invalid {type=}'
                raise ValueError(msg)

    @staticmethod
    def _process_offset_settings(
        y_train: Float32[Array, ' n'] | Bool[Array, ' n'],
        offset: float | Float32[Any, ''] | None,
    ) -> Float32[Array, '']:
        if offset is not None:
            return jnp.asarray(offset)
        elif y_train.size < 1:
            return jnp.array(0.0)
        else:
            mean = y_train.mean()

        if y_train.dtype == bool:
            bound = 1 / (1 + y_train.size)
            mean = jnp.clip(mean, bound, 1 - bound)
            return ndtri(mean)
        else:
            return mean

    @staticmethod
    def _process_leaf_sdev_settings(
        y_train: Float32[Array, ' n'] | Bool[Array, ' n'],
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
    def _determine_splits(
        x_train: Real[Array, 'p n'],
        usequants: bool,
        numcut: int,
        xinfo: Float[Array, 'p n'] | None,
    ) -> tuple[Real[Array, 'p m'], UInt[Array, ' p']]:
        if xinfo is not None:
            if xinfo.ndim != 2 or xinfo.shape[0] != x_train.shape[0]:
                msg = f'{xinfo.shape=} different from expected ({x_train.shape[0]}, *)'
                raise ValueError(msg)
            return prepcovars.parse_xinfo(xinfo)
        elif usequants:
            return prepcovars.quantilized_splits_from_matrix(x_train, numcut + 1)
        else:
            return prepcovars.uniform_splits_from_matrix(x_train, numcut + 1)

    @staticmethod
    def _bin_predictors(x, splits) -> UInt[Array, 'p n']:
        return prepcovars.bin_predictors(x, splits)

    @staticmethod
    def _setup_mcmc(
        x_train: Real[Array, 'p n'],
        y_train: Float32[Array, ' n'] | Bool[Array, ' n'],
        offset: Float32[Array, ''],
        w: Float[Array, ' n'] | None,
        max_split: UInt[Array, ' p'],
        lamda: Float32[Array, ''] | None,
        sigma_mu: FloatLike,
        sigdf: FloatLike,
        power: FloatLike,
        base: FloatLike,
        maxdepth: int,
        ntree: int,
        init_kw: dict[str, Any] | None,
        rm_const: bool | None,
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
            min_points_per_decision_node=10,
            min_points_per_leaf=5,
        )

        if rm_const is None:
            kw.update(filter_splitless_vars=False)
        elif rm_const:
            kw.update(filter_splitless_vars=True)
        else:
            n_empty = jnp.count_nonzero(max_split == 0)
            if n_empty:
                msg = f'There are {n_empty} predictors without decision rules'
                raise ValueError(msg)
            kw.update(filter_splitless_vars=False)

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
    def _extract_sigma(
        trace: mcmcloop.MainTrace,
    ) -> Float32[Array, ' trace_length'] | None:
        if trace.sigma2 is None:
            return None
        else:
            return jnp.sqrt(trace.sigma2)

    def _predict(self, x):
        return mcmcloop.evaluate_trace(self._main_trace, x)
