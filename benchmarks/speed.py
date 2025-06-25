# bartz/benchmarks/speed.py
#
# Copyright (c) 2025, Giacomo Petrillo
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

"""Measure the speed of the MCMC and its interfaces."""

from contextlib import redirect_stdout
from functools import partial
from inspect import signature
from io import StringIO
from itertools import product
from typing import Literal

from asv_runner.benchmarks.mark import skip_for_params
from jax import block_until_ready, clear_caches, eval_shape, jit, random, vmap
from jax import numpy as jnp
from jax.tree_util import tree_map

from bartz import mcmcstep
from bartz.mcmcloop import run_mcmc

try:
    from bartz.BART import gbart
except ImportError:
    from bartz import BART as gbart

try:
    from bartz.mcmcstep import step
except ImportError:
    from bartz.mcmcstep import mcmc_step as step

try:
    from bartz.mcmcstep import init
except ImportError:
    from bartz.mcmcstep import make_bart as init


# asv config
timeout = 30.0

# config
P = 100
N = 10000
NTREE = 50
NITERS = 10


def gen_data(p: int, n: int):
    """Generate pretty nonsensical data."""
    X = jnp.arange(p * n, dtype=jnp.uint8).reshape(p, n)
    X = vmap(jnp.roll)(X, jnp.arange(p))
    max_split = jnp.full(p, 255, jnp.uint8)
    y = jnp.cos(jnp.linspace(0, 2 * jnp.pi / 32 * n, n))
    return X, y, max_split


def make_p_nonterminal(maxdepth: int):
    """Prepare the p_nonterminal argument to `mcmcstep.init`."""
    depth = jnp.arange(maxdepth - 1)
    base = 0.95
    power = 2
    return base / (1 + depth).astype(float) ** power


Kind = Literal['plain', 'weights', 'binary']


@partial(jit, static_argnums=(0, 1, 2, 3))
def simple_init(p: int, n: int, ntree: int, kind: Kind = 'plain', /, **kwargs):
    """Simplified version of `bartz.mcmcstep.init` with data pre-filled."""
    X, y, max_split = gen_data(p, n)

    kw: dict = dict(
        X=X,
        y=y,
        max_split=max_split,
        num_trees=ntree,
        p_nonterminal=make_p_nonterminal(6),
        sigma_mu2=1 / ntree,
        sigma2_alpha=1.0,
        sigma2_beta=1.0,
        min_points_per_decision_node=10,
        filter_splitless_vars=False,
    )

    # adapt arguments for old versions
    sig = signature(init)
    if 'sigma_mu2' not in sig.parameters:
        kw.pop('sigma_mu2')
    if 'min_points_per_decision_node' not in sig.parameters:
        kw.pop('min_points_per_decision_node')
        kw.update(min_points_per_leaf=5)
    if 'filter_splitless_vars' not in sig.parameters:
        kw.pop('filter_splitless_vars')
    if 'suffstat_batch_size' in sig.parameters:
        # bypass the tracing bug fixed in v0.2.1
        kw.update(suffstat_batch_size=None)

    match kind:
        case 'weights':
            if 'error_scale' not in sig.parameters:
                msg = 'weights not supported'
                raise NotImplementedError(msg)
            kw['error_scale'] = jnp.ones(n)

        case 'binary':
            if not hasattr(mcmcstep, 'step_z'):
                msg = 'binary not supported'
                raise NotImplementedError(msg)
            kw['y'] = y > 0
            kw.pop('sigma2_alpha')
            kw.pop('sigma2_beta')

    kw.update(kwargs)

    return init(**kw)


Mode = Literal['compile', 'run']
Cache = Literal['cold', 'warm']


class TimeRunMcmc:
    """Timings of `run_mcmc`."""

    # asv config
    params: tuple[tuple[Mode, ...], tuple[int, ...], tuple[Cache, ...]] = (
        ('compile', 'run'),
        (0, NITERS),
        ('cold', 'warm'),
    )
    param_names = ('mode', 'niters', 'cache')
    warmup_time = 0.0
    number = 1

    def setup(self, mode: Mode, niters: int, cache: Cache):
        """Prepare the arguments, compile the function, and run to warm-up."""
        self.kw = dict(
            key=random.key(2025_04_25_15_57),
            bart=simple_init(P, N, NTREE),
            n_save=niters // 2,
            n_burn=niters // 2,
            n_skip=1,
            callback=lambda **_: None,
        )

        # adapt arguments for old versions
        sig = signature(run_mcmc)
        if 'callback' not in sig.parameters:
            self.kw.pop('callback')

        # catch bug and skip if found
        try:
            array_kw = {k: v for k, v in self.kw.items() if isinstance(v, jnp.ndarray)}
            nonarray_kw = {
                k: v for k, v in self.kw.items() if not isinstance(v, jnp.ndarray)
            }
            partial_run_mcmc = partial(run_mcmc, **nonarray_kw)
            eval_shape(partial_run_mcmc, **array_kw)
        except ZeroDivisionError:
            if niters:
                raise
            else:
                msg = 'skipping due to division by zero bug with zero iterations'
                raise NotImplementedError(msg) from None

        # decide how much to cold-start
        match cache:
            case 'cold':
                clear_caches()
            case 'warm':
                # prepare copies of the args because of buffer donation
                key = jnp.copy(self.kw['key'])
                bart = tree_map(jnp.copy, self.kw['bart'])
                self.time_run_mcmc(mode)
                # put copies in place of donated buffers
                self.kw.update(key=key, bart=bart)

    # skip compile with 0 iters bc in past versions 0 iters had a noop code path
    @skip_for_params(list(product(['compile'], [0], params[2])))
    def time_run_mcmc(self, mode: Mode, *_):
        """Time running or compiling the function."""
        match mode:
            case 'compile':
                # re-wrap and jit the function in the benchmark case because otherwise
                # the compiled function gets cached even if I call `compile` explicitly
                @partial(
                    jit, static_argnames=('n_save', 'n_skip', 'n_burn', 'callback')
                )
                def f(**kw):
                    return run_mcmc(**kw)

                f.lower(**self.kw).compile()

            case 'run':
                block_until_ready(run_mcmc(**self.kw))


class TimeStep:
    """Benchmarks of `mcmcstep.step`."""

    params: tuple[tuple[Mode, ...], tuple[Kind, ...]] = (
        ('compile', 'run'),
        ('plain', 'binary', 'weights'),
    )
    param_names = ('mode', 'kind')

    def setup(self, mode: Mode, kind: Kind):
        """Create an initial MCMC state and random seed, compile & warm-up."""
        self.kw = dict(
            key=random.key(2025_06_24_12_07), bart=simple_init(P, N, NTREE, kind)
        )

        self.func = jit(step).lower(**self.kw).compile()
        if mode == 'run':
            block_until_ready(self.func(**self.kw))

    def time_step(self, mode: Mode, _):
        """Time running compiling `step` or running it a few times."""
        match mode:
            case 'compile':

                @jit
                def f(**kw):
                    return step(**kw)

                f.lower(**self.kw).compile()

            case 'run':
                block_until_ready(self.func(**self.kw))


class TimeGbart:
    """Benchmarks of `BART.gbart`."""

    # asv config
    params: tuple[tuple[int, ...], tuple[Cache, ...]] = ((0, NITERS), ('cold', 'warm'))
    param_names = ('niters', 'cache')
    warmup_time = 0.0
    number = 1

    def setup(self, niters: int, cache: Cache):
        """Prepare the arguments and run once to warm-up."""
        key = random.key(2025_06_24_14_55)
        keys = list(random.split(key, 3))

        # generate simulated data
        sigma = 0.1
        T = 2
        X = random.uniform(keys.pop(), (P, N), float, -2, 2)
        f = lambda X: jnp.sum(jnp.cos(2 * jnp.pi / T * X), axis=0)
        y = f(X) + sigma * random.normal(keys.pop(), (N,))

        # arguments
        self.kw = dict(
            x_train=X, y_train=y, nskip=niters // 2, ndpost=niters // 2, seed=keys.pop()
        )

        # decide how much to cold-start
        match cache:
            case 'cold':
                clear_caches()
            case 'warm':
                self.time_gbart()

    def time_gbart(self, *_):
        """Time instantiating the class."""
        with redirect_stdout(StringIO()):
            bart = gbart(**self.kw)
            block_until_ready((bart._mcmc_state, bart._main_trace))
