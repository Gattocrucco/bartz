import functools
import inspect

import jax
from jax import numpy as jnp
from jax import random

import bartz


class TimeCompilation:
    def setup(self):
        key = random.key(202504251557)
        keys = list(random.split(key, 3))

        # generate simulated data
        p = 2
        n = 30
        sigma = 0.1
        T = 2
        X = random.uniform(keys.pop(), (p, n), float, -2, 2)
        f = lambda X: jnp.sum(jnp.cos(2 * jnp.pi / T * X), axis=0)
        y = f(X) + sigma * random.normal(keys.pop(), (n,))

        # dry run the interface to generate an initial bart state conveniently
        if inspect.ismodule(bartz.BART):
            gbart = bartz.BART.gbart
        else:
            gbart = bartz.BART  # pre v0.2.0
        bart = gbart(X, y, ndpost=0, nskip=0)
        self.kw = dict(
            key=keys.pop(),
            bart=bart._mcmc_state,
            n_save=1,
            n_burn=1,
            n_skip=1,
            callback=lambda **_: None,
        )

    def time_run_mcmc_compile(self):
        @functools.partial(
            jax.jit, static_argnames=('n_save', 'n_burn', 'n_skip', 'callback')
        )
        def f(**kw):
            return bartz.mcmcloop.run_mcmc(**kw)

        f.lower(**self.kw).compile()
