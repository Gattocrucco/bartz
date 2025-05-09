import functools

from jax import numpy as jnp
from jax import random
import jax
import bartz

class TimeCompilation:

    def setup(self):
        p = 2
        n = 30
        sigma = 0.1
        T = 2
        key = random.key(202504251557)
        keys = list(random.split(key, 16))
        X = random.uniform(keys.pop(), (p, n), float, -2, 2)
        f = lambda X: jnp.sum(jnp.cos(2 * jnp.pi / T * X), axis=0)
        y = f(X) + sigma * random.normal(keys.pop(), (n,))
        bart = bartz.BART.gbart(X, y, ndpost=0, nskip=0)
        self.args = (
            bart._mcmc_state,
            1, 1, 1,
            bartz.mcmcloop.make_simple_print_callback(100),
            keys.pop(),
        )

    def time_run_mcmc_compile(self):
        @functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
        def f(*args):
            return bartz.mcmcloop.run_mcmc(*args)
        f.lower(*self.args).compile()
