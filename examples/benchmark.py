import time
import functools

import jax
from jax import numpy as jnp
from jax import random
import bartz

# BART config
kw = dict(ntree=200, numcut=255, maxdepth=6)
ndpost = 1000

# DGP definition
n = 1000 # number of datapoints
p = 100 # number of covariates
def f(x): # conditional mean
    T = 2
    return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=0)
def gen_X(key, p, n):
    return random.uniform(key, (p, n), float, -2, 2)
def gen_y(key, X):
    return f(X) + 0.1 * random.normal(key, X.shape[1:])

# set up random seed
key = random.key(202403281647)
key, key1, key2, key3, key4 = random.split(key, 5)

# generate data
X = gen_X(key1, p, n)
y = gen_y(key2, X)

# create initial state
print('initialize...')
state = bartz.BART.gbart(X, y, nskip=0, ndpost=0, seed=key3, **kw)._mcmc_state

# compile mcmc loop
print('compile...')
def callback(**_):
    jax.debug.callback(lambda: print('.', end='', flush=True))
def run_bart(state, key):
    return bartz.mcmcloop.run_mcmc(state, 0, ndpost, 1, callback, key)
run_bart = jax.jit(run_bart).lower(state, key4).compile()

# run bartz
print('run...')
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.time = time.perf_counter() - self.start
with Timer() as timer:
    jax.block_until_ready(run_bart(state, key4))
print()

print(f'{ndpost} iterations in {timer.time:#.2g}s ({timer.time / ndpost:#.2g}s per iteration)')
