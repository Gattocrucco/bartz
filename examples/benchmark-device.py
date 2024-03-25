import time
import functools

import jax
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import bartz

# TODO: I'm timing pre-processing the covariates. I guess it's fast, but to be sure I should time just the MCMC.

# Devices to benchmark on
devices = {
    'cpu': jax.devices('cpu')[0],
    # 'gpu': jax.devices('gpu')[0],
}

# BART config
kw = dict(nskip=0, ndpost=100, numcut=255)
nchains = 8

# DGP definition
nvec = [100, 200, 500, 1000, 2000, 5000, 10000] # number of datapoints
p = 10 # number of covariates
sigma = 0.1 # noise standard deviation
def f(x): # conditional mean
    T = 2
    return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=0)
def gen_X(key, p, n):
    return random.uniform(key, (p, n), float, -2, 2)
def gen_y(key, X):
    return f(X) + sigma * random.normal(key, X.shape[1:])

# random seed
key = random.key(202403241634)

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.time = time.perf_counter() - self.start

times = {}
for n in nvec:

    print(f'n = {n}')

    # generate data
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    X = gen_X(subkey1, p, n)
    y = f(X) + sigma * random.normal(subkey2, (n,))

    # repeat for each device
    for label, device in devices.items():

        keys = random.split(subkey3, nchains)
        X, y, keys = jax.device_put((X, y, keys), device)

        # pre-compile function
        @jax.jit
        @functools.partial(jax.vmap, in_axes=(None, None, 0))
        def task(X, y, key):
            return bartz.BART.gbart(X, y, **kw, seed=key)._mcmc_state
        task = task.lower(X, y, keys).compile()

        # clock
        print(label)
        with Timer() as timer:
            jax.block_until_ready(task(X, y, keys))

        # save result
        times.setdefault(label, []).append(timer.time)
        print(f'{timer.time:.2g}s')

# plot execution time
fig, ax = plt.subplots(num='benchmark', clear=True)

times_array = jnp.array(list(times.values())).T
ax.plot(nvec, times_array, label=list(times.keys()))
ax.legend()
ax.set(xlabel='n', ylabel='time (s)', xscale='log', yscale='log')
ax.grid(linestyle='--')
ax.grid(which='minor', linestyle=':')

fig.show()
