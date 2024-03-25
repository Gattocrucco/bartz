import time
import functools

import jax
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import bartz

# Devices to benchmark on
devices = {
    'cpu': jax.devices('cpu')[0],
    # 'gpu': jax.devices('gpu')[0],
}

# BART config
mcmc_iterations = 1
nchains = 8

# DGP definition
nvec = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000] # number of datapoints
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

    print(f'n = {n}', end='', flush=True)

    # generate data
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    X = gen_X(subkey1, p, n)
    y = f(X) + sigma * random.normal(subkey2, (n,))

    # build initial mcmc state
    state = bartz.BART.gbart(X, y, nskip=0, ndpost=0, seed=0)._mcmc_state

    # clean up memory
    del X, y

    # repeat for each device
    for label, device in devices.items():

        # commit inputs to the target device
        keys = random.split(subkey3, nchains)
        state, keys = jax.device_put((state, keys), device)

        # pre-compile mcmc loop
        @jax.jit
        @functools.partial(jax.vmap, in_axes=(None, 0))
        def task(state, key):
            return bartz.mcmcloop.run_mcmc(state, 0, mcmc_iterations, 1, lambda **_: None, key)
        task = task.lower(state, keys).compile()

        # clock
        with Timer() as timer:
            jax.block_until_ready(task(state, keys))

        # save result
        times.setdefault(label, []).append(timer.time)
        print(f', {label}: {timer.time:.2g}s', end='', flush=True)

    print()

# plot execution time
fig, ax = plt.subplots(num='benchmark', clear=True)

times_array = jnp.array(list(times.values())).T
ax.plot(nvec, times_array, label=list(times.keys()))
ax.legend()
ax.set(xlabel='n', ylabel='time (s)', xscale='log', yscale='log')
ax.grid(linestyle='--')
ax.grid(which='minor', linestyle=':')

fig.show()
