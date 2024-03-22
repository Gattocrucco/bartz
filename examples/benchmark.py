import time
import pathlib
import subprocess
import datetime

import jax
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import bartz

from tests.rbartpackages import BART

# BART config
kw = dict(nskip=100, ndpost=100, numcut=255)

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

# set up random seed
key = random.key(202403212335)

# modify arguments for R BART
rkw = kw.copy()
rkw.update(usequants=True, rm_const=False, mc_cores=1)
if 'lamda' in kw:
    rkw['lambda'] = rkw.pop['lamda']

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.time = time.perf_counter() - self.start

times = []
for n in nvec:

    print(f'\n********** n = {n} **********\n')

    key, key1, key2, key3, key4 = random.split(key, 5)

    # generate data
    X = gen_X(key1, p, n)
    y = f(X) + sigma * random.normal(key2, (n,))

    # fit with bartz
    bartz.BART(X, y, **kw, seed=key3) # run once for jit
    with Timer() as t1:
        bart = bartz.BART(X, y, **kw, seed=key3)
        jax.block_until_ready(bart._mcmc_state)

    # fit with BART
    seed = random.randint(key4, (), 0, jnp.uint32(2 ** 31)).item()
    with Timer() as t2:
        BART.mc_gbart(X.T, y, **rkw, seed=seed)

    times.append([t1.time, t2.time])
times = jnp.array(times)

# plot execution time
fig, ax = plt.subplots(num='benchmark', clear=True)

ax.plot(nvec, times, label=['bartz', 'BART'])
ax.legend()
ax.set(xlabel='n', ylabel='time (s)', xscale='log', yscale='log')
ax.grid(linestyle='--')
ax.grid(which='minor', linestyle=':')

fig.show()

# save figure
script = pathlib.Path(__file__)
outdir = script.with_suffix('')
outdir.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True)
commit = commit.stdout.decode().strip()[:7]
figname = f'{fig.get_label()}_{timestamp}_{commit}.png'
fig.savefig(outdir / figname)
