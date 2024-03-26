import time
import pathlib
import subprocess
import datetime
import gc

import jax
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import bartz

from tests.rbartpackages import BART

# BART config
kw_shared = dict(ntree=200, nskip=0, ndpost=1000, numcut=255)
kw_bartz = dict(maxdepth=6)
kw_BART = dict(usequants=True, rm_const=False, mc_cores=1)

# TODO add comparisons w.r.t. p (at n=1000) and ntree (at n=1000, p=10)

# DGP definition
nvec = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000] # number of datapoints
p = 100 # number of covariates
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

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.time = time.perf_counter() - self.start

times = []
for n in [10] + nvec: # the 10 is for warmup and will be dropped from results

    print(f'\n********** n = {n} **********\n')

    key, key1, key2, key3, key4 = random.split(key, 5)

    # generate data
    X = gen_X(key1, p, n)
    y = f(X) + sigma * random.normal(key2, (n,))

    # pre-compile bartz function
    @jax.jit
    def bartz_task(X, y, key):
        bart = bartz.BART.gbart(X, y, **kw_bartz, **kw_shared, seed=key)
        yhat = bart.yhat_train # needed for fairness: bartz does not compute
                               # this if not used explicitly, BART does
        return bart._mcmc_state, yhat
    compiled = bartz_task.lower(X, y, key3).compile()

    # fit with bartz
    with Timer() as t1:
        jax.block_until_ready(compiled(X, y, key3))

    # fit with BART
    seed = random.randint(key4, (), 0, jnp.uint32(2 ** 31)).item()
    with Timer() as t2:
        BART.mc_gbart(X.T, y, **kw_BART, **kw_shared, seed=seed)

    times.append([t1.time, t2.time])

    del X, y
    gc.collect()

times = jnp.array(times[1:])

def textbox(ax, text, loc='lower left', **kw):
    """
    Draw a box with text on a matplotlib plot.
    
    Parameters
    ----------
    ax : matplotlib axis
        The plot where the text box is drawn.
    text : str
        The text.
    loc : str
        The location of the box. Format: 'lower/center/upper left/center/right'.
    
    Keyword arguments
    -----------------
    Additional keyword arguments are passed to ax.annotate. If you pass a
    dictionary for the `bbox` argument, the defaults are updated instead of
    resetting the bounding box properties.
    
    Return
    ------
    The return value is that from ax.annotate.
    """
    
    M = 8
    locparams = {
        'lower left'   : dict(xy=(0  , 0  ), xytext=( M,  M), va='bottom', ha='left'  ),
        'lower center' : dict(xy=(0.5, 0  ), xytext=( 0,  M), va='bottom', ha='center'),
        'lower right'  : dict(xy=(1  , 0  ), xytext=(-M,  M), va='bottom', ha='right' ),
        'center left'  : dict(xy=(0  , 0.5), xytext=( M,  0), va='center', ha='left'  ),
        'center center': dict(xy=(0.5, 0.5), xytext=( 0,  0), va='center', ha='center'),
        'center right' : dict(xy=(1  , 0.5), xytext=(-M,  0), va='center', ha='right' ),
        'upper left'   : dict(xy=(0  , 1  ), xytext=( M, -M), va='top'   , ha='left'  ),
        'upper center' : dict(xy=(0.5, 1  ), xytext=( 0, -M), va='top'   , ha='center'),
        'upper right'  : dict(xy=(1  , 1  ), xytext=(-M, -M), va='top'   , ha='right' ),
    }
    
    kwargs = dict(
        xycoords='axes fraction',
        textcoords='offset points',
        bbox=dict(
            facecolor='white',
            alpha=0.75,
            edgecolor='#ccc',
            boxstyle='round'
        ),
    )
    kwargs.update(locparams[loc])
    
    newkw = dict(kw)
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(kwargs.get(k, None), dict):
            kwargs[k].update(v)
            newkw.pop(k)
    kwargs.update(newkw)
    
    return ax.annotate(text, **kwargs)

# plot execution time
fig, ax = plt.subplots(num='benchmark', clear=True, layout='constrained')

ax.plot(nvec, times, label=['bartz', 'BART'])
ax.legend()
ax.set(title='bartz vs. BART', xlabel='n', ylabel='time (s)', xscale='log', yscale='log')
ax.grid(linestyle='--')
ax.grid(which='minor', linestyle=':')
textbox(ax, f"""\
p = {p}
ntree = {kw_shared['ntree']}
maxdepth = {kw_bartz['maxdepth']}
ndpost = {kw_shared['ndpost']}
nskip = {kw_shared['nskip']}
train predictions included
serial""", loc='lower right')

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
