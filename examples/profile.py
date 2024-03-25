import jax
from jax import profiler
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import bartz

# BART config
kw = dict(nskip=100, ndpost=100)

# DGP definition
n = 1000 # number of datapoints
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
key = random.key(202403212114)
key, key1, key2, key3, key4, key5 = random.split(key, 6)

# generate data
X_train = gen_X(key1, p, n)
y_train = gen_y(key2, X_train)
X_test = gen_X(key3, p, n)
y_test = gen_y(key4, X_test)

# fit with bartz
with profiler.trace('jax-trace', create_perfetto_trace=True):
    bart = bartz.BART.gbart(X_train, y_train, x_test=X_test, **kw, seed=key5)
    jax.block_until_ready(bart._mcmc_state)
