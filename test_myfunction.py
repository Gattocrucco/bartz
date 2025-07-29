import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from functools import partial

import jax
from jax import numpy as jnp
from jax import random

@partial(jax.jit, static_argnums=(1, 2))
def dgp(key, n, p, max_interactions, error_sdev):
    """ DGP. Uses data-based standardization, so you have to generate train &
    test at once. """

    # split random key
    keys = list(random.split(key, 4))

    # generate matrices
    X = random.uniform(keys.pop(), (p, n))
    beta = random.normal(keys.pop(), (p,))
    A = random.normal(keys.pop(), (p, p))
    error = random.normal(keys.pop(), (n,))

    # make A banded to limit the number of interactions
    num_nonzero = 1 + (max_interactions - 1) // 2
    num_nonzero = jnp.clip(num_nonzero, 0, p)
    interaction_pattern = jnp.arange(p) < num_nonzero
    multi_roll = jax.vmap(jnp.roll, in_axes=(None, 0))
    nonzero = multi_roll(interaction_pattern, jnp.arange(p))
    A *= nonzero

    # compute terms
    linear = beta @ X
    quadratic = jnp.einsum('ai,bi,ab->i', X, X, A)
    error *= error_sdev

    # equalize the terms
    latent = linear / jnp.std(linear) + quadratic / jnp.std(quadratic)
    latent /= jnp.std(latent)  # because linear and quadratic are correlated

    # compute response
    y = latent + error

    return X, y

from collections import namedtuple

Data = namedtuple('Data', 'X_train y_train X_test y_test')

def make_synthetic_dataset(key, n_train, n_test, p, sigma):
    X, y = dgp(key, n_train + n_test, p, 5, sigma)
    X_train, y_train = X[:, :n_train], y[:n_train]
    X_test, y_test = X[:, n_train:], y[n_train:]
    return Data(X_train, y_train, X_test, y_test)

from time import perf_counter

from bartz.BART import gbart

n_train = 1000  # number of training points
p = 1000           # number of predictors/features
sigma = 0.1        # error standard deviation

n_test = 1000      # number of test points
n_tree = 1000    # number of trees used by bartz

# seeds for random sampling
keys = list(random.split(random.key(202404161853), 2))

# generate the data on CPU to avoid running out of GPU memory
cpu = jax.devices('cpu')[0]
key = jax.device_put(keys.pop(), cpu) # the random key is the only jax-array input, so it determines the device used
data = make_synthetic_dataset(key, n_train, n_test, p, sigma)

# move the data to GPU (if there is a GPU)
device = jax.devices()[0] # the default jax device is gpu if there is one
data = jax.device_put(data, device)

# run bartz
start = perf_counter()
bart = gbart(data.X_train, data.y_train, ntree=n_tree, printevery=10, seed=keys.pop())
end = perf_counter()

# compute predictions
yhat_test = bart.predict(data.X_test) # posterior samples, n_samples x n_test
yhat_test_mean = jnp.mean(yhat_test, axis=0) # posterior mean point-by-point
yhat_test_var = jnp.var(yhat_test, axis=0) # posterior variance point-by-point

# RMSE
rmse = jnp.sqrt(jnp.mean(jnp.square(yhat_test_mean - data.y_test)))
expected_error_variance = jnp.mean(jnp.square(bart.sigma))
expected_rmse = jnp.sqrt(jnp.mean(yhat_test_var + expected_error_variance))
avg_sigma = jnp.sqrt(expected_error_variance)

print(f'total sdev: {jnp.std(data.y_train):#.2g}')
print(f'error sdev: {sigma:#.2g}')
print(f'RMSE: {rmse:#.2g}')
print(f'expected RMSE: {expected_rmse:#.2g}')
print(f'model error sdev: {avg_sigma:#.2g}')
print(f'time: {(end - start) / 60:#.2g} min')