import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from functools import partial
# pip install -e.

import jax
from jax import numpy as jnp
from jax import random
import numpy as np
import pandas

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
    # y = latent

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

outcomes = pandas.read_csv('./data/NHANES_outcomes.csv')
predictors = pandas.read_csv('./data/NHANES_predictors.csv')
outcomes_names = outcomes.columns.tolist()
X_full = predictors.values
y_full = outcomes.values
print(f"Shape of full X matrix: {X_full.shape}")
print(f"Shape of full y vector: {y_full.shape}")

n_total_samples = X_full.shape[0]
n_train = int(n_total_samples * 0.7) 
n_test = n_total_samples - n_train

X_train, X_test = X_full[n_train:, :], X_full[:n_train, :]
y_train, y_test = y_full[n_train:, :], y_full[:n_train, :]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

Data = namedtuple('Data', 'X_train y_train X_test y_test')
data = Data(X_train=X_train.T, y_train=y_train, X_test=X_test.T, y_test=y_test)

device = jax.devices()[0]
data = jax.device_put(data, device)

print("Data successfully loaded, processed, and moved to device.")


n_train = 1000  # number of training points
p = 1000           # number of predictors/features
sigma = 0.1        # error standard deviation

n_test = 1200      # number of test points
n_tree = 200    # number of trees used by bartz

# seeds for random sampling
keys = list(random.split(random.key(202404161853), 2))

# generate the data on CPU to avoid running out of GPU memory
# cpu = jax.devices('cpu')[0]
# key = jax.device_put(keys.pop(), cpu) # the random key is the only jax-array input, so it determines the device used
# data = make_synthetic_dataset(key, n_train, n_test, p, sigma)

# # move the data to GPU (if there is a GPU)
# device = jax.devices()[0] # the default jax device is gpu if there is one
# data = jax.device_put(data, device)

# run bartz
start = perf_counter()
# bart = gbart(data.X_train, data.y_train, ntree=n_tree, printevery=10, seed=keys.pop())
# mvy_train = np.stack((data.y_train, random.normal(key, shape=(n_train,))), axis=1)
# mvy_test= np.stack((data.y_test, random.normal(key, shape=(n_test,))), axis=1)
# mvy_train = jnp.stack((data.y_train, jnp.ones(n_train)), axis=1)
# mvy_test  = jnp.stack((data.y_test,  jnp.ones(n_test)),  axis=1)

# print("size of y is now ", mvy_train.shape)
print("size of y is now ", data.y_train.shape)
# bart = gbart(data.X_train, mvy_train, ntree=n_tree, nskip = 1000, ndpost = 1000, printevery=100, seed=keys.pop())
bart = gbart(data.X_train, data.y_train, ntree=n_tree, nskip = 1000, ndpost = 1000, printevery=100, seed=keys.pop())

end = perf_counter()

# compute predictions
yhat_test = bart.predict(data.X_test) # posterior samples, n_samples x n_test
yhat_test_mean = jnp.mean(yhat_test, axis=0) # posterior mean point-by-point
yhat_test_var = jnp.var(yhat_test, axis=0) # posterior variance point-by-point
sigam2_cov = bart.sigma2_cov_mean
print('covariance matrix is, ', sigam2_cov)

# consider ATE
print("data.X_test.shape", data.X_test.shape)

X_test_treated = data.X_test.at[0, :].set(1)
X_test_control = data.X_test.at[0, :].set(0)
print('X_test_treated is now:',X_test_treated)
ate_test = bart.predict(X_test_treated) - bart.predict(X_test_control)

print(ate_test.shape)
ite_test_mean = jnp.mean(yhat_test, axis=0) # posterior mean point-by-point
ate_test_mean = jnp.mean(ite_test_mean, axis = 0)
ate_test_var = jnp.var(yhat_test, axis=0)
print("ate_test_mean is, ", ate_test_mean)

# RMSE 
# rmse = jnp.sqrt(jnp.mean(jnp.square(yhat_test_mean - data.y_test)))
rmse = jnp.sqrt(jnp.mean(jnp.square(yhat_test_mean - data.y_test)))
rmse_per_col = jnp.sqrt(jnp.mean(jnp.square(yhat_test_mean - data.y_test), axis=0))
for i, rmse in enumerate(rmse_per_col):
    col_name = outcomes_names[i] if i < len(outcomes_names) else f"col_{i}"
    print(f"RMSE for {col_name}: {rmse:.3f}")
print('yhat_test is of shape', yhat_test.shape)
print('yhat_test ptrdicted is', yhat_test_mean[1:5,])
print('yhat_test truth is', data.y_test[1:5, ])
print(f'RMSE: {rmse:#.2g}')

# expected_error_variance = jnp.mean(jnp.square(bart.sigma))
# expected_rmse = jnp.sqrt(jnp.mean(yhat_test_var + expected_error_variance))
# avg_sigma = jnp.sqrt(expected_error_variance)

# print(f'total sdev: {jnp.std(data.y_train):#.2g}')
# print(f'error sdev: {sigma:#.2g}')
# print(f'RMSE: {rmse:#.2g}')
# print(f'expected RMSE: {expected_rmse:#.2g}')
# print(f'model error sdev: {avg_sigma:#.2g}')
print(f'time: {(end - start) / 60:#.2g} min')