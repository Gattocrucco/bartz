import warnings

from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import bartz

warnings.filterwarnings('error', r'scatter inputs have incompatible types.*', FutureWarning)

# DGP config
n = 2000 # number of datapoints (train + test)
p = 1 # number of covariates
sigma = 0.1 # noise standard deviation
def f(x): # conditional mean
    R = 2
    r2 = jnp.einsum('ij,ij->j', x, x)
    return jnp.cos(2 * jnp.pi / R * jnp.sqrt(r2))

# generate data
key = random.key(202403132205)
key, key1, key2 = random.split(key, 3)
X = random.normal(key1, (p, n))
y = f(X) + sigma * random.normal(key2, (n,))

# split in train/test
n_train = n // 2
X_train, X_test = X[:, :n_train], X[:, n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# set test = train for debugging
# X_test = X_train
# y_test = y_train

# fit with bartz
bart = bartz.BART(X_train, y_train, x_test=X_test, ntree=200, nskip=500, ndpost=500, seed=91287346)

# compute RMSE
resid = y_test - bart.yhat_test_mean
rmse = jnp.sqrt(resid @ resid / resid.size)
print(f'data sdev = {y_test.std():#.2g}')
print(f'sigma: {bart.sigma.mean():#.2g} (true: {sigma:#.2g})')
print(f'RMSE: {rmse.item():#.2g}')

# plot true vs. predicted
fig, axs = plt.subplots(1, 2, num='example1', clear=True, figsize=[10, 5])

ax = axs[0]
l = min(bart.yhat_test_mean.min(), y_test.min())
h = max(bart.yhat_test_mean.max(), y_test.max())
ax.plot([l, h], [l, h], color='lightgray')
ax.plot(bart.yhat_test_mean, y_test, '.')
ax.set(xlabel='predicted (post mean)', ylabel='true', title='true vs. predicted on test set')

ax = axs[1]
ax.plot(X_test[0], y_test, '.', label='data')
ax.plot(X_test[0], bart.yhat_test_mean, 'o', label='pred', markerfacecolor='none')
ax.legend()

fig.show()
