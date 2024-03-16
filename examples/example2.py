import warnings
import sys

from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import bartz

# sys.path.insert(1, 'examples')
from rbartpackages import BART

warnings.filterwarnings('error', r'scatter inputs have incompatible types.*', FutureWarning)

# DGP config
n = 1000 # number of datapoints (train + test)
p = 1 # number of covariates
sigma = 0.1 # noise standard deviation
def f(x): # conditional mean
    T = 2
    return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=0)

# generate data
key = random.key(202403142235)
key, key1, key2, key3 = random.split(key, 4)
X = random.normal(key1, (p, n))
y = f(X) + sigma * random.normal(key2, (n,))

# split in train/test
n_train = n // 2
X_train, X_test = X[:, :n_train], X[:, n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# set test = train for debugging
X_test = X_train
y_test = y_train

# fit with bartz
kw = dict(ntree=1, nskip=20, ndpost=20, numcut=255, printevery=20)
bart1 = bartz.BART(X_train, y_train, x_test=X_test, **kw, seed=key3)

# fit with BART
bart2 = BART.mc_gbart(X_train.T, y_train, X_test.T, **kw, usequants=True, rm_const=False, mc_cores=1, seed=20240314)

barts = dict(bartz=bart1, BART=bart2)

# compute RMSE
print(f'\ndata sdev = {y_test.std():#.2g}')
for label, bart in barts.items():
    resid = y_test - bart.yhat_test_mean
    rmse = jnp.sqrt(resid @ resid / resid.size)
    print(f'{label}:')
    print(f'    sigma: {bart.sigma.mean():#.2g} (true: {sigma:#.2g})')
    print(f'    RMSE: {rmse.item():#.2g}')

# plot true vs. predicted
fig, axs = plt.subplots(1, 2, num='example2', clear=True, figsize=[10, 5])

ax = axs[0]
l = jnp.inf
h = -jnp.inf
for label, bart in barts.items():
    l = min([l, bart.yhat_test_mean.min(), y_test.min()])
    h = max([h, bart.yhat_test_mean.max(), y_test.max()])
    ax.plot(bart.yhat_test_mean, y_test, '.', label=label)
ax.plot([l, h], [l, h], color='lightgray')
ax.set(xlabel='predicted (post mean)', ylabel='true', title='true vs. predicted on test set')
ax.legend()

ax = axs[1]
ax.plot(X_test[0], y_test, '.', label='data')
for label, bart in barts.items():
    ax.plot(X_test[0], bart.yhat_test_mean, 'o', label=label, markerfacecolor='none')
ax.legend()

fig.show()
