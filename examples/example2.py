import warnings

from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import bartz

from tests.rbartpackages import BART

warnings.filterwarnings('error', r'scatter inputs have incompatible types.*', FutureWarning)

# DGP config
n = 500 # number of datapoints
p = 1 # number of covariates
sigma = 0.1 # noise standard deviation
def f(x): # conditional mean
    T = 2
    return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=0)
# def f(x):
#     return jnp.sum(x, axis=0)
# def f(x):
#     return jnp.sum(jnp.abs(x), axis=0)
# def gen_X(key, p, n):
#     return random.normal(key, (p, n))
def gen_X(key, p, n):
    return random.uniform(key, (p, n), float, -2, 2)
# def gen_X(key, p, n):
#     return jnp.repeat(jnp.arange(n)[None, :] - n / 2, p, axis=0)

# set up random seed
key = random.key(202403142235)
key, key1, key2, key3, key4, key5, key6 = random.split(key, 7)

# generate data
X_train = gen_X(key1, p, n)
y_train = f(X_train) + sigma * random.normal(key2, (n,))
X_test = gen_X(key3, p, n)
y_test = f(X_test) + sigma * random.normal(key4, (n,))

# set test = train for debugging
# X_test = X_train
# y_test = y_train

# fit with bartz
kw = dict(ntree=50, nskip=1000, ndpost=1000, numcut=255, printevery=100)
bart1 = bartz.BART(X_train, y_train, x_test=X_test, **kw, seed=key5)

bad = bart1._check_trees()
total = bad.size
bad_count = jnp.count_nonzero(bad)
if bad_count:
    print(f'*******  bad trees: {bad_count}/{total}  *******')

# fit with BART
if 'lamda' in kw:
    kw.update({'lambda': kw.pop('lamda')})
bart2 = BART.mc_gbart(X_train.T, y_train, X_test.T, **kw, usequants=True, rm_const=False, mc_cores=1, seed=20240314)

barts = dict(bartz=bart1, BART=bart2)

# compute RMSE
print(f'\ndata sdev = {y_test.std():#.2g}')
for label, bart in barts.items():
    resid = y_test - bart.yhat_test_mean
    rmse = jnp.sqrt(resid @ resid / resid.size)
    sigma_m = jnp.sqrt(jnp.mean(jnp.square(bart.sigma)))
    print(f'{label}:')
    print(f'    sigma: {sigma_m:#.2g} (true: {sigma:#.2g})')
    print(f'    RMSE: {rmse.item():#.2g}')
# compute RMSE
print(f'\ndata sdev = {y_test.std():#.2g}')
for label, bart in barts.items():
    resid = y_test - bart.yhat_test_mean
    rmse = jnp.sqrt(resid @ resid / resid.size)
    y_pred = bart.yhat_test + bart.sigma[:, None] * random.normal(key6, bart.yhat_test.shape)
    totsdev = jnp.sqrt(jnp.var(y_pred, axis=0).mean())
    sigma_m = jnp.sqrt(jnp.mean(jnp.square(bart.sigma)))
    print(f'{label}:')
    print(f'    sigma: {sigma_m:#.2g} (true: {sigma:#.2g})')
    print(f'    avg_pred_sdev: {totsdev:#.2g}')
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
for label, bart in barts.items():
    ax.plot(X_test[0], bart.yhat_test_mean, '.', label=label)
ax.plot(X_test[0], y_test, 'o', label='data', markerfacecolor='none')
ax.legend()

fig.show()

# plot posterior histograms of sigma
fig, ax = plt.subplots(num='example2-sigma', clear=True)

ax.hist([bart1.sigma, bart2.sigma], bins='auto', density=True, label=list(barts))
ax.set(title='Posterior on error sdev', xlabel='$\\sigma$', ylabel='probability density')
ax.legend()

fig.show()
