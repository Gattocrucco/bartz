[![PyPI](https://img.shields.io/pypi/v/bartz)](https://pypi.org/project/bartz/)

# BART vectoriZed

A branchless vectorized implementation of Bayesian Additive Regression Trees (BART) in JAX.

BART is a nonparametric Bayesian regression technique. Given predictors $X$ and responses $y$, BART finds a function to predict $y$ given $X$. The result of the inference is a sample of possible functions, representing the uncertainty over the determination of the function.

This Python module provides an implementation of BART that runs on GPU, to process large datasets faster. It is also a good on CPU. Most other implementations of BART are for R, and run on CPU only.
