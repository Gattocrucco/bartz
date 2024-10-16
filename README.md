[![PyPI](https://img.shields.io/pypi/v/bartz)](https://pypi.org/project/bartz/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13931477.svg)](https://doi.org/10.5281/zenodo.13931477)


# BART vectoriZed

A branchless vectorized implementation of Bayesian Additive Regression Trees (BART) in JAX.

BART is a nonparametric Bayesian regression technique. Given predictors $X$ and responses $y$, BART finds a function to predict $y$ given $X$. The result of the inference is a sample of possible functions, representing the uncertainty over the determination of the function.

This Python module provides an implementation of BART that runs on GPU, to process large datasets faster. It is also good on CPU. Most other implementations of BART are for R, and run on CPU only.

On CPU, bartz runs at the speed of dbarts (the fastest implementation I know of), but using half the memory. On GPU, the speed premium depends on sample size; with 50000 datapoints and 5000 trees, on an Nvidia Tesla V100 GPU it's 12 times faster than a single Apple M1 CPU core, and this factor is linearly proportional to the number of datapoints.

The maximum practically realizable speedup is currently 200x.

[This Colab notebook](https://colab.research.google.com/drive/1BHl_Nnh0VY-cUvCe5Topub4mgnOkGGO5?usp=sharing) runs BART with $n=100\,000$ observations, $p=1000$ predictors, $10\,000$ trees, for 1000 MCMC iterations, in 5 minutes.

## Links

- [Documentation (latest release)](https://gattocrucco.github.io/bartz/docs)
- [Documentation (development version)](https://gattocrucco.github.io/bartz/docs-dev)
- [Repository](https://github.com/Gattocrucco/bartz)
- [Code coverage](https://gattocrucco.github.io/bartz/coverage)

## Other BART packages

- [stochtree](https://github.com/StochasticTree) C++ library with R and Python bindings taylored to researchers who want to make their own BART variants
- [bnptools](https://github.com/rsparapa/bnptools) Feature-rich R packages for BART and some variants
- [dbarts](https://github.com/vdorie/dbarts) Fast R package
- [bartMachine](https://github.com/kapelner/bartMachine) Fast R package, supports missing predictors imputation
- [SoftBART](https://github.com/theodds/SoftBART) R package with a smooth version of BART
- [bcf](https://github.com/jaredsmurray/bcf) R package for BCF, version of BART for causal inference
- [flexBART](https://github.com/skdeshpande91/flexBART) Fast R package, supports categorical predictors
- [flexBCF](https://github.com/skdeshpande91/flexBCF) R package, bcf optimized for large datasets
- [XBART](https://github.com/JingyuHe/XBART) R/Python package, faster variant of BART
- [BART](https://github.com/JingyuHe/BART) R package, BART warm-started with XBART
- [XBCF](https://github.com/socket778/XBCF) XBART BCF
- [BayesTree](https://cran.r-project.org/package=BayesTree) R package, original BART implementation
- [bartCause](https://github.com/vdorie/bartCause) R package, pre-made BART-based workflows for causal inference
- [OpenBT](https://bitbucket.org/mpratola/openbt) Heteroskedastic BART, rotate & perturb proposals, C++ library with R interface
- [stan4bart](https://github.com/vdorie/stan4bart)
- [VCBART](https://github.com/skdeshpande91/VCBART)
- [lsqfitgp](https://github.com/Gattocrucco/lsqfitgp) Infinite trees limit of BART
- [monbart](https://github.com/jaredsmurray/monbart)
- [mBART](https://github.com/remcc/mBART_shlib)
- [SequentialBART](https://github.com/mjdaniels/SequentialBART)
- [sparseBART](https://github.com/cspanbauer/sparseBART)
- [pymc-bart](https://github.com/pymc-devs/pymc-bart) BART within PyMC
- [semibart](https://github.com/zeldow/semibart)
- [CSP-BART](https://github.com/ebprado/CSP-BART)
- [AMBARTI](https://github.com/ebprado/AMBARTI)
- [MOTR-BART](https://github.com/ebprado/MOTR-BART)
- [bcfbma](https://github.com/EoghanONeill/bcfbma)
- [bartBMAnew](https://github.com/EoghanONeill/bartBMAnew)
- [BART-BMA](https://github.com/BelindaHernandez/BART-BMA) (superseded by bartBMAnew)
- [gpbart](https://github.com/MateusMaiaDS/gpbart)
- [GPBART](https://github.com/nchenderson/GPBART)
- [bartpy](https://github.com/JakeColtman/bartpy)
- [BayesTreePrior](https://github.com/AlexiaJM/BayesTreePrior) Sample the prior of BART
- [BayesTree.jl](https://github.com/mathcg/BayesTree.jl)
- [longbet](https://github.com/google/longbet)
- [XSBART](https://github.com/richael008/XSBART) Accelerated SoftBart
- [mpibart](https://matthewpratola.com/mpibart) Old parallel implementation
- [bartMan](https://github.com/AlanInglis/bartMan) R package, posterior analysis and diagnostics
