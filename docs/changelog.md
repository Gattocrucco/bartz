<!--
bartz/docs/changelog.md

Copyright (c) 2024, Giacomo Petrillo

This file is part of bartz.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

<!-- This changelog is written in Markdown and without line splits to make it
  copy-pastable to github releases. -->

# Changelog

## 0.2.0

* Rename `bartz.BART` to `bartz.BART.gbart`.
* Expose submodule `bartz.jaxext` with auxiliary functions for jax.
* Shorter compilation time if no burnin or saved samples are drawn. This is useful when using the interface only to create the initial MCMC state, or when saving all samples to inspect the warm-up phase.
* 20x faster on GPU.
* 2x faster on CPU.
* Use less temporary memory to quantilize covariates, avoiding out-of-memory errors on GPU.

## 0.1.0

* Optimize the MCMC step to only traverse each tree once.
* Now `bartz` runs at the same speed as the R package `BART` (tested at $p=10$, $n=100\ldots 10000$).
* The MCMC functions are heavily changed, but the interface is the same.

## 0.0.1

* `BART` has attributes `maxdepth`, `sigest`.
* Fix errors with scaling of noise variance prior.
* Fix iteration report when `keepevery` is not 1.
* Lower required versions of dependencies to allow running on Colab.

## 0.0

First release.
