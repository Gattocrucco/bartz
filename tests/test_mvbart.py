"""Test multivariate BART components."""

from jax import numpy as jnp
from jax import random
from numpy.testing import assert_allclose

from bartz.mcmcstep import _sample_wishart_bartlett


def test_wishart_bartlett_output():
    """Test the basic properties of the wishart sampler output."""
    key = random.key(123)
    scale_key, sample_key = random.split(key)
    df = 5
    k = 3
    # Create a random positive definite matrix for the scale
    A = random.normal(scale_key, (k, k))
    scale = A @ A.T + jnp.eye(k)
    scale_inv = jnp.linalg.inv(scale)

    # Sample from the wishart distribution
    wishart_sample = _sample_wishart_bartlett(sample_key, df, scale_inv)

    # Check shape
    assert wishart_sample.shape == (k, k)
    assert_allclose(wishart_sample, wishart_sample.T, rtol=1e-6)

    # Check for positive definiteness (eigenvalues should be positive)
    eigenvalues = jnp.linalg.eigvalsh(wishart_sample)
    assert jnp.all(eigenvalues > 0)
