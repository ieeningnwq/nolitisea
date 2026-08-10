"""Shared pytest fixtures providing synthetic chaotic series."""

import numpy as np
import pytest


@pytest.fixture
def henon_series():
    """Return a 5000-point Henon x-component series."""
    raise NotImplementedError


@pytest.fixture
def lorenz_series():
    """Return a 5000-point Lorenz x-component series (dt=0.01)."""
    raise NotImplementedError


@pytest.fixture
def white_noise():
    """Return 5000 standard-normal samples seeded for reproducibility."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(5000)
