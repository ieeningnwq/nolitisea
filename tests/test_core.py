"""Unit tests for the core infrastructure."""

import numpy as np


def test_delay_embedding_shape():
    """delay_embedding should return (n - (dim-1)*delay, dim)."""
    raise NotImplementedError


def test_find_neighbors_within_eps():
    """find_neighbors should only return points within eps."""
    raise NotImplementedError


def test_rescale_data_range():
    """rescale_data should map the series into [lo, hi]."""
    raise NotImplementedError


def test_variance_consistency():
    """variance should match numpy's mean and variance."""
    raise NotImplementedError
