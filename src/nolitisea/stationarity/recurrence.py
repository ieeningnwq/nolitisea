"""Recurrence plot (TISEAN ``recurr``)."""

import numpy as np


def recurrence_matrix(series, dim, delay, eps, metric="euclidean"):
    """Build a binary recurrence matrix.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    eps : float
        Recurrence threshold.
    metric : str, default "euclidean"
        Distance metric name.

    Returns
    -------
    numpy.ndarray
        Boolean matrix of shape ``(n_points, n_points)`` where
        ``R[i, j] = ||x_i - x_j|| < eps``.
    """
    raise NotImplementedError


def recurrence_rate(rmat):
    """Return the recurrence rate of a recurrence matrix.

    Parameters
    ----------
    rmat : numpy.ndarray
        Boolean recurrence matrix.

    Returns
    -------
    float
        Fraction of recurring points.
    """
    raise NotImplementedError
