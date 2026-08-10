"""Delay-coordinate embedding routines shared by every phase-space method."""

import numpy as np


def delay_embedding(series, dim, delay=1):
    """Build a delay-coordinate embedding matrix.

    Parameters
    ----------
    series : array_like
        One-dimensional scalar series.
    dim : int
        Embedding dimension ``m``.
    delay : int, default 1
        Time delay ``tau`` between coordinates.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, dim)`` where
        ``n_points = len(series) - (dim - 1) * delay``.
    """
    raise NotImplementedError


def mixed_embedding(series_list, dims, delays):
    """Build a multivariate mixed delay embedding.

    Parameters
    ----------
    series_list : sequence of array_like
        One series per variable.
    dims : sequence of int
        Embedding dimension per variable.
    delays : sequence of int
        Delay per variable.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, sum(dims))``.
    """
    raise NotImplementedError


def embedding_indices(n, dim, delay):
    """Return the valid base indices usable for an embedding.

    Parameters
    ----------
    n : int
        Length of the original series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.

    Returns
    -------
    numpy.ndarray
        Integer array of base indices that fit inside the series.
    """
    raise NotImplementedError
