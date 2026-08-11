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

    Raises
    ------
    ValueError
        If the input series is not 1-dimensional or if the series
        length is insufficient for the requested embedding parameters.
    """
    arr = np.asarray(series)

    # Input validation
    if arr.ndim != 1:
        raise ValueError("Input series must be a 1-dimensional array.")

    n_points = len(arr) - (dim - 1) * delay

    if n_points <= 0:
        raise ValueError(
            f"Series length ({len(arr)}) is too short for the specified "
            f"dimension ({dim}) and delay ({delay})."
        )

    # Create the delay embedding by slicing the array with varying offsets
    # and stacking them as columns in a 2D matrix
    embedded = np.column_stack(
        [arr[i * delay : n_points + i * delay] for i in range(dim)]
    )

    return embedded


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
