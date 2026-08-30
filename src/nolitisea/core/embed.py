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

    Each variable is embedded with its own dimension and delay; all rows
    share the same base index, i.e. row ``r`` holds
    ``[x_i(r), x_i(r + delay_i), ..., x_i(r + (dim_i - 1) * delay_i)]``
    for every variable ``i``.  The number of rows is the minimum value
    compatible with all variables,
    ``n_points = min_i(n_i - (dim_i - 1) * delay_i)``.

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
        Array of shape ``(n_points, sum(dims))`` where the delay
        coordinates of variable ``i`` occupy the column block
        ``sum(dims[:i]) : sum(dims[:i + 1])``.

    Raises
    ------
    ValueError
        If ``series_list``, ``dims`` and ``delays`` disagree in length,
        if any embedding dimension or delay is below 1, or if any series
        is not 1-dimensional or too short for its requested embedding.
    """
    series_list = list(series_list)
    dims = list(dims)
    delays = list(delays)
    if not (len(series_list) == len(dims) == len(delays)):
        raise ValueError(
            "series_list, dims and delays must have the same length, got "
            f"{len(series_list)}, {len(dims)} and {len(delays)}."
        )
    if not series_list:
        raise ValueError("At least one series is required.")

    blocks = []
    n_points = None
    for i, (series, dim, delay) in enumerate(zip(series_list, dims, delays)):
        if dim < 1:
            raise ValueError(f"variable {i}: embedding dimension must be >= 1.")
        if delay < 1:
            raise ValueError(f"variable {i}: delay must be >= 1.")
        try:
            block = delay_embedding(series, dim, delay)
        except ValueError as exc:
            raise ValueError(f"variable {i}: {exc}") from exc
        n_points = (
            block.shape[0] if n_points is None else min(n_points, block.shape[0])
        )
        blocks.append(block)

    return np.hstack([block[:n_points] for block in blocks])


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
