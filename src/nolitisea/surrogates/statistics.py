"""Discriminating statistics for surrogate testing."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import lag_block_delay_embed

__all__ = ["predict_stat", "time_reversibility"]


def time_reversibility(series, delay=1):
    """Time-reversal asymmetry statistic.

    For a scalar series ``x`` and a time ``delay`` the statistic is::

        T = sum (x[n] - x[n-delay])**3 / sum (x[n] - x[n-delay])**2

    with both sums running over all ``n >= delay``.  ``T`` vanishes (in
    expectation) for time-reversible processes such as Gaussian linear
    noise and is clearly non-zero for asymmetric chaotic maps, which
    makes it a standard discriminating statistic for surrogate data
    tests.

    Parameters
    ----------
    series : array_like
        Input scalar (1-D) series.
    delay : int, default 1
        Time delay of the finite difference.

    Returns
    -------
    float
        Time-reversal asymmetry value.  ``nan`` for a constant series
        (zero denominator).

    Raises
    ------
    ValueError
        If ``series`` is not 1-D, ``delay < 1``, or the series is too
        short for the requested delay.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("series must be a 1-D scalar series")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if x.size <= delay:
        raise ValueError(
            f"series of length {x.size} is too short for delay={delay}"
        )

    dx = x[delay:] - x[:-delay]
    t2 = float(np.sum(dx**2))
    t3 = float(np.sum(dx**3))
    if t2 == 0.0:
        return float("nan")
    return t3 / t2


def predict_stat(series, dim, delay, n_forecast=1, eps=None, frac=None):
    """Prediction-error discriminating statistic.

    Locally constant (zeroth-order) forecast of a scalar series:

    1. Build the ``dim``-dimensional delay embedding with time
       ``delay``; the neighbour database contains exactly those vectors
       whose ``n_forecast``-steps-ahead target still lies inside the
       series.
    2. For every reference point, find all neighbours within Chebyshev
       (max-norm) distance ``eps`` and drop the reference point itself.
    3. Predict ``x[t + n_forecast]`` as the mean of the neighbours'
       values ``x[j + n_forecast]``.  When the reference point has no
       other neighbour inside the ball, fall back to the series mean.
    4. Return the root-mean-square prediction error over all valid
       reference points.

    Parameters
    ----------
    series : array_like
        Input scalar (1-D) series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay of the embedding.
    n_forecast : int, default 1
        Forecast horizon in samples.
    eps : float, optional
        Absolute neighbourhood radius. Either ``eps``
        or ``frac`` must be given; when both are supplied, ``frac``
        takes precedence.
    frac : float, optional
        Neighbourhood radius as a fraction of the population standard
        deviation of the series.

    Returns
    -------
    float
        Root-mean-square ``n_forecast``-steps-ahead prediction error.

    Raises
    ------
    ValueError
        If neither ``eps`` nor ``frac`` is a positive number, if
        ``dim``/``delay``/``n_forecast`` are invalid, if ``series`` is
        not 1-D, or if the series is too short for the requested
        embedding and horizon.

    Notes
    -----
    The neighbour set is the full-vector Chebyshev ball (the coordinate
    order inside a delay vector is irrelevant for the max norm).
    Neighbours are accumulated in ascending index order — a
    deterministic rule that replaces implementation-defined box-scan
    orders, so only floating-point rounding of the mean can differ
    between implementations.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("series must be a 1-D scalar series")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if n_forecast < 1:
        raise ValueError(f"n_forecast must be >= 1, got {n_forecast}")

    n = x.size
    valid_start = (dim - 1) * delay
    n_refs = n - n_forecast - valid_start
    if n_refs < 1:
        raise ValueError(
            f"series of length {n} is too short for dim={dim}, "
            f"delay={delay}, n_forecast={n_forecast} (need at least "
            f"{valid_start + n_forecast + 1} samples)"
        )

    if frac is not None and frac > 0:
        # Population standard deviation (divide by n)
        eps = float(np.std(x)) * float(frac)
    if eps is None or eps <= 0:
        raise ValueError(
            "either eps or frac must be given as a positive number"
        )

    # Database: delay vectors ending at t = valid_start .. n - n_forecast - 1
    # (row r <-> endpoint t = r + valid_start).
    E = lag_block_delay_embed(x, embed=dim, delay=delay)[:n_refs]
    tree = cKDTree(E)

    # x[t + n_forecast] for every reference endpoint t; length n_refs.
    targets = x[valid_start + n_forecast :]
    mean_x = float(x.mean())

    sq_error = np.empty(n_refs)
    for r in range(n_refs):
        nbrs = np.sort(
            np.asarray(tree.query_ball_point(E[r], eps, p=np.inf), dtype=np.intp)
        )
        if nbrs.size > 1:
            # The reference point itself is always at distance 0 -> drop it.
            # Ball rows map to endpoint times t = row + valid_start; the
            # forecast reads y(endpoint + ifc).
            contrib = nbrs[nbrs != r]
            pred = x[contrib + valid_start + n_forecast].sum() / (nbrs.size - 1)
        else:
            pred = mean_x
        sq_error[r] = (targets[r] - pred) ** 2

    return float(np.sqrt(sq_error.mean()))
