"""Permutation entropy of one or more time series."""

import math

import numpy as np

__all__ = ["permutation_entropy"]


def _ordinal_codes(windows):
    """Map every window to its ordinal pattern encoded as an integer.

    Parameters
    ----------
    windows : numpy.ndarray
        Array of shape ``(n_windows, n_series, order)`` holding the
        embedded windows.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n_windows, n_series)`` with codes in
        ``range(order!)``.  Equal values in a window are ordered by
        their time of appearance (earlier position gets the smaller
        rank), the tie convention of the original method.
    """
    order = windows.shape[-1]

    # Stable double argsort gives ranks; with tied values the earlier
    # position receives the smaller rank (stable ordering by index).
    ranks = np.argsort(
        np.argsort(windows, axis=-1, kind="stable"), axis=-1, kind="stable"
    )

    # Lehmer (factorial number system) encoding of each rank permutation,
    # which gives a bijective mapping onto range(order!).
    pairwise = ranks[..., :, None] > ranks[..., None, :]
    later_mask = np.triu(np.ones((order, order), dtype=bool), k=1)
    digits = np.sum(pairwise & later_mask, axis=-1)
    weights = np.array(
        [math.factorial(order - 1 - j) for j in range(order)], dtype=np.int64
    )
    return np.sum(digits * weights, axis=-1).astype(np.intp)


def permutation_entropy(series, order=3, delay=1, normalize=True):
    """Permutation entropy (Bandt and Pompe) of one or more time series.

    Each sliding window of length ``order`` sampled every ``delay``
    samples,

        X(t), X(t + delay), ..., X(t + (order - 1) * delay),

    is replaced by its ordinal pattern (the permutation that sorts the
    window values).  The Shannon entropy of the empirical distribution
    of the ``order!`` possible patterns is the permutation entropy.  It
    measures the irregularity of a series: monotonous or periodic
    series use few patterns and have low entropy, while uncorrelated
    noise visits all patterns almost uniformly and has high entropy.

    Parameters
    ----------
    series : array_like
        A 1-D array of shape ``(n_samples,)`` (a single series) or a
        2-D array of shape ``(n_samples, n_series)`` where each column
        is an independent time series and rows are time steps.
    order : int, default 3
        Embedding order (window length).  Must be >= 2.
    delay : int, default 1
        Time lag between consecutive coordinates of a window.
    normalize : bool, default True
        If True the entropy is divided by ``ln(order!)`` so the result
        lies in ``[0.0, 1.0]``; otherwise the Shannon entropy in nats
        is returned, bounded by ``ln(order!)``.

    Returns
    -------
    numpy.ndarray
        A 1-D array of shape ``(n_series,)`` with the permutation
        entropy of each column (shape ``(1,)`` for a 1-D input).

    Raises
    ------
    ValueError
        If the input is not 1-D or 2-D, ``order`` < 2, ``delay`` < 1,
        or the series is too short to form a single window
        (fewer than ``(order - 1) * delay + 1`` samples).

    Notes
    -----
    Tied values inside a window are ordered by their time of
    appearance (the earlier position gets the smaller rank), as in the
    original publication.  A constant series contains a single
    pattern and therefore has permutation entropy zero.

    References
    ----------
    .. [1] Bandt, C., & Pompe, B. (2002). Permutation entropy: a
           natural complexity measure for time series.  *Physical
           Review Letters*, 88(17), 174102.
    """
    x = np.asarray(series, dtype=np.float64)

    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(f"Input must be 1-D or 2-D, got {x.ndim}D.")

    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}.")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}.")

    n_samples, n_series = x.shape
    n_windows = n_samples - (order - 1) * delay
    if n_windows < 1:
        raise ValueError(
            f"series of length {n_samples} is too short for order={order}, "
            f"delay={delay}: need at least {(order - 1) * delay + 1} samples."
        )

    # Build all windows: shape (n_windows, n_series, order).
    windows = np.stack(
        [x[j * delay : j * delay + n_windows, :] for j in range(order)],
        axis=-1,
    )

    codes = _ordinal_codes(windows)
    n_patterns = math.factorial(order)

    entropies = np.empty(n_series, dtype=np.float64)
    for s in range(n_series):
        counts = np.bincount(codes[:, s], minlength=n_patterns).astype(np.float64)
        probabilities = counts[counts > 0.0] / n_windows
        entropies[s] = -np.sum(probabilities * np.log(probabilities))

    if normalize:
        entropies /= np.log(n_patterns)

    # Replace possible negative zeros (a single pattern yields -0.0).
    entropies[entropies == 0.0] = 0.0

    return entropies
