"""Linear cross-correlation function."""

import numpy as np

__all__ = ["cross_correlation"]


def cross_correlation(a, b, max_lag=100):
    """Normalized linear cross-correlation of two scalar series.

    Parameters
    ----------
    a, b : array_like
        The two series; they must have the same length.
    max_lag : int, default 100
        Largest absolute lag.

    Returns
    -------
    dict
        ``"lags"`` : lags ``-max_lag .. max_lag``, shape ``(2 * max_lag + 1,)``.
        ``"corr"`` : normalized cross-correlation at those lags.
        ``"mean_a"``, ``"std_a"`` : mean and biased (``1/n``) standard
        deviation of the raw first series.
        ``"mean_b"``, ``"std_b"`` : same for the second series.

    Raises
    ------
    ValueError
        For empty input, a length mismatch, a negative ``max_lag``, or a
        series with zero variance.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = x.size
    if n == 0 or y.size == 0:
        raise ValueError("input series must not be empty")
    if n != y.size:
        raise ValueError(f"length mismatch: a has {n} points, b has {y.size}")
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    if n < 2:
        raise ValueError("at least two points are required")
    if max_lag >= n:
        max_lag = n - 1

    mean_a = float(np.mean(x))
    mean_b = float(np.mean(y))
    std_a = float(np.std(x))
    std_b = float(np.std(y))
    
    if std_a == 0.0 or std_b == 0.0:
        raise ValueError("cross-correlation is undefined for a constant series")

    xc = x - mean_a
    yc = y - mean_b
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(lags.size)
    for k, lag in enumerate(lags):
        if lag >= 0:
            num = float(np.dot(xc[: n - lag], yc[lag:]))
            count = n - lag
        else:
            num = float(np.dot(xc[-lag:], yc[: n + lag]))
            count = n + lag
        corr[k] = num / count / std_a / std_b

    return {
        "lags": lags,
        "corr": corr,
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
    }
