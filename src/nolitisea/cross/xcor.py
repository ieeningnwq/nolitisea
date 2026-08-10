"""Linear cross-correlation (TISEAN ``xcor``)."""

import numpy as np


def cross_correlation(a, b, max_lag):
    """Compute the cross-correlation between two series for lags ``0..max_lag``.

    Parameters
    ----------
    a : array_like
        First series.
    b : array_like
        Second series.
    max_lag : int
        Maximum lag.

    Returns
    -------
    numpy.ndarray
        Cross-correlation values of length ``max_lag + 1``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: linear cross-correlation.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
