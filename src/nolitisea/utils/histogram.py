"""Histogram computation (TISEAN ``histogram``)."""

import numpy as np


def histogram(series, bins=20, rng=None):
    """Compute a histogram of the series.

    Parameters
    ----------
    series : array_like
        Input series.
    bins : int, default 20
        Number of bins.
    rng : tuple of float or None
        Optional ``(min, max)`` range.

    Returns
    -------
    tuple
        ``(counts, centers)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: compute a histogram.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
