"""False nearest neighbours for embedding-dimension selection (TISEAN ``false_nearest``)."""

import numpy as np


def false_nearest(series, dim_max, delay=1, rt=10.0, fs=2.0):
    """Compute the fraction of false nearest neighbours for each dimension.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim_max : int
        Maximum embedding dimension to test.
    delay : int, default 1
        Time delay.
    rt : float, default 10.0
        Tolerance for the distance ratio.
    fs : float, default 2.0
        Attractor size factor for the absolute criterion.

    Returns
    -------
    tuple
        ``(dims, fractions)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: false nearest neighbours.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
