"""Cross-correlation integral (TISEAN ``xc2``)."""

import numpy as np


def cross_correlation_integral(a, b, dim, delay, eps_list):
    """Compute the cross-correlation integral of two embedded series.

    Parameters
    ----------
    a : array_like
        First series.
    b : array_like
        Second series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    eps_list : array_like
        Radii at which to evaluate the integral.

    Returns
    -------
    numpy.ndarray
        Cross-correlation values aligned with ``eps_list``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: cross-correlation integral.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
