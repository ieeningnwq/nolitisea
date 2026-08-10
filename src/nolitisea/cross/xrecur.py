"""Cross-recurrence plot (TISEAN ``xrecur``)."""

import numpy as np


def cross_recurrence(a, b, dim, delay, eps, metric="euclidean"):
    """Build a cross-recurrence matrix between two embedded series.

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
    eps : float
        Recurrence threshold.
    metric : str, default "euclidean"
        Distance metric name.

    Returns
    -------
    numpy.ndarray
        Boolean matrix of shape ``(n_a, n_b)``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: cross-recurrence plot.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
