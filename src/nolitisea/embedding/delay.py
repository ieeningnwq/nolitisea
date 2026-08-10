"""Delay embedding output (TISEAN ``delay``)."""

import numpy as np


def delay_vectors(series, dim, delay):
    """Return the delay-embedding matrix for inspection.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, dim)``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: write a delay embedding.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
