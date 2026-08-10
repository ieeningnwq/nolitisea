"""Nonstationarity test via cross-prediction (TISEAN ``nstat_z``)."""

import numpy as np


def nstat_z(series, dim, delay, n_parts=10):
    """Split the series into parts and cross-predict each part.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    n_parts : int, default 10
        Number of segments.

    Returns
    -------
    numpy.ndarray
        Error matrix of shape ``(n_parts, n_parts)``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: nonstationarity cross-prediction test.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
