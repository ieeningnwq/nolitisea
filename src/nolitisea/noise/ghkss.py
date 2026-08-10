"""Locally linear projection noise reduction (TISEAN ``ghkss``)."""

import numpy as np


def ghkss(series, dim, delay, q=None, n_neighbors=0, n_iter=1):
    """Grassberger-H Kantz-Schreiber noise reduction via local linear maps.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    q : int or None
        Number of noise directions to project out (auto when ``None``).
    n_neighbors : int, default 0
        Minimum number of neighbours (0 = use all within ``eps``).
    n_iter : int, default 1
        Number of correction passes.

    Returns
    -------
    numpy.ndarray
        Corrected series.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: nonlinear noise reduction.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
