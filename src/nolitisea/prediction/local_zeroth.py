"""Locally zeroth-order (constant) prediction.

Replaces TISEAN ``lzo-test``, ``lzo-run`` and ``lzo-gm``.
"""

import numpy as np


def lzo_test(series, dim, delay, n_forecast=1, eps=None, n_neighbors=0):
    """Single-step prediction test using a locally constant model.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    n_forecast : int, default 1
        Forecast horizon.
    eps : float or None
        Neighbourhood radius (auto when ``None``).
    n_neighbors : int, default 0
        Minimum number of neighbours (0 = use all within ``eps``).

    Returns
    -------
    tuple
        ``(predicted, error)`` arrays.
    """
    raise NotImplementedError


def lzo_run(series, dim, delay, n_steps, eps=None):
    """Iterate a locally constant model ``n_steps`` into the future.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    n_steps : int
        Number of iterations.
    eps : float or None
        Neighbourhood radius.

    Returns
    -------
    numpy.ndarray
        Predicted future values of length ``n_steps``.
    """
    raise NotImplementedError


def lzo_gm(series, dim, delay):
    """Compare local zeroth-order prediction with the global mean.

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
    tuple
        ``(local_error, global_error)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: dispatch to one of the lzo tools.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
