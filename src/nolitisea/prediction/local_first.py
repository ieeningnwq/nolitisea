"""Locally first-order (linear) prediction.

Replaces TISEAN ``lfo-test``, ``lfo-run`` and ``lfo-ar``.
"""

import numpy as np


def lfo_test(series, dim, delay, n_forecast=1, eps=None):
    """Single-step prediction test using a locally linear model.

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
        Neighbourhood radius.

    Returns
    -------
    tuple
        ``(predicted, error)`` arrays.
    """
    raise NotImplementedError


def lfo_run(series, dim, delay, n_steps, eps=None):
    """Iterate a locally linear model ``n_steps`` into the future.

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
        Predicted future values.
    """
    raise NotImplementedError


def lfo_ar(series, dim, delay, ar_order):
    """Compare local linear prediction with a global AR model.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    ar_order : int
        Order of the global AR model.

    Returns
    -------
    tuple
        ``(local_error, ar_error)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: dispatch to one of the lfo tools.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
