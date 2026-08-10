"""Discriminating statistics for surrogate testing.

Replaces TISEAN ``timerev`` and ``predict``.
"""

import numpy as np


def time_reversibility(series):
    """Compute the time-reversal asymmetry statistic.

    Parameters
    ----------
    series : array_like
        Input scalar series.

    Returns
    -------
    float
        Time-reversal asymmetry value.
    """
    raise NotImplementedError


def predict_stat(series, dim, delay, n_forecast=1):
    """Compute a prediction-error discriminating statistic.

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

    Returns
    -------
    float
        Mean prediction error.
    """
    raise NotImplementedError
