"""ARIMA model fitting (TISEAN ``arima-model``)."""

import numpy as np


def fit_arima(series, p, d, q):
    """Fit an ARIMA(``p``, ``d``, ``q``) model.

    Parameters
    ----------
    series : array_like
        Input series.
    p : int
        AR order.
    d : int
        Differencing order.
    q : int
        MA order.

    Returns
    -------
    dict
        Fitted parameters and residuals.
    """
    raise NotImplementedError


def difference(series, d):
    """Apply ``d``-th order differencing.

    Parameters
    ----------
    series : array_like
        Input series.
    d : int
        Differencing order.

    Returns
    -------
    numpy.ndarray
        Differenced series.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: fit and possibly iterate an ARIMA model.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
