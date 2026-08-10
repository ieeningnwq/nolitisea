"""Autoregressive model fitting and iteration (TISEAN ``ar-model``)."""

import numpy as np


def fit_ar(series, order):
    """Fit an AR(``order``) model by least squares.

    Parameters
    ----------
    series : array_like
        Input series.
    order : int
        AR order ``p``.

    Returns
    -------
    tuple
        ``(coeffs, noise_var)`` where ``coeffs`` has length ``order``.
    """
    raise NotImplementedError


def iterate_ar(coeffs, n, x_init=None, noise_std=0.0):
    """Iterate an AR model for ``n`` steps.

    Parameters
    ----------
    coeffs : array_like
        AR coefficients.
    n : int
        Number of steps.
    x_init : array_like or None
        Initial values (length ``len(coeffs)``).
    noise_std : float, default 0.0
        Driving-noise standard deviation.

    Returns
    -------
    numpy.ndarray
        Generated series of length ``n``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: fit and optionally iterate an AR model.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
