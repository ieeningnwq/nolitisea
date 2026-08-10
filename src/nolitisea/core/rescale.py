"""Scaling and statistics helpers.

Replaces ``rescale_data`` and ``variance`` from the TISEAN routines.
"""

import numpy as np


def rescale_data(series, lo=0.0, hi=1.0):
    """Linearly rescale a series to ``[lo, hi]``.

    Parameters
    ----------
    series : array_like
        Input series.
    lo : float, default 0.0
        Target minimum.
    hi : float, default 1.0
        Target maximum.

    Returns
    -------
    tuple
        ``(scaled, scale, offset)`` so that
        ``scaled = series * scale + offset``.
    """
    raise NotImplementedError


def variance(series):
    """Return the mean and variance of a series.

    Parameters
    ----------
    series : array_like
        Input series.

    Returns
    -------
    tuple
        ``(mean, var)``.
    """
    raise NotImplementedError


def rms_normalize(series):
    """Remove the mean and divide by the standard deviation.

    Parameters
    ----------
    series : array_like
        Input series.

    Returns
    -------
    numpy.ndarray
        Normalised series with zero mean and unit variance.
    """
    raise NotImplementedError
