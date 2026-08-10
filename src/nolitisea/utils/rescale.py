"""Rescale and summarise a series (TISEAN ``rescale`` / ``rms``)."""

import numpy as np


def rescale(series, lo=0.0, hi=1.0):
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
    numpy.ndarray
        Rescaled series.
    """
    raise NotImplementedError


def rms(series):
    """Return mean, standard deviation and value range.

    Parameters
    ----------
    series : array_like
        Input series.

    Returns
    -------
    tuple
        ``(mean, std, min, max)``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: rescale a series.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
