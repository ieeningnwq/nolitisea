"""Resampling utilities (TISEAN ``resample``)."""

import numpy as np


def resample(series, factor):
    """Resample by an integer factor.

    Parameters
    ----------
    series : array_like
        Input series.
    factor : int
        Resampling factor (``>1`` = upsample, ``<1`` = downsample).

    Returns
    -------
    numpy.ndarray
        Resampled series.
    """
    raise NotImplementedError


def resample_interp(series, new_dt, old_dt=1.0):
    """Resample to a new sampling interval via linear interpolation.

    Parameters
    ----------
    series : array_like
        Input series.
    new_dt : float
        Target sampling interval.
    old_dt : float, default 1.0
        Original sampling interval.

    Returns
    -------
    numpy.ndarray
        Resampled series.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: resample a series.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
