"""Post-processing of the correlation integral.

Replaces TISEAN ``c2t``, ``c2g``, ``c2d`` and ``av-d2``.
"""

import numpy as np


def c2t(eps_list, c_values):
    """Takens estimator of the correlation dimension.

    Parameters
    ----------
    eps_list : array_like
        Radii.
    c_values : array_like
        Correlation integral values.

    Returns
    -------
    float
        Takens estimate of D2.
    """
    raise NotImplementedError


def c2g(eps_list, c_values, sigma=None):
    """Gaussian-kernel estimator of C2.

    Parameters
    ----------
    eps_list : array_like
        Radii.
    c_values : array_like
        Correlation integral values.
    sigma : float or None
        Kernel width (auto when ``None``).

    Returns
    -------
    numpy.ndarray
        Gaussian-kernel C2 values.
    """
    raise NotImplementedError


def c2d(eps_list, c_values, log_eps=True):
    """Local slope of the correlation integral.

    Parameters
    ----------
    eps_list : array_like
        Radii.
    c_values : array_like
        Correlation integral values.
    log_eps : bool, default True
        Use logarithmic spacing of ``eps``.

    Returns
    -------
    tuple
        ``(eps_mid, slopes)`` arrays.
    """
    raise NotImplementedError


def av_d2(eps_list, c_values, window=5):
    """Smooth the d2 output with a moving average.

    Parameters
    ----------
    eps_list : array_like
        Radii.
    c_values : array_like
        Correlation integral values.
    window : int, default 5
        Smoothing window size.

    Returns
    -------
    tuple
        ``(eps_list, smoothed)`` arrays.
    """
    raise NotImplementedError
