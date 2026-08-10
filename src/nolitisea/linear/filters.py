"""Linear filters: notch, Wiener, 1-2-1 low-pass, Savitzky-Golay.

Replaces TISEAN ``notch``, ``wiener``, ``low121`` and ``sav_gol``.
"""

import numpy as np


def notch_filter(series, freq, q=30.0, fs=1.0):
    """Apply a notch filter rejecting ``freq``.

    Parameters
    ----------
    series : array_like
        Input series.
    freq : float
        Frequency to reject (in cycles per unit time).
    q : float, default 30.0
        Quality factor.
    fs : float, default 1.0
        Sampling frequency.

    Returns
    -------
    numpy.ndarray
        Filtered series.
    """
    raise NotImplementedError


def wiener_filter(series, noise_var):
    """Apply a frequency-domain Wiener filter.

    Parameters
    ----------
    series : array_like
        Input series.
    noise_var : float
        Estimated noise variance.

    Returns
    -------
    numpy.ndarray
        Filtered series.
    """
    raise NotImplementedError


def low121(series):
    """Apply a simple 1-2-1 low-pass filter.

    Parameters
    ----------
    series : array_like
        Input series.

    Returns
    -------
    numpy.ndarray
        Filtered series.
    """
    raise NotImplementedError


def savitzky_golay(series, window, order, deriv=0):
    """Apply a Savitzky-Golay smoothing/derivative filter.

    Parameters
    ----------
    series : array_like
        Input series.
    window : int
        Window length (must be odd).
    order : int
        Polynomial order.
    deriv : int, default 0
        Derivative order (0 = smoothing).

    Returns
    -------
    numpy.ndarray
        Filtered series.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: dispatch to one of the filters above.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
