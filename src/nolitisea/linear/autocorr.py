"""Autocorrelation function (TISEAN ``corr``)."""

import numpy as np


def autocorrelation(x, max_lag=None, norm=True, detrend=True):
    """Return the autocorrelation of the given scalar time series.

    Calculates the autocorrelation of the given scalar time series
    using the Wiener-Khinchin theorem.

    Parameters
    ----------
    x : array_like
        1-D real time series of length N.
    max_lag : int, optional (default = N)
        Return the autocorrelation only up to this time delay.
    norm : bool, optional (default = True)
        Normalize the autocorrelation so that it is equal to 1 for
        zero time delay.
    detrend: bool, optional (default = True)
        Subtract the mean from the time series (i.e., a constant
        detrend).  This is done so that for uncorrelated data, the
        autocorrelation vanishes for all nonzero time delays.

    Returns
    -------
    r : array
        Array with the autocorrelation up to max_lag.
    """
    x = np.asarray(x)
    N = len(x)

    if not max_lag:
        max_lag = N
    else:
        max_lag = min(N, max_lag)

    if detrend:
        x = x - np.mean(x)

    y = np.fft.fft(x, 2 * N - 1)
    r = np.real(np.fft.ifft(y * y.conj(), 2 * N - 1))

    if norm:
        return r[:max_lag] / r[0]
    else:
        return r[:max_lag]
