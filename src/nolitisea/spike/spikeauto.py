"""Autocorrelation of event times (TISEAN ``spikeauto``)."""

import numpy as np


def spike_autocorrelation(event_times, max_lag, bin_width=1.0):
    """Compute the autocorrelation of an event-time series.

    Parameters
    ----------
    event_times : array_like
        Sorted event times.
    max_lag : float
        Maximum lag to consider.
    bin_width : float, default 1.0
        Width of the histogram bins.

    Returns
    -------
    tuple
        ``(lags, counts)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: spike autocorrelation.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
