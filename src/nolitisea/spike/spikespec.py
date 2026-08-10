"""Power spectrum of event times (TISEAN ``spikespec``)."""

import numpy as np


def spike_spectrum(event_times, n_freq=512, max_freq=None):
    """Compute the power spectrum of an event-time series.

    Parameters
    ----------
    event_times : array_like
        Sorted event times.
    n_freq : int, default 512
        Number of frequency bins.
    max_freq : float or None
        Maximum frequency (auto when ``None``).

    Returns
    -------
    tuple
        ``(freqs, power)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: spike power spectrum.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
