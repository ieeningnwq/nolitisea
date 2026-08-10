"""Maximum-entropy (AR-based) spectrum estimation (TISEAN ``mem_spec``)."""

import numpy as np


def mem_spectrum(series, order, n_freq=512):
    """Estimate the power spectrum via the maximum-entropy method.

    Parameters
    ----------
    series : array_like
        Input series.
    order : int
        AR model order used for the estimator.
    n_freq : int, default 512
        Number of frequency bins.

    Returns
    -------
    tuple
        ``(freqs, power)`` arrays of length ``n_freq``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: compute the maximum-entropy spectrum.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
