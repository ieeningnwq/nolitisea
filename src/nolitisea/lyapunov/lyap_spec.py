"""Full Lyapunov spectrum (TISEAN ``lyap_spec``)."""

import numpy as np


def lyap_spec(series, dim, delay, n_iter, dt=1.0):
    """Estimate the full Lyapunov spectrum via orthonormal renormalisation.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension (number of exponents).
    delay : int
        Time delay.
    n_iter : int
        Number of renormalisation steps.
    dt : float, default 1.0
        Sampling interval.

    Returns
    -------
    numpy.ndarray
        Array of Lyapunov exponents (descending order).
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: full Lyapunov spectrum.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
