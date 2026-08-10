"""Generate an Ikeda-map time series (TISEAN ``ikeda``)."""

import numpy as np


def ikeda(n, u=0.9, x0=0.0, y0=0.0):
    """Iterate the Ikeda map ``n`` steps.

    Parameters
    ----------
    n : int
        Number of iterations.
    u : float, default 0.9
        Map parameter ``u``.
    x0 : float, default 0.0
        Initial ``x``.
    y0 : float, default 0.0
        Initial ``y``.

    Returns
    -------
    numpy.ndarray
        The ``x`` component of length ``n``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: generate and write an Ikeda series.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
