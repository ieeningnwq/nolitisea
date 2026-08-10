"""Maximal Lyapunov exponent via Rosenstein's algorithm (TISEAN ``lyap_r``)."""

import numpy as np


def lyap_r(series, dim, delay, max_steps, min_sep=0, eps=None):
    """Estimate the maximal Lyapunov exponent using Rosenstein's method.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    max_steps : int
        Maximum evolution time.
    min_sep : int, default 0
        Minimum temporal separation between neighbours.
    eps : float or None
        Neighbourhood radius (auto when ``None``).

    Returns
    -------
    tuple
        ``(times, divergence)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: Rosenstein maximal Lyapunov exponent.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
