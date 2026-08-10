"""Maximal Lyapunov exponent via the Kantz algorithm (TISEAN ``lyap_k``)."""

import numpy as np


def lyap_k(series, dim, delay, max_steps, eps_list=None, n_ref=None):
    """Estimate the maximal Lyapunov exponent using Kantz's method.

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
    eps_list : array_like or None
        Neighbourhood radii (auto when ``None``).
    n_ref : int or None
        Number of reference points (``None`` = all).

    Returns
    -------
    tuple
        ``(times, divergence)`` arrays; slope of the linear region
        gives the maximal exponent.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: Kantz maximal Lyapunov exponent.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
