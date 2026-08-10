"""Finite-size Lyapunov exponent (TISEAN ``fsle``)."""

import numpy as np


def fsle(series, dim, delay, eps_list, n_ref=None):
    """Estimate the finite-size Lyapunov exponent.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    eps_list : array_like
        Initial perturbation sizes.
    n_ref : int or None
        Number of reference points (``None`` = all).

    Returns
    -------
    tuple
        ``(eps_list, fsle_values)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: finite-size Lyapunov exponent.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
