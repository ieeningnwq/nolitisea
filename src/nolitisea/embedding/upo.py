"""Unstable periodic orbits (TISEAN ``upo`` / ``upoembed``)."""

import numpy as np


def find_upo(series, dim, delay, min_period, max_period, eps):
    """Search for unstable periodic orbits in phase space.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    min_period : int
        Shortest period to consider.
    max_period : int
        Longest period to consider.
    eps : float
        Closure tolerance.

    Returns
    -------
    list
        List of detected orbits with period and stability.
    """
    raise NotImplementedError


def upo_embed(upo_data):
    """Convert ``find_upo`` output into plottable embedding files.

    Parameters
    ----------
    upo_data : list
        Output of :func:`find_upo`.

    Returns
    -------
    list
        Embedding arrays, one per orbit.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: find unstable periodic orbits.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
