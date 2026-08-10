"""Renyi entropies of order q (TISEAN ``boxcount``)."""

import numpy as np


def renyi_entropy(series, dim, delay, q, eps):
    """Estimate the order-``q`` Renyi entropy via box counting.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    q : float
        Renyi order.
    eps : float
        Box size.

    Returns
    -------
    float
        Estimated Renyi entropy.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: box-counting Renyi entropy.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
