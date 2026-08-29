"""Correlation integral and dimension (TISEAN ``d2``)."""

import numpy as np


def correlation_integral(series, dim, delay, eps_list, n_ref=None):
    """Compute the correlation integral ``C(eps)`` for several radii.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    eps_list : array_like
        Radii at which to evaluate ``C``.
    n_ref : int or None
        Number of reference points (``None`` = all).

    Returns
    -------
    numpy.ndarray
        ``C(eps)`` values aligned with ``eps_list``.
    """
    raise NotImplementedError


def d2(series, dim_max, delay, eps_list=None):
    """Main entry: correlation integral for embeddings up to ``dim_max``.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim_max : int
        Maximum embedding dimension.
    delay : int
        Time delay.
    eps_list : array_like or None
        Radii (auto when ``None``).

    Returns
    -------
    dict
        Mapping ``dim -> (eps_list, c_values)``.
    """
    raise NotImplementedError


