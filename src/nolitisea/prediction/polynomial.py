"""Polynomial model fitting.

Replaces TISEAN ``polynom``, ``polynomp``, ``polyback`` and ``polypar``.
"""

import numpy as np


def fit_polynom(series, dim, delay, degree, terms=None):
    """Fit a polynomial model.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    degree : int
        Maximum polynomial degree.
    terms : list[tuple] or None
        Specific monomial exponents to include (``polynomp`` mode).

    Returns
    -------
    dict
        Fitted coefficients and term list.
    """
    raise NotImplementedError


def polyback(series, dim, delay, degree, threshold=0.05):
    """Backward-elimination polynomial fitting.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    degree : int
        Maximum polynomial degree.
    threshold : float, default 0.05
        Significance threshold for term removal.

    Returns
    -------
    dict
        Selected terms and coefficients.
    """
    raise NotImplementedError


def polypar(dim, degree):
    """Generate a parameter file listing all monomials.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    list[tuple]
        List of monomial exponent tuples.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: polynomial model fitting.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
