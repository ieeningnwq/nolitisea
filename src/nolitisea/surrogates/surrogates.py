"""AAFT / iterative AAFT surrogate generation (TISEAN ``surrogates``)."""

import numpy as np


def aaft(series, n_surrogates=1, seed=None):
    """Generate amplitude-adjusted Fourier transform surrogates.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    n_surrogates : int, default 1
        Number of surrogates to generate.
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_surrogates, len(series))``.
    """
    raise NotImplementedError


def iterative_aaft(series, n_surrogates=1, n_iter=10, seed=None):
    """Generate iterated amplitude-adjusted Fourier surrogates.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    n_surrogates : int, default 1
        Number of surrogates.
    n_iter : int, default 10
        Number of refinement iterations.
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_surrogates, len(series))``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: generate surrogate data.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
