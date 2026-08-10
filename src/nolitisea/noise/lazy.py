"""Simple nonlinear noise reduction (TISEAN ``lazy``)."""

import numpy as np


def lazy_noise_reduction(series, dim, delay, n_iter=1, eps=None):
    """Reduce noise by locally constant phase-space projection.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    n_iter : int, default 1
        Number of correction passes.
    eps : float or None
        Neighbourhood radius (auto when ``None``).

    Returns
    -------
    numpy.ndarray
        Corrected series.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: simple nonlinear noise reduction.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
