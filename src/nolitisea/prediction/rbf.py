"""Radial basis function prediction (TISEAN ``rbf``)."""

import numpy as np


def fit_rbf(series, dim, delay, n_centers, eps=None):
    """Fit a radial basis function model.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    n_centers : int
        Number of RBF centres.
    eps : float or None
        RBF width (auto when ``None``).

    Returns
    -------
    dict
        Model containing ``centers``, ``weights`` and ``eps``.
    """
    raise NotImplementedError


def predict_rbf(model, series, n_steps):
    """Iterate an RBF model ``n_steps`` into the future.

    Parameters
    ----------
    model : dict
        Model returned by :func:`fit_rbf`.
    series : array_like
        Input scalar series.
    n_steps : int
        Number of iterations.

    Returns
    -------
    numpy.ndarray
        Predicted future values.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: fit and/or iterate an RBF model.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
