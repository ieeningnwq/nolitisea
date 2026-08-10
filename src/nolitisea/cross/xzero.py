"""Locally zeroth-order cross-prediction (TISEAN ``xzero``)."""

import numpy as np


def xzero(a, b, dim, delay, eps=None):
    """Predict ``b`` from ``a`` using a locally constant model.

    Parameters
    ----------
    a : array_like
        Driver series.
    b : array_like
        Target series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    eps : float or None
        Neighbourhood radius (auto when ``None``).

    Returns
    -------
    tuple
        ``(predicted, error)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: locally zeroth-order cross-prediction.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
