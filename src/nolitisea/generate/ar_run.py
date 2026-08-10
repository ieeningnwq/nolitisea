"""Iterate an autoregressive model (TISEAN ``ar-run``)."""

import numpy as np


def ar_run(coeffs, n, x_init=None, noise_std=0.0, seed=None):
    """Iterate an AR model for ``n`` steps.

    Parameters
    ----------
    coeffs : array_like
        AR coefficients ``[a1, a2, ..., ap]`` so that
        ``x[t] = sum a_i * x[t-i] + noise``.
    n : int
        Number of steps to iterate.
    x_init : array_like or None
        Initial ``p`` values (zeros when ``None``).
    noise_std : float, default 0.0
        Standard deviation of the driving noise.
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    numpy.ndarray
        Generated series of length ``n``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: iterate an AR model.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
