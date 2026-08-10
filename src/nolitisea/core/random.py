"""Random number generation helpers.

Replaces ``rand.c`` and ``rand_arb_dist.c`` from the TISEAN routines.
"""

import numpy as np


def rnd_init(seed):
    """Seed the internal generator.

    Parameters
    ----------
    seed : int
        Non-negative seed value.
    """
    raise NotImplementedError


def gaussian(size=1):
    """Draw standard normal samples.

    Parameters
    ----------
    size : int or tuple, default 1
        Output shape.

    Returns
    -------
    numpy.ndarray or float
        Standard normal samples.
    """
    raise NotImplementedError


def uniform(size=1):
    """Draw samples uniformly from ``[0, 1)``.

    Parameters
    ----------
    size : int or tuple, default 1
        Output shape.

    Returns
    -------
    numpy.ndarray or float
        Uniform samples.
    """
    raise NotImplementedError


def rnd_long():
    """Return a random non-negative integer.

    Returns
    -------
    int
        Pseudo-random integer.
    """
    raise NotImplementedError


def rand_arb_dist(values, weights, size, seed=None):
    """Sample from an arbitrary discrete distribution.

    Parameters
    ----------
    values : array_like
        Support of the distribution.
    weights : array_like
        Probability mass for each value (need not be normalised).
    size : int
        Number of samples to draw.
    seed : int or None
        Optional seed.

    Returns
    -------
    numpy.ndarray
        Array of sampled values.
    """
    raise NotImplementedError
