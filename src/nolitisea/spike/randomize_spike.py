"""Surrogate spike trains preserving autocorrelation or spectrum.

Replaces TISEAN ``randomize_spikeauto_exp_random`` and
``randomize_spikespec_exp_event``.
"""

import numpy as np


def randomize_spikeauto(event_times, n_iter=1000, seed=None):
    """Generate surrogates preserving the event autocorrelation.

    Parameters
    ----------
    event_times : array_like
        Sorted event times.
    n_iter : int, default 1000
        Number of annealing iterations.
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    numpy.ndarray
        Surrogate event times.
    """
    raise NotImplementedError


def randomize_spikespec(event_times, n_iter=1000, seed=None):
    """Generate surrogates preserving the event power spectrum.

    Parameters
    ----------
    event_times : array_like
        Sorted event times.
    n_iter : int, default 1000
        Number of annealing iterations.
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    numpy.ndarray
        Surrogate event times.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: spike-train surrogate generation.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
