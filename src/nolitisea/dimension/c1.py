"""Fixed-mass information dimension D1 (TISEAN ``c1``)."""

import numpy as np


def fixed_mass_d1(series, dim, delay, mass_list):
    """Estimate the information dimension by the fixed-mass method.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    mass_list : array_like
        Sequence of fixed masses to use.

    Returns
    -------
    tuple
        ``(mass_list, log_lengths)`` arrays for slope estimation.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: fixed-mass D1 estimation.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
