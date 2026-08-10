"""Locate extrema of a series (TISEAN ``extrema``)."""

import numpy as np


def extrema(series, mode="max"):
    """Return the indices and values of local extrema.

    Parameters
    ----------
    series : array_like
        Input series.
    mode : str, default "max"
        ``"max"``, ``"min"`` or ``"both"``.

    Returns
    -------
    tuple
        ``(indices, values)`` arrays.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: extract extrema.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
