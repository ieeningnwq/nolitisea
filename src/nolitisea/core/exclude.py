"""Interval exclusion helper.

Replaces ``exclude_interval`` from the TISEAN routines.
"""

import numpy as np


def exclude_interval(length, start, end):
    """Return indices of ``range(length)`` excluding ``[start, end]``.

    Parameters
    ----------
    length : int
        Total number of points.
    start : int
        First index to exclude.
    end : int
        Last index to exclude (inclusive).

    Returns
    -------
    numpy.ndarray
        Array of kept indices.
    """
    raise NotImplementedError
