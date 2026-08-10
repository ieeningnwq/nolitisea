"""Select rows and/or columns from a data file (TISEAN ``choose``)."""

import numpy as np


def choose_rows(data, start, end):
    """Return rows ``[start, end)`` of ``data``.

    Parameters
    ----------
    data : numpy.ndarray
        2-D input array.
    start : int
        First row index (inclusive).
    end : int
        Last row index (exclusive).

    Returns
    -------
    numpy.ndarray
        Selected rows.
    """
    raise NotImplementedError


def choose_columns(data, cols):
    """Return the requested columns of ``data``.

    Parameters
    ----------
    data : numpy.ndarray
        2-D input array.
    cols : sequence of int
        Zero-based column indices.

    Returns
    -------
    numpy.ndarray
        Selected columns.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: select rows/columns.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
