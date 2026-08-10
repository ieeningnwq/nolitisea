"""Linear algebra helpers: solve and invert.

Replaces ``solvele`` and ``invert_matrix`` from the TISEAN routines.
"""

import numpy as np


def solve_linear(a, b):
    """Solve the linear system ``A x = b``.

    Parameters
    ----------
    a : numpy.ndarray
        Square coefficient matrix.
    b : numpy.ndarray
        Right-hand side vector or matrix.

    Returns
    -------
    numpy.ndarray
        Solution ``x`` with the same shape as ``b``.
    """
    raise NotImplementedError


def invert_matrix(a):
    """Return the inverse of a square matrix.

    Parameters
    ----------
    a : numpy.ndarray
        Square invertible matrix.

    Returns
    -------
    numpy.ndarray
        Matrix inverse.
    """
    raise NotImplementedError


def pseudo_inverse(a):
    """Return the Moore-Penrose pseudo-inverse of ``a``.

    Used by over-determined polynomial fits.

    Parameters
    ----------
    a : numpy.ndarray
        Rectangular matrix.

    Returns
    -------
    numpy.ndarray
        Pseudo-inverse of ``a``.
    """
    raise NotImplementedError
