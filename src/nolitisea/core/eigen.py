"""Symmetric eigenvalue solvers used by PCA and the Lyapunov spectrum.

Replaces ``eigen.c`` from the TISEAN routines.
"""

import numpy as np


def eig_sym(a):
    """Return eigenvalues and eigenvectors of a symmetric matrix.

    Parameters
    ----------
    a : numpy.ndarray
        Real symmetric matrix.

    Returns
    -------
    tuple
        ``(values, vectors)`` with values in ascending order and
        ``vectors[:, i]`` corresponding to ``values[i]``.
    """
    raise NotImplementedError


def jacobi_eigen(a, max_iter=100):
    """Compute eigenvalues with the Jacobi rotation method.

    Provided as a pedagogical fallback to :func:`eig_sym`.

    Parameters
    ----------
    a : numpy.ndarray
        Real symmetric matrix.
    max_iter : int, default 100
        Maximum number of sweeps.

    Returns
    -------
    tuple
        ``(values, vectors)``.
    """
    raise NotImplementedError
