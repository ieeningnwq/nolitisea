"""Multivariate box partitioning primitives shared by the dimension and
entropy estimators."""

import numpy as np


def assign_boxes(points, eps):
    """Map each point to integer box coordinates.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape ``(n, d)``.
    eps : float
        Box side length.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n, d)`` of box coordinates.
    """
    raise NotImplementedError


def box_neighbor_offsets(dim):
    """Generate the ``3**dim`` neighbour offsets of a box.

    Parameters
    ----------
    dim : int
        Number of dimensions.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3**dim, dim)`` with offsets in ``{-1,0,1}``.
    """
    raise NotImplementedError
