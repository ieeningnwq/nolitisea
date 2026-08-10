"""Box-assisted nearest-neighbour search.

Replaces ``make_box`` / ``find_neighbors`` / ``make_multi_box`` /
``find_multi_neighbors`` from the TISEAN routines.
"""

import numpy as np


def make_box(series, dim, delay, eps, box_size=256):
    """Build a box-hash index over an embedded scalar series.

    Parameters
    ----------
    series : numpy.ndarray
        One-dimensional scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    eps : float
        Box side length (also the neighbour radius).
    box_size : int, default 256
        Number of boxes per axis (power of two recommended).

    Returns
    -------
    tuple
        ``(box, lst)`` where ``box`` maps each cell to the first point
        and ``lst`` chains the remaining points.
    """
    raise NotImplementedError


def find_neighbors(series, point, eps, dim, delay, box, box_size=256):
    """Return indices of points within ``eps`` of ``point``.

    Parameters
    ----------
    series : numpy.ndarray
        Original scalar series.
    point : numpy.ndarray
        Query point in delay-coordinate space.
    eps : float
        Neighbourhood radius.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    box : tuple
        Box index returned by :func:`make_box`.
    box_size : int, default 256
        Boxes per axis.

    Returns
    -------
    numpy.ndarray
        Indices of neighbouring points.
    """
    raise NotImplementedError


def find_neighbors_kdtree(points, query, eps):
    """Alternative neighbour search using ``scipy.spatial.cKDTree``.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape ``(n, d)`` of reference points.
    query : numpy.ndarray
        Single point of shape ``(d,)``.
    eps : float
        Search radius.

    Returns
    -------
    numpy.ndarray
        Indices of points within ``eps`` of ``query``.
    """
    raise NotImplementedError


def find_multi_neighbors(series_list, point_list, eps, dims, delays, box):
    """Neighbour search for a multivariate mixed embedding.

    Parameters
    ----------
    series_list : sequence of numpy.ndarray
        One array per variable.
    point_list : sequence of numpy.ndarray
        Query point per variable.
    eps : float
        Neighbourhood radius.
    dims : sequence of int
        Embedding dimension per variable.
    delays : sequence of int
        Delay per variable.
    box : tuple
        Box index returned by :func:`make_multi_box`.

    Returns
    -------
    numpy.ndarray
        Indices of neighbouring points.
    """
    raise NotImplementedError


def make_multi_box(series_list, dims, delays, eps, box_size=256):
    """Build a box index for a multivariate embedding.

    Parameters
    ----------
    series_list : sequence of numpy.ndarray
        One array per variable.
    dims : sequence of int
        Embedding dimension per variable.
    delays : sequence of int
        Delay per variable.
    eps : float
        Box side length.
    box_size : int, default 256
        Boxes per axis.

    Returns
    -------
    tuple
        ``(box, lst)`` multi-dimensional box index.
    """
    raise NotImplementedError
