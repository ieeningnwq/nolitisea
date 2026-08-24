"""Recurrence plot."""

import numpy as np
from scipy.spatial.distance import cdist

from nolitisea.core.embed import delay_embedding


def recurrence_matrix(series, dim, delay, eps, metric="euclidean"):
    """Build a binary recurrence matrix.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    eps : float
        Recurrence threshold.
    metric : str, default "euclidean"
        Distance metric name.

    Returns
    -------
    numpy.ndarray
        Boolean matrix of shape ``(n_points, n_points)`` where
        ``R[i, j] = ||x_i - x_j|| < eps``.
    """
    data = np.asarray(series)
    
    # Phase space reconstruction (Time Delay Embedding)
    # Construct state vectors by taking 'dim' elements separated by 'delay'
    embedded_vectors = delay_embedding(data, dim, delay)
    
    # Calculate the pairwise distance matrix
    # cdist computes the distance between every pair of vectors efficiently
    distance_matrix = cdist(embedded_vectors, embedded_vectors, metric=metric)
    
    # Apply the Heaviside step function threshold
    # Returns a boolean matrix where True represents a recurrence
    recurrence_mask = distance_matrix <= eps
    
    return recurrence_mask


def recurrence_rate(rmat):
    """Return the recurrence rate of a recurrence matrix.

    Parameters
    ----------
    rmat : numpy.ndarray
        Boolean recurrence matrix.

    Returns
    -------
    float
        Fraction of recurring points.
    """
    raise NotImplementedError
