import numpy as np


def pairwise_row_distance(x, y, metric="chebyshev"):
    """Compute the distance between all sequential pairs of points.

    Computes the distance between all sequential pairs of points from
    two arrays.

    Paramters
    ---------
    x : ndarray
        Input array.
    y : ndarray
        Input array.
    metric : string, optional (default = 'chebyshev')
        Metric to use while computing distances.

    Returns
    -------
    d : ndarray
        Array containing distances.
    """
    if metric == "cityblock":
        func = cityblock_pairwise_row_distance
    elif metric == "euclidean":
        func = euclidean_pairwise_row_distance
    elif metric == "chebyshev":
        func = chebyshev_pairwise_row_distance
    else:
        raise ValueError(
            'Unknown metric.  Should be one of "cityblock", '
            '"euclidean", or "chebyshev".'
        )

    return func(x, y)


def cityblock_pairwise_row_distance(x, y):
    return np.abs(x - y).sum(axis=1)


def euclidean_pairwise_row_distance(x, y):
    return np.sqrt(((x - y) ** 2).sum(axis=1))


def chebyshev_pairwise_row_distance(x, y):
    return np.abs(x - y).max(axis=1)
