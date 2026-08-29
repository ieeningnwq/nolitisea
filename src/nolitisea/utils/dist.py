import numpy as np
from numba import jit


def pairwise_row_distance(x, y, metric="chebyshev"):
    """Compute the distance between all sequential pairs of points.

    Computes the distance between all sequential pairs of points from
    two arrays using scipy.spatial.distance.

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


@jit("float64[:](float64[:, :], float64[:, :])", nopython=True, nogil=True)
def cityblock_pairwise_row_distance(x, y):
    n = x.shape[0]
    d = x.shape[1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(d):
            diff = x[i, j] - y[i, j]
            s += diff if diff >= 0.0 else -diff
        out[i] = s
    return out


@jit("float64[:](float64[:, :], float64[:, :])", nopython=True, nogil=True)
def euclidean_pairwise_row_distance(x, y):
    n = x.shape[0]
    d = x.shape[1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(d):
            diff = x[i, j] - y[i, j]
            s += diff * diff
        out[i] = np.sqrt(s)
    return out


@jit("float64[:](float64[:, :], float64[:, :])", nopython=True, nogil=True)
def chebyshev_pairwise_row_distance(x, y):
    n = x.shape[0]
    d = x.shape[1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        m = 0.0
        for j in range(d):
            diff = x[i, j] - y[i, j]
            a = diff if diff >= 0.0 else -diff
            m = max(m, a)
        out[i] = m
    return out
