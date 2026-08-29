"""Box-assisted nearest-neighbour search.

Replaces ``make_box`` / ``find_neighbors`` / ``make_multi_box`` /
``find_multi_neighbors`` from the TISEAN routines.
"""

import numpy as np
from scipy.spatial import cKDTree  # type: ignore


def find_neighbors(y, metric='chebyshev', theiler=0, maxnum=None):
    """Find nearest neighbors of all points in the given array.

    Finds the nearest neighbors of all points in the given array using
    SciPy's KDTree search.

    Parameters
    ----------
    y : ndarray
        N-dimensional array containing time-delayed vectors.
    metric : string, optional (default = 'chebyshev')
        Metric to use for distance computation.  Must be one of
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".
    theiler : int, optional (default = 0)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.  This is crucial while computing
        Lyapunov exponents and the correlation dimension.
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).

    Returns
    -------
    index : array
        Array containing indices of near neighbors.
    dist : array
        Array containing near neighbor distances.
    """
    if metric == 'cityblock':
        p = 1
    elif metric == 'euclidean':
        p = 2
    elif metric == 'chebyshev':
        p = np.inf
    else:
        raise ValueError('Unknown metric.  Should be one of "cityblock", '
                         '"euclidean", or "chebyshev".')

    tree = cKDTree(y)
    n = len(y)

    if not maxnum:
        maxnum = (theiler + 1) + 1 + (theiler + 1)
    else:
        maxnum = max(1, maxnum)

    if maxnum >= n:
        raise ValueError('maxnum is bigger than array length.')

    dists = np.empty(n)
    indices = np.empty(n, dtype=int)

    for i, x in enumerate(y):
        dist, index = tree.query(x, k=maxnum, p=p)
        valid = (np.abs(index - i) > theiler) & (dist > 0)

        if not np.count_nonzero(valid):
            raise RuntimeError('Could not find any near neighbor with a '
                            'nonzero distance.  Try increasing the '
                            'value of maxnum.')
        dists[i] = dist[valid][0]
        indices[i] = index[valid][0]
    return np.squeeze(indices), np.squeeze(dists)



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