"""Simple nonlinear noise reduction."""

import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import delay_embedding


def lazy(x, dim=1, delay=1, r=0, metric='chebyshev', repeat=1):
    """Simple noise reduction based on local phase space averaging.

    Simple noise reduction scheme based on local phase space averaging
    (Schreiber 1993; Kantz & Schreiber 2004).

    Parameters
    ----------
    x : array
        1-D real input array containing the time series.
    dim : int, optional (default = 1)
        Embedding dimension.
    delay : int, optional (default = 1)
        Time delay.
    r : float, optional (default = 0)
        Radius of neighborhood (see Notes).
    metric : string, optional (default = 'chebyshev')
        Metric to use for distance computation.  Must be one of
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".
    repeat: int, optional (default = 1)
        Number of iterations.

    Return
    ------
    y : array
        1-D real output array containing the time series after noise
        reduction.

    Notes
    -----
    Choosing the right neighborhood radius is crucial for proper noise
    reduction.  A large radius will result in too much filtering.  By
    default, a radius of zero is used, which means that no noise
    reduction is done.  Note that the radius also depends on the metric
    used for distance computation.  Best results are often obtained
    using large embedding dimensions with unit delay and the Chebyshev
    metric.
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

    # Choose the middle coordinate appropriately.
    if dim % 2 == 0:
        mid = delay * dim // 2
    else:
        mid = delay * (dim - 1) // 2

    y = np.copy(x)

    for _ in range(repeat):
        z = np.copy(y)
        ps = delay_embedding(y, dim=dim, delay=delay)

        tree = cKDTree(ps)

        # State-space averaging.
        # (We don't use tree.query_ball_tree() as it almost always
        # results in a memory overflow, even though it's faster.)
        for i in range(len(ps)):
            neighbors = tree.query_ball_point(ps[i], r=r, p=p)
            y[i + mid] = np.mean(ps[neighbors][:, mid // delay])

        # Choose the average correction as the new radius.
        r = np.sqrt(np.mean((y - z) ** 2))

        # Stop as soon as the series stops changing.
        if r == 0:
            break

    return y