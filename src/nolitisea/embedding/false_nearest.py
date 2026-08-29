"""False nearest neighbours for embedding-dimension selection."""
from __future__ import annotations

import numpy as np
from functools import partial
from scipy.spatial import cKDTree  # type: ignore

from nolitisea.core.embed import delay_embedding
from nolitisea.utils.dist import pairwise_row_distance
from nolitisea.utils.parallel import parallel_map


def _kennel_method(ts, m, delay, R_tol, A_tol, theiler, maxnum, metric):
    """Compute Kennel FNN fractions for a single embedding dimension."""
    valid_len = len(ts) - m * delay
    emb_d_1 = delay_embedding(ts, m, delay)[:valid_len]
    emb_d_2 = delay_embedding(ts, m+1, delay)[:valid_len]

    # Find near neighbors in dimension m.
    index, dist = neighbors(emb_d_1, metric=metric, window=theiler, maxnum=maxnum)

    # Find all potential false neighbors using Kennel et al.'s tests.
    f1 = np.abs(emb_d_2[:, -1] - emb_d_2[index, -1]) / dist > R_tol
    # emb_d[index] is a writable (n, m) copy of each point's neighbor row
    f2 = pairwise_row_distance(emb_d_2, emb_d_2[index], metric=metric) / np.std(ts) > A_tol
    f3 = f1 | f2
    return {'f1': np.mean(f1), 'f2': np.mean(f2), 'f3': np.mean(f3)}


def kennel_method(
    ts: np.ndarray,
    min_emb: int = 1,
    max_emb: int = 5,
    delay: int = 1,
    R_tol: float = 10.0,
    A_tol: float = 2.0,
    theiler: int = 0,
    maxnum: int|None = None,
    metric: str = "chebyshev",
    n_jobs: int|None = None,
    backend: str = "thread",
) -> dict:
    """
    Calculates the fraction of false nearest neighbors (FNN).

    References
        ----------
        .. [1] Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992). Determining
               embedding dimension for phase-space reconstruction using a geometrical
               construction. Physical review A, 45(6), 3403.

    Parameters
    ----------
    ts : np.ndarray
        1-dimensional time series array.
    min_emb : int
        Minimum embedding dimension to test.
    max_emb : int
        Maximum embedding dimension to test.
    delay : int
        Time delay between coordinates.
    R_tol : float, optional (default = 10.0)
        Tolerance parameter for FNN Test I.
    A_tol : float, optional (default = 2.0)
        Tolerance parameter for FNN Test II.
    theiler : int
        Theiler window to exclude temporally correlated neighbors.
    maxnum : int|None
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).
    metric : str
        Distance metric to use: 'chebyshev', 'euclidean', or 'cityblock'.
    n_jobs : int, optional (default = None)
        Workers for the per-dimension loop.  ``None``/``1`` runs
        sequentially; ``-1`` uses all CPUs (see
        ``nolitisea.utils.parallel.parallel_map``).
    backend : {"thread", "process"}, optional (default = "thread")
        Executor backend for the per-dimension loop.

    Returns
    -------
    dict
        Mapping of embedding dimension to the fraction of false nearest neighbors.
    """
    dims = [m for m in range(min_emb, max_emb + 1) if len(ts) - m * delay > 0]

    worker = partial(
        _kennel_method,
        ts,
        delay=delay,
        R_tol=R_tol,
        A_tol=A_tol,
        theiler=theiler,
        maxnum=maxnum,
        metric=metric,
    )
    per_dim = parallel_map(worker, dims, n_jobs=n_jobs, backend=backend)
    return dict(zip(dims, per_dim))

def neighbors(y, metric='chebyshev', window=0, maxnum=None):
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
    window : int, optional (default = 0)
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
        maxnum = (window + 1) + 1 + (window + 1)
    else:
        maxnum = max(1, maxnum)

    if maxnum >= n:
        raise ValueError('maxnum is bigger than array length.')

    dists = np.empty(n)
    indices = np.empty(n, dtype=int)

    for i, x in enumerate(y):
        dist, index = tree.query(x, k=maxnum, p=p)
        valid = (np.abs(index - i) > window) & (dist > 0)

        if not np.count_nonzero(valid):
            raise RuntimeError('Could not find any near neighbor with a '
                            'nonzero distance.  Try increasing the '
                            'value of maxnum.')
        dists[i] = dist[valid][0]
        indices[i] = index[valid][0]
    return np.squeeze(indices), np.squeeze(dists)


def cao_method(
    ts: np.ndarray, max_emb: int = 10, delay: int = 1, metric: str = "chebyshev"
) -> dict:
    """
    Calculates the E(d) and E^*(d) metrics using Cao's method to determine
    the minimum embedding dimension of a scalar time series.

    Parameters
    ----------
    ts : np.ndarray
        1-dimensional time series array.
    max_emb : int
        The maximum embedding dimension to test.
    delay : int
        Time delay between coordinates.
    metric : str
        Distance metric to use: 'chebyshev', 'euclidean', or 'cityblock'.

    Returns
    -------
    dict
        A dictionary containing lists of 'E1' and 'E2' values up to max_emb.

    References
    ----------
    .. [1] Cao, L. (1997). Practical method for determining the minimum
           embedding dimension of a scalar time series. Physica D: Nonlinear
           Phenomena, 110(1-2), 43-50.
    """
    # Ensure contiguous float64 array for strict Numba JIT compatibility
    ts = np.ascontiguousarray(ts, dtype=np.float64)

    e_values = np.zeros(max_emb + 1)
    es_values = np.zeros(max_emb + 1)

    for d in range(1, max_emb + 2):
        n_points = len(ts) - d * delay
        if n_points <= 0:
            break

        # Create phase space for dimension d and enforce memory layout
        emb_d = delay_embedding(ts, d, delay)

        emb_d = np.ascontiguousarray(emb_d, dtype=np.float64)

        a_i = np.zeros(n_points)
        star_i = np.zeros(n_points)

        for i in range(n_points):
            # Broadcast the target point to 2D to match the expected Numba signature
            target_point_array = np.ascontiguousarray(
                np.broadcast_to(emb_d[i], emb_d.shape), dtype=np.float64
            )

            # Calculate distances in dimension d using the Numba dist function
            distances = pairwise_row_distance(emb_d, target_point_array, metric=metric)

            # Exclude self-match
            distances[i] = np.inf
            nn_idx = np.argmin(distances)

            dist_d = distances[nn_idx]

            # Extract the raw d+1 dimensional vectors directly from the time series via slicing
            v_i = ts[i : i + (d + 1) * delay : delay]
            v_nn = ts[nn_idx : nn_idx + (d + 1) * delay : delay]

            # Reshape into 2D arrays (1, d+1) to satisfy the Numba dist function signature
            v_i_2d = np.ascontiguousarray(v_i.reshape(1, -1), dtype=np.float64)
            v_nn_2d = np.ascontiguousarray(v_nn.reshape(1, -1), dtype=np.float64)

            # Calculate distance in dimension d+1 using the Numba dist function
            dist_d_plus_1 = pairwise_row_distance(v_i_2d, v_nn_2d, metric=metric)[0]

            # Extract the absolute difference of just the newly added dimension for E2
            new_comp_diff = np.abs(ts[i + d * delay] - ts[nn_idx + d * delay])

            a_i[i] = dist_d_plus_1 / dist_d if dist_d > 0 else 0
            star_i[i] = new_comp_diff

        e_values[d - 1] = np.mean(a_i)
        es_values[d - 1] = np.mean(star_i)

    # Calculate E1 and E2 metrics safely avoiding division by zero
    e1 = np.zeros(max_emb)
    e2 = np.zeros(max_emb)

    for k in range(max_emb):
        e1[k] = e_values[k + 1] / e_values[k] if e_values[k] > 0 else 0
        e2[k] = es_values[k + 1] / es_values[k] if es_values[k] > 0 else 0

    return {"E1": e1.tolist(), "E2": e2.tolist()}
