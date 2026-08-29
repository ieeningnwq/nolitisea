"""False nearest neighbours for embedding-dimension selection."""
from __future__ import annotations

from functools import partial

import numpy as np

from nolitisea.core.embed import delay_embedding
from nolitisea.core.neighbors import find_neighbors
from nolitisea.utils.dist import pairwise_row_distance
from nolitisea.utils.parallel import parallel_map
from typing import Optional


def _kennel_method(ts, m, delay, R_tol, A_tol, theiler, maxnum, metric):
    """Compute Kennel FNN fractions for a single embedding dimension."""
    valid_len = len(ts) - m * delay
    emb_d_1 = delay_embedding(ts, m, delay)[:valid_len]
    emb_d_2 = delay_embedding(ts, m+1, delay)[:valid_len]

    # Find near neighbors in dimension m.
    index, dist = find_neighbors(emb_d_1, metric=metric, theiler=theiler, maxnum=maxnum)

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


def _cao_method(ts, m, delay, theiler, maxnum, metric):
    """Return E(d) and E^*(d) for a single d.

    Returns E(d) and E^*(d) for the Cao's method for a single d.  This
    function is meant to be called from the main cao_method() function.  See
    the docstring of cao_method( for more.)
    """
    valid_len = len(ts) - m * delay
    emb_d_1 = delay_embedding(ts, m, delay)[:valid_len]
    emb_d_2 = delay_embedding(ts, m+1, delay)[:valid_len]

    # Find near neighbors in dimension m.
    index, dist = find_neighbors(emb_d_1, metric=metric, theiler=theiler, maxnum=maxnum)
    # Compute the magnification and the increase in the near-neighbor
    # distances and return the averages.
    E = pairwise_row_distance(emb_d_2, emb_d_2[index], metric=metric) / dist
    Es = np.abs(emb_d_2[:, -1] - emb_d_2[index, -1])
    return np.mean(E), np.mean(Es)

def cao_method(
    ts: np.ndarray, min_emb: int = 1, max_emb: int = 10, delay: int = 1, metric: str = "chebyshev",
    theiler: int = 0, maxnum: int|None = None, n_jobs: int|None = None, backend: str = "thread"
) -> dict:
    """
    Calculates the E(d) and E^*(d) metrics using Cao's method to determine
    the minimum embedding dimension of a scalar time series.If to calculate E1 and E2, use
    E1 = E[1:] / E[:-1]
    E2 = Es[1:] / Es[:-1]

    References
    ----------
    .. [1] Cao, L. (1997). Practical method for determining the minimum
           embedding dimension of a scalar time series. Physica D: Nonlinear
           Phenomena, 110(1-2), 43-50.

    Parameters
    ----------
    ts : np.ndarray
        1-dimensional time series array.
    min_emb : int
        The minimum embedding dimension to test.
    max_emb : int
        The maximum embedding dimension to test.
    delay : int
        Time delay between coordinates.
    metric : str
        Distance metric to use: 'chebyshev', 'euclidean', or 'cityblock'.
    theiler : int
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.  This is crucial while computing
    maxnum : int
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).
    n_jobs : int
        Number of jobs to run in parallel.  None means to use all available
        cores.
    backend : str
        Backend to use for parallelization.  "thread" means to use threads.
        "process" means to use processes.

    Returns
    -------
    dict
        A dictionary containing lists of 'E1' and 'E2' values up to max_emb.
    """
    dims = [m for m in range(min_emb, max_emb + 1) if len(ts) - m * delay > 0]
    worker = partial(
        _cao_method,
        ts,
        delay=delay,
        theiler=theiler,
        maxnum=maxnum,
        metric=metric,
    )
    per_dim = parallel_map(worker, dims, n_jobs=n_jobs, backend=backend)
    return dict(zip(dims, per_dim))