"""nearest-neighbour search."""

import numpy as np
from scipy.spatial import cKDTree  # type: ignore


def find_neighbors(y, metric="chebyshev", theiler=0, maxnum=None):
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
    if metric == "cityblock":
        p = 1
    elif metric == "euclidean":
        p = 2
    elif metric == "chebyshev":
        p = np.inf
    else:
        raise ValueError(
            'Unknown metric.  Should be one of "cityblock", '
            '"euclidean", or "chebyshev".'
        )

    tree = cKDTree(y)
    n = len(y)

    if maxnum is None:
        # Default: enough candidates to absorb the self-match (1) plus the
        # Theiler window (theiler) while still leaving one non-self,
        # non-Theiler neighbour, with a small buffer.  Mirrors TISEAN's
        # heuristic of ``2 * theiler + 3``.
        maxnum = 2 * (theiler + 1) + 1
    elif maxnum < theiler + 2:
        raise ValueError(
            f"maxnum must be >= theiler + 2 = {theiler + 2} (k-nearest "
            f"query includes the self-match; the Theiler window blocks "
            f"theiler more neighbours; need at least one surviving "
            f"neighbour), got {maxnum}"
        )

    if maxnum >= n:
        raise ValueError(
            f"maxnum={maxnum} must be < array length n={n}"
        )

    # Vectorised batch k-nearest-neighbour query.  Passing the whole
    # ``y`` array at once lets SciPy stream through the tree in C rather
    # than going back into Python for each row — roughly 2× faster than
    # the per-row loop.
    dists_all, indices_all = tree.query(y, k=maxnum, p=p)

    # ``k=1`` collapses the last axis.  Explicitly restore it so that
    # the downstream boolean mask / argmax machinery always sees a 2-D
    # array.
    if maxnum == 1:
        dists_all = dists_all[:, np.newaxis]
        indices_all = indices_all[:, np.newaxis]

    # Valid = inside the eps-ball, outside the Theiler window, non-self.
    row_indices = np.arange(n)[:, np.newaxis]
    valid_mask = (np.abs(indices_all - row_indices) > theiler) & (dists_all > 0)

    has_valid = valid_mask.any(axis=1)
    if not has_valid.all():
        bad_rows = np.where(~has_valid)[0].tolist()[:5]
        raise RuntimeError(
            f"Could not find any non-self, non-Theiler neighbour for "
            f"rows {bad_rows} ... Try increasing maxnum (must be >= "
            f"theiler + 2 = {theiler + 2})."
        )

    # Pick the first (smallest-distance) surviving neighbour per row.
    first_valid_idx = valid_mask.argmax(axis=1)
    row_arr = np.arange(n)
    return indices_all[row_arr, first_valid_idx], dists_all[row_arr, first_valid_idx]