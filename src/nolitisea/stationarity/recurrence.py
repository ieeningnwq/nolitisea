"""Recurrence plot."""

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]
from scipy.spatial.distance import cdist

from nolitisea.core.embed import delay_embedding, lag_block_delay_embed


# ==========================================
# 1. Recurrence Matrix Generation
# ==========================================
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
    distance_matrix = cdist(embedded_vectors, embedded_vectors, metric=metric)  # type: ignore

    # Apply the Heaviside step function threshold
    # Returns a boolean matrix where True represents a recurrence
    recurrence_mask = distance_matrix < eps

    return recurrence_mask


def recurrence_matrix_fixed_rr(series, dim, delay, target_rr, metric="euclidean"):
    """Build a binary recurrence matrix at an approximately fixed recurrence rate (RR).

    Instead of providing a specific distance threshold, this function chooses
    the threshold from the order statistics of the observed pairwise distances
    (excluding the main diagonal) so that the closest attainable proportion of
    recurrences is retained.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    target_rr : float
        The desired Recurrence Rate, expressed as a fraction between 0.0 and 1.0
        (e.g., 0.05 for 5% RR).
    metric : str, default "euclidean"
        Distance metric name.

    Returns
    -------
    recurrence_mask : numpy.ndarray
        Boolean matrix of shape ``(n_points, n_points)`` representing the recurrence plot.
    eps : float
        The automatically selected distance threshold (an observed off-diagonal
        distance).  ``float('inf')`` is returned when ``target_rr == 1.0`` so
        that every pair is retained.

    Raises
    ------
    ValueError
        If ``target_rr`` is not in the closed interval ``[0.0, 1.0]``.
    RuntimeError
        If all embedded vectors coincide (all pairwise distances are zero),
        e.g. for a constant input series: no finite threshold can produce a
        positive RR in a degenerate single-point state space.

    Notes
    -----
    With unique (continuous) distances the realised off-diagonal RR differs
    from ``target_rr`` by at most half a granularity step
    (``1 / n_points / (n_points - 1)``).  When distances are tied (quantised
    data) all pairs sharing a boundary distance are either kept or excluded
    together, so the realised RR is the closest value allowed by the
    threshold grid and can deviate further.
    """
    if not (0.0 <= float(target_rr) <= 1.0):
        raise ValueError(f"target_rr must be in [0.0, 1.0], got {target_rr}")

    data = np.asarray(series)

    # Phase space reconstruction (Time Delay Embedding)
    embedded_vectors = delay_embedding(data, dim, delay)

    # Calculate the pairwise distance matrix efficiently using cdist
    distance_matrix = cdist(embedded_vectors, embedded_vectors, metric=metric)  # pyright: ignore[reportCallIssue, reportArgumentType]

    n_points = distance_matrix.shape[0]

    if n_points <= 1:
        # Cannot calculate off-diagonal order statistics for a 1x1 or empty matrix
        return distance_matrix < np.inf, 0.0

    # Look at pairwise distances excluding the main diagonal (Line of Identity).
    # np.triu_indices(n, k=1) gives the indices for the upper triangle, omitting the main diagonal.
    upper_tri_indices = np.triu_indices(n_points, k=1)
    off_diagonal_distances = distance_matrix[upper_tri_indices]

    # A degenerate state space (e.g. a constant series) has no positive
    # distances: every finite threshold gives RR = 0, so raise instead of
    # silently returning an empty plot.
    if not np.any(off_diagonal_distances > 0.0):
        raise RuntimeError(
            "cannot fix recurrence rate: all embedded vectors coincide "
            "(zero pairwise distances), e.g. a constant input series"
        )

    n_pairs = off_diagonal_distances.size
    sorted_distances = np.sort(off_diagonal_distances)

    # Number of off-diagonal pairs to retain, rounded to the nearest integer.
    # The threshold is an observed order statistic rather than an interpolated
    # quantile, so the result is reproducible and respects tied distances.
    k = round(float(target_rr) * n_pairs)
    if k <= 0:
        # Strictly below the smallest observed distance: no pair is retained.
        eps = float(sorted_distances[0])
    elif k >= n_pairs:
        # Keep every pair regardless of its distance.
        eps = float("inf")
    else:
        # k-th order statistic: with unique distances exactly k pairs are
        # strictly smaller; pairs tied with this boundary are strictly excluded,
        # which may result in keeping slightly fewer pairs than expected.
        eps = float(sorted_distances[k])

    # Apply the Heaviside step function threshold using the selected eps
    recurrence_mask = distance_matrix < eps

    return recurrence_mask, eps


# ==========================================
# 2. Line Extraction Utilities
# ==========================================


def _get_line_lengths(bool_1d_array):
    """
    Helper function to find lengths of consecutive True values (Run-Length Encoding).
    """
    padded = np.concatenate(([False], bool_1d_array, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return ends - starts


def extract_diagonal_lengths(rmat, l_min=2):
    """
    Extract lengths of all diagonal lines in the recurrence matrix (excluding LOI).
    """
    n_points = rmat.shape[0]
    lengths = []

    # Iterate through all possible diagonals from bottom-left to top-right
    # k < 0: lower triangle
    # k > 0: upper triangle
    # k = 0: main diagonal (Line of Identity)
    for k in range(-n_points + 1, n_points):
        if k == 0:
            continue  # Explicitly skip the main diagonal

        diag_arr = np.diag(rmat, k=k)
        lengths.extend(_get_line_lengths(diag_arr))

    lengths = np.array(lengths)
    return lengths[lengths >= l_min]


def extract_vertical_lengths(rmat, v_min=2):
    """
    Extract lengths of all vertical lines in the recurrence matrix.
    """
    n_points = rmat.shape[0]
    lengths = []

    for col_idx in range(n_points):
        col_arr = rmat[:, col_idx]
        lengths.extend(_get_line_lengths(col_arr))

    lengths = np.array(lengths)
    return lengths[lengths >= v_min]


# ==========================================
# 3. Independent RQA Metric Functions
# ==========================================


def recurrence_rate(rmat):
    """Calculate the Recurrence Rate (RR) from a recurrence matrix.

    The Recurrence Rate is defined as the percentage of recurrence points
    in a Recurrence Plot, excluding the main diagonal (Line of Identity).
    The main diagonal is explicitly set to False to guarantee accuracy,
    avoiding errors if the diagonal was artificially modified prior to calculation.

    Reference
    ---------
    Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007).
    Recurrence plots for the analysis of complex systems.
    Physics Reports, 438(5-6), 237-329.
    DOI: 10.1016/j.physrep.2006.11.001

    Parameters
    ----------
    rmat : numpy.ndarray
        A square boolean recurrence matrix.

    Returns
    -------
    float
        The computed Recurrence Rate (RR), bounded between 0.0 and 1.0.
    """
    # Create a boolean copy to avoid modifying the original input matrix (side effects)
    rmat_copy = np.array(rmat, dtype=bool, copy=True)
    n_points = rmat_copy.shape[0]

    # If matrix is 1x1 or empty, RR is mathematically undefined or zero
    if n_points <= 1:
        return 0.0

    # Explicitly force the main diagonal to False
    np.fill_diagonal(rmat_copy, False)

    # Sum all recurrence points (True values) in the matrix without the diagonal
    recurrences_no_diagonal = np.sum(rmat_copy)

    # Total possible pairs excluding the main diagonal
    total_possible_pairs = n_points * (n_points - 1)

    # Calculate Recurrence Rate
    rr = recurrences_no_diagonal / total_possible_pairs

    return float(rr)


def determinism(rmat, l_min=2):
    """Calculate Determinism (DET)."""
    rmat_copy = np.array(rmat, dtype=bool, copy=True)
    np.fill_diagonal(rmat_copy, False)
    total_recurrences = np.sum(rmat_copy)

    if total_recurrences == 0:
        return 0.0

    diag_lengths = extract_diagonal_lengths(rmat, l_min)
    return float(np.sum(diag_lengths) / total_recurrences)


def average_diagonal_length(rmat, l_min=2):
    """Calculate the Average Diagonal Length (L_avg)."""
    diag_lengths = extract_diagonal_lengths(rmat, l_min)
    if len(diag_lengths) == 0:
        return 0.0
    return float(np.mean(diag_lengths))


def max_diagonal_length(rmat, l_min=2):
    """Calculate the Maximal Diagonal Length (L_max)."""
    diag_lengths = extract_diagonal_lengths(rmat, l_min)
    if len(diag_lengths) == 0:
        return 0
    return int(np.max(diag_lengths))


def entropy_diagonal_lines(rmat, l_min=2):
    """Calculate the Shannon Entropy of diagonal line lengths (ENTR)."""
    diag_lengths = extract_diagonal_lengths(rmat, l_min)
    if len(diag_lengths) == 0:
        return 0.0

    counts = np.bincount(diag_lengths)[l_min:]
    probabilities = counts[counts > 0] / np.sum(counts)
    entr = -np.sum(probabilities * np.log(probabilities))
    return float(entr)


def laminarity(rmat, v_min=2):
    """Calculate Laminarity (LAM)."""
    rmat_copy = np.array(rmat, dtype=bool, copy=True)
    np.fill_diagonal(rmat_copy, False)
    total_recurrences = np.sum(rmat_copy)

    if total_recurrences == 0:
        return 0.0

    vert_lengths = extract_vertical_lengths(rmat, v_min)
    return float(np.sum(vert_lengths) / total_recurrences)


def trapping_time(rmat, v_min=2):
    """Calculate Trapping Time (TT)."""
    vert_lengths = extract_vertical_lengths(rmat, v_min)
    if len(vert_lengths) == 0:
        return 0.0
    return float(np.mean(vert_lengths))


def max_vertical_length(rmat, v_min=2):
    """Calculate the Maximal Vertical Length (V_max)."""
    vert_lengths = extract_vertical_lengths(rmat, v_min)
    if len(vert_lengths) == 0:
        return 0
    return int(np.max(vert_lengths))


# ==========================================
# 4. ``recurr``
# ==========================================


def recurr(series, embed=2, delay=1, eps=None, fraction=1.0, seed=0):
    """Recurrence plot via a Chebyshev neighbour search.

    Each component is rescaled to ``[0, 1]`` independently, the multivariate delay embedding is
    built, and every pair of embedding vectors whose Chebyshev
    distance is below ``eps`` is reported as a recurrence.  The
    neighbour search uses a single
    :func:`scipy.spatial.cKDTree.query_ball_tree` call.

    Parameters
    ----------
    series : array_like
        Input series.  A 1-D array is treated as a single component;
        a 2-D array must have shape ``(n_times, n_vars)``.
    embed : int, default 2
        Embedding dimension per component;
        ``DIM`` is inferred from the number of columns).
    delay : int, default 1
        Time delay between consecutive embedding coordinates.
    eps : float or None
        Recurrence threshold in the units of the input data.  ``None``
        (default) uses the data interval divided by 1000.  The
        threshold is expressed in the rescaled ``[0, 1]`` units: a
        user-supplied ``eps`` is divided by the largest component
        range.
    fraction : float, default 1.0
        Fraction of eligible recurrence pairs to keep.  ``1.0`` keeps every pair (deterministic); smaller
        values subsample uniformly with the seeded generator, so the
        exact pairs kept for ``fraction < 1`` are
        implementation-specific; use ``fraction = 1.0`` for a
        reproducible comparison.
    seed : int, default 0
        Seed for the subsampling generator (only used when
        ``fraction < 1``).

    Returns
    -------
    dict
        ``"pairs"`` : ``(N, 2)`` int array of embedding row indices
        ``(i, j)`` with ``i < j`` (upper triangle of the recurrence
        matrix); the matrix is symmetric, so the full plot is obtained
        by mirroring.
        ``"eps"`` : the threshold in rescaled units.
        ``"n_points"`` : the number of embedding vectors.

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the requested
        ``embed``/``delay``.
    RuntimeError
        If any component is constant (zero value range).

    Notes
    -----
    ``cKDTree`` uses a closed ball (``<= eps``).  On
    continuous data the closed-ball threshold coincides with a strict
    comparison; on quantised data the boundary pairs may
    differ, consistent with the convention used
    throughout this package.

    References
    ----------
    .. [1] Eckmann, J.-P., Kamphorst, S. O., & Ruelle, D. (1987).
           Recurrence plots of dynamical systems.  *Europhysics
           Letters*, 4(9), 973-977.
    .. [2] Hegger, R., Kantz, H., & Schreiber, T. (1999). Practical
           implementation of nonlinear time series methods: The TISEAN
           package.  *Chaos*, 9(2), 413-435.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(
            "series must be a 1-D array or a 2-D array of shape (n_times, n_vars)"
        )
    # Copy so the rescaling below does not modify the caller's array.
    arr = arr.copy()
    _, n_vars = arr.shape

    if embed < 1:
        raise ValueError(f"embed must be >= 1, got {embed}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    # Rescale every component to [0, 1]; maxmax is the largest original
    # range, used to express a user threshold in rescaled units.
    maxmax = 0.0
    for c in range(n_vars):
        comp = arr[:, c]
        rng = np.ptp(comp)
        if rng < 1e-30:
            raise RuntimeError(f"component {c} is constant (zero range)")
        arr[:, c] = (comp - comp.min()) / rng
        maxmax = max(maxmax, rng)

    eps_rescaled = 1.0e-3 if eps is None else abs(float(eps)) / maxmax

    # Interleaved multivariate delay embedding (row r <-> base time
    # r + (embed-1)*delay); the Chebyshev distance is identical.
    E = lag_block_delay_embed(arr, embed, delay)
    n_points = E.shape[0]

    tree = cKDTree(E)
    # query_ball_tree returns, for every point, every neighbour within
    # eps_rescaled (Chebyshev); keep the upper triangle so each pair is
    # reported once.
    nb_lists = tree.query_ball_tree(tree, eps_rescaled, p=np.inf)
    i_list = []
    j_list = []
    for i, nb in enumerate(nb_lists):
        if len(nb) == 0:
            continue
        nb = np.asarray(nb)
        sel = nb > i
        if sel.any():
            i_list.append(np.full(int(sel.sum()), i))
            j_list.append(nb[sel])
    if i_list:
        pairs = np.column_stack([np.concatenate(i_list), np.concatenate(j_list)])
    else:
        pairs = np.empty((0, 2), dtype=np.intp)

    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        keep = rng.random(pairs.shape[0]) < fraction
        pairs = pairs[keep]

    return {"pairs": pairs, "eps": eps_rescaled, "n_points": n_points}
