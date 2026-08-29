"""Recurrence plot."""

import numpy as np
from scipy.spatial.distance import cdist

from nolitisea.core.embed import delay_embedding


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
    distance_matrix = cdist(embedded_vectors, embedded_vectors, metric=metric) # type: ignore

    # Apply the Heaviside step function threshold
    # Returns a boolean matrix where True represents a recurrence
    recurrence_mask = distance_matrix < eps

    return recurrence_mask


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
