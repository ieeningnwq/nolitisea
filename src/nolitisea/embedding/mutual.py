"""Mutual information for delay selection."""

import warnings

import numpy as np


def mutual_information(
    x: np.ndarray, y: np.ndarray, bins: int = 10, normalize: bool = False
) -> float:
    """
    Calculate the Mutual Information (MI) between two 1D arrays based on Shannon entropy.

    This function uses a 2D histogram approach to estimate the joint probability
    distribution P(X, Y) and the marginal distributions P(X) and P(Y).

    Parameters:
    -----------
    x : np.ndarray
        The first 1D data array.
    y : np.ndarray
        The second 1D data array. Must be of the same length as x.
    bins : int, optional
        The number of bins to use for the histogram discretization (default is 10).
        For shorter sequences, the number of bins should not be excessively large;
        for longer sequences, it can be dynamically determined using Sturges' formula or the Freedman-Diaconis rule.
    normalize : bool, optional
        If True, returns the Normalized Mutual Information (NMI) scaled between [0, 1]
        using the formula: NMI = 2 * I(X;Y) / (H(X) + H(Y)). Default is False.

    Returns:
    --------
    mi : float
        The estimated mutual information in bits (base 2 logarithm).
    """
    # Ensure inputs are 1D numpy arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Check for dimensional consistency
    if len(x) != len(y):
        raise ValueError("The input arrays x and y must have the same length.")

    if len(x) == 0:
        return 0.0

    # 1. Compute the 2D histogram to get the joint frequency distribution
    # c_xy is a 2D matrix of shape (bins, bins) containing the bin counts
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)

    # 2. Convert frequencies to joint probabilities P(X, Y)
    total_samples = np.sum(c_xy)
    p_xy = c_xy / total_samples

    # 3. Compute marginal probabilities P(X) and P(Y)
    p_x = np.sum(p_xy, axis=1)  # Sum over Y axis
    p_y = np.sum(p_xy, axis=0)  # Sum over X axis

    # 4. Filter out zero probabilities to avoid log(0) errors in entropy calculation
    # According to information theory, limit (p*log(p)) as p->0 is 0.
    non_zero_indices = p_xy > 0
    p_xy_nz = p_xy[non_zero_indices]

    # Compute the outer product of P(X) and P(Y) to get independent joint probabilities
    p_x_p_y = np.outer(p_x, p_y)
    p_x_p_y_nz = p_x_p_y[non_zero_indices]

    # 5. Calculate Mutual Information: I(X;Y) = sum( P(x,y) * log2( P(x,y) / (P(x)*P(y)) ) )
    # Suppress divide by zero warnings just in case, though non_zero_indices handles it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = np.sum(p_xy_nz * np.log2(p_xy_nz / p_x_p_y_nz))

    # 6. Apply normalization if requested
    if normalize:
        # Calculate marginal entropies H(X) and H(Y)
        p_x_nz = p_x[p_x > 0]
        h_x = -np.sum(p_x_nz * np.log2(p_x_nz))

        p_y_nz = p_y[p_y > 0]
        h_y = -np.sum(p_y_nz * np.log2(p_y_nz))

        entropy_sum = h_x + h_y

        if entropy_sum == 0:
            return 0.0

        # NMI formulation: 2 * I(X;Y) / (H(X) + H(Y))
        mi_normalized = 2.0 * mi / entropy_sum
        return float(mi_normalized)

    return float(mi)


def embedded_mutual_information(series, max_lag, n_bins=16):
    """Estimate the mutual information for lags ``0..max_lag``.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    max_lag : int
        Maximum lag.
    n_bins : int, default 16
        Number of histogram bins for the estimation.

    Returns
    -------
    numpy.ndarray
        Mutual information values of length ``max_lag + 1``.
    """
    raise NotImplementedError


def first_minimum(series, max_lag, n_bins=16):
    """Return the lag of the first local minimum of the mutual information.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    max_lag : int
        Maximum lag to scan.
    n_bins : int, default 16
        Number of histogram bins.

    Returns
    -------
    int
        Estimated delay.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: estimate the mutual information.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
