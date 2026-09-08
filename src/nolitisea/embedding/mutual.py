"""Mutual information for delay selection."""

import warnings

import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial.distance import pdist, squareform


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

    For every lag ``tau`` in ``[0, max_lag]`` the mutual information
    between ``series[:-tau]`` and ``series[tau:]`` is computed via
    :func:`mutual_information` using an equi-width histogram with
    ``n_bins`` bins.  The result is the classic delay-selection curve
    of Fraser & Swinney (1986).

    Parameters
    ----------
    series : array_like
        Input scalar series.
    max_lag : int
        Maximum lag to evaluate (inclusive).  Must satisfy
        ``max_lag < len(series)``.
    n_bins : int, default 16
        Number of histogram bins for the joint distribution.

    Returns
    -------
    numpy.ndarray
        Mutual information values of length ``max_lag + 1``, with
        ``mi[tau] = I(x(t); x(t + tau))``.  ``mi[0]`` equals the
        marginal entropy ``H(x)``.

    Raises
    ------
    ValueError
        If ``series`` is not 1-D, if ``max_lag`` is negative or not
        smaller than the series length, or if ``n_bins`` is less than 1.
    """
    arr = np.asarray(series)
    if arr.ndim != 1:
        raise ValueError(
            f"series must be a 1-D array, got shape {arr.shape}"
        )
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    if max_lag >= arr.size:
        raise ValueError(
            f"max_lag ({max_lag}) must be smaller than the series "
            f"length ({arr.size})"
        )
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    mi = np.empty(max_lag + 1, dtype=np.float64)
    mi[0] = mutual_information(arr, arr, bins=n_bins)
    for tau in range(1, max_lag + 1):
        mi[tau] = mutual_information(arr[:-tau], arr[tau:], bins=n_bins)
    return mi


def first_minimum(series, max_lag, n_bins=16):
    """Return the lag of the first local minimum of the mutual information.

    Computes the delay-selection curve with
    :func:`embedded_mutual_information` and returns the lag of its
    first local minimum, the standard delay estimate for phase-space
    reconstruction (Fraser & Swinney 1986).

    Parameters
    ----------
    series : array_like
        Input scalar series.
    max_lag : int
        Maximum lag to scan.  Must be at least 2 so that a local
        minimum can exist, and smaller than the series length.
    n_bins : int, default 16
        Number of histogram bins.

    Returns
    -------
    int
        Estimated delay (the lag of the first local minimum).  If no
        local minimum is found, returns the lag of the global minimum
        as a fallback.

    Raises
    ------
    ValueError
        If ``max_lag < 2`` (no local minimum can exist) or if the
        arguments are invalid for :func:`embedded_mutual_information`.
    """
    if max_lag < 2:
        raise ValueError(
            f"max_lag must be >= 2 to locate a local minimum, got {max_lag}"
        )
    mi = embedded_mutual_information(series, max_lag, n_bins=n_bins)
    local_minima = argrelextrema(mi, np.less)[0]
    if local_minima.size:
        return int(local_minima[0])
    return int(np.argmin(mi))


def _build_gaussian_kernel(series, sigma=None):
    """
    Build a Gaussian (Radial Basis Function) kernel matrix for a given series.

    Parameters
    ----------
    series : array_like
        Input 1D time series or 2D feature matrix (samples, features).
    sigma : float, optional
        Bandwidth parameter for the Gaussian kernel. If None, it is estimated
        using the median of the pairwise distances (Silverman's heuristic).

    Returns
    -------
    numpy.ndarray
        The computed Gram matrix (kernel matrix).
    """
    data = np.asarray(series)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Calculate pairwise Euclidean distances
    sq_dists = pdist(data, metric="sqeuclidean")

    # Estimate sigma using the median distance if not provided
    if sigma is None:
        # Avoid zero division if all points are identical
        median_sq_dist = np.median(sq_dists)
        sigma_sq = median_sq_dist if median_sq_dist > 0 else 1.0
    else:
        sigma_sq = sigma**2

    # Compute the Gaussian Kernel Matrix
    K_condensed = np.exp(-sq_dists / (2.0 * sigma_sq))
    K = squareform(K_condensed)

    # Fill the main diagonal with 1.0 (distance to itself is 0)
    np.fill_diagonal(K, 1.0)

    return K


def _matrix_renyi_entropy(K, alpha=2.0):
    """
    Calculate the Matrix-based Renyi Entropy of a kernel matrix.

    Parameters
    ----------
    K : numpy.ndarray
        A square, symmetric kernel matrix.
    alpha : float, default 2.0
        The order of the Renyi entropy. alpha=2 corresponds to collision entropy,
        which is highly optimized in this implementation.

    Returns
    -------
    float
        The computed Renyi entropy in bits.
    """
    # Normalize the kernel matrix to act as a density matrix (Trace = 1)
    trace_K = np.trace(K)
    A = K / trace_K

    if alpha == 2.0:
        # Fast path for alpha = 2.0
        # The trace of A^2 for a symmetric matrix is simply the sum of its squared elements.
        # This dramatically reduces time complexity from O(N^3) to O(N^2).
        trace_A_alpha = np.sum(A**2)
    else:
        # General path for other values of alpha using eigenvalue decomposition O(N^3)
        eigenvalues = np.linalg.eigvalsh(A)
        # Filter out negative eigenvalues caused by floating-point inaccuracies
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        trace_A_alpha = np.sum(eigenvalues**alpha)

    # Compute the entropy
    entropy = (1.0 / (1.0 - alpha)) * np.log2(trace_A_alpha)

    return entropy


def matrix_renyi_mutual_information(x, y, alpha=2.0, sigma_x=None, sigma_y=None):
    """
    Calculate the Matrix-based Renyi Mutual Information between two time series.

    This method estimates mutual information directly from the eigenspectrum of
    kernel matrices, entirely avoiding Probability Density Function (PDF) estimation
    (such as binning or KDE).

    Reference
    ---------
    Sanchez Giraldo, L. G., Rao, A. R., & Principe, J. C. (2014).
    Measures of entropy from data using infinitely divisible kernels.
    IEEE Transactions on Information Theory, 60(10), 6212-6222.
    DOI: 10.1109/TIT.2014.2340023

    Parameters
    ----------
    x : array_like
        First input time series.
    y : array_like
        Second input time series. Must have the same length as x.
    alpha : float, default 2.0
        The order of the Renyi entropy.
    sigma_x : float, optional
        Bandwidth for the Gaussian kernel of x.
    sigma_y : float, optional
        Bandwidth for the Gaussian kernel of y.

    Returns
    -------
    float
        The estimated mutual information I(X;Y) in bits.
    """
    if len(x) != len(y):
        raise ValueError("Input series x and y must have the same length.")

    # 1. Build kernel matrices for individual variables
    K_x = _build_gaussian_kernel(x, sigma=sigma_x)
    K_y = _build_gaussian_kernel(y, sigma=sigma_y)

    # 2. Build the joint kernel matrix
    # In the RKHS framework, the joint representation is the Hadamard (element-wise) product
    K_xy = K_x * K_y

    # 3. Calculate individual entropies S(X) and S(Y)
    H_x = _matrix_renyi_entropy(K_x, alpha=alpha)
    H_y = _matrix_renyi_entropy(K_y, alpha=alpha)

    # 4. Calculate joint entropy S(X, Y)
    H_xy = _matrix_renyi_entropy(K_xy, alpha=alpha)

    # 5. Calculate Mutual Information: I(X;Y) = S(X) + S(Y) - S(X,Y)
    mutual_info = H_x + H_y - H_xy

    # Prevent negative output due to numerical floating-point errors
    return max(0.0, float(mutual_info))
