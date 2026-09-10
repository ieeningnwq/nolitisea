import numpy as np

__all__ = ["higuchi_fd"]


def higuchi_fd(series_matrix, kmax=10):
    """Higuchi fractal dimension of one or more time series.

    For each scalar time series X(0), ..., X(N-1) the method constructs,
    for every time lag k = 1, ..., kmax and every start offset
    m = 0, ..., k-1, a decimated sub-series

        X(m), X(m+k), X(m+2k), ..., X(m + n_mk * k),

    where n_mk = floor((N - m - 1) / k) is the number of intervals.
    The normalised length of that sub-series is

        L_m(k) = sum_i |X(m + i*k) - X(m + (i-1)*k)|
                 * (N - 1) / (n_mk * k**2),

    and the mean curve length is L(k) = <L_m(k)>_m.  The Higuchi
    fractal dimension is the negative slope of the log-log fit
    L(k) ~ k**(-D), estimated by least-squares linear regression of
    ln L(k) versus ln k.

    Expected values: ~1.0 for a smooth curve, ~1.5 for a random walk
    (Brownian motion), ~2.0 for white noise.

    Parameters
    ----------
    series_matrix : array_like
        A 1-D array of shape ``(n_samples,)`` (a single series) or a
        2-D array of shape ``(n_samples, n_series)`` where each column
        is an independent time series and rows are time steps.
    kmax : int
        Maximum time lag k. Default 10. Must be >= 2.

    Returns
    -------
    numpy.ndarray
        A 1-D array of shape ``(n_series,)`` with the HFD of each
        column (shape ``(1,)`` for a 1-D input).

    Raises
    ------
    ValueError
        If the input is not 1-D or 2-D, ``kmax`` < 2, or the series is
        too short (fewer than ``2 * kmax`` samples).

    References
    ----------
    .. [1] Higuchi, T. (1988).  Approach to an irregular time series
           analysis on the basis of the fractal theory.  Physica D:
           Nonlinear Phenomena, 31(2), 277-283.
    """
    x = np.asarray(series_matrix, dtype=np.float64)

    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(
            f"Input must be 1-D or 2-D, got {x.ndim}D."
        )
    if kmax < 2:
        raise ValueError(f"kmax must be >= 2, got {kmax}.")

    n_samples, n_series = x.shape

    # Every sub-series needs at least one interval.  The worst case is
    # k = kmax, m = kmax - 1, which requires n_samples >= 2 * kmax.
    if n_samples < 2 * kmax:
        raise ValueError(
            f"series of length {n_samples} is too short for kmax={kmax}: "
            f"need at least {2 * kmax} samples."
        )
    
    # Pre-allocate array to store the mean curve length L(k) for all series
    # Shape: (kmax, series_count)
    L = np.empty((kmax, n_series))

    # Step 2: Iterate over each interval k
    for k in range(1, kmax + 1):
        # Temporary array to store length Lm for each starting point m
        # Shape: (k, series_count)
        Lm_k = np.empty((k, n_series))
        
        for m in range(k):
            # Calculate the number of intervals for the current sub-series
            n_intervals = int((n_samples - m - 1) / k)
            
            # Generate row indices for the sub-series
            indices = np.arange(m, m + n_intervals * k + 1, k)
            
            # Vectorized absolute differences along the time axis (axis=0)
            # x[indices, :] shape: (len(indices), n_series)
            diffs = np.diff(x[indices, :], axis=0)
            abs_diff_sum = np.sum(np.abs(diffs), axis=0)
            
            # Normalization factor to compensate for the decimation
            norm_factor = (n_samples - 1) / (n_intervals * k)
            
            # Calculate length for the current m and k
            Lm_k[m] = abs_diff_sum * norm_factor / k
            
        # Average lengths over all m starting points
        # np.mean along axis=0 averages the k different Lm lengths for each series
        L[k - 1] = np.mean(Lm_k, axis=0)

    # Step 3: Perform log-log linear regression
    # x_fit = ln(k), shape: (kmax,)
    # y_fit = ln(L(k)), shape: (kmax, series_count)
    k_array = np.arange(1, kmax + 1)
    x_fit = np.log(k_array)
    y_fit = np.log(L)
    
    # np.polyfit fits a 1st degree polynomial (line) independently for each column in y_fit
    # coeffs shape: (2, series_count), where coeffs[0] contains the slopes
    coeffs = np.polyfit(x_fit, y_fit, 1)
    slopes = coeffs[0]
    
    # Step 4: Extract HFD values
    # According to Higuchi's algorithm, L(k) ~ k^(-D), so HFD is the negative slope
    hfd_values = -slopes

    return hfd_values

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # Generate synthetic testing data
    np.random.seed(42)
    num_samples = 2000
    num_series = 5
    
    # Create a random walk matrix of shape (2000, 5)
    # The expected HFD for a random walk is approximately 1.5
    test_data = np.cumsum(np.random.randn(num_samples, num_series), axis=0)
    
    # Set the maximum interval parameter
    k_max_param = 15 
    
    # Calculate HFD for all series simultaneously
    hfd_results = higuchi_fd(test_data, k_max_param)
    
    print("Calculated Higuchi Fractal Dimensions for each series:")
    for idx, hfd in enumerate(hfd_results):
        print(f"Series {idx + 1}: {hfd:.4f}")