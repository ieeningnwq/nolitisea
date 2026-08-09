import warnings

import numpy as np


def acorr(x, maxtau=None, norm=True, detrend=True):
    """Return the autocorrelation of the given scalar time series.

    Calculates the autocorrelation of the given scalar time series
    using the Wiener-Khinchin theorem.

    Parameters
    ----------
    x : array_like
        1-D real time series of length N.
    maxtau : int, optional (default = N)
        Return the autocorrelation only up to this time delay.
    norm : bool, optional (default = True)
        Normalize the autocorrelation so that it is equal to 1 for
        zero time delay.
    detrend: bool, optional (default = True)
        Subtract the mean from the time series (i.e., a constant
        detrend).  This is done so that for uncorrelated data, the
        autocorrelation vanishes for all nonzero time delays.

    Returns
    -------
    r : array
        Array with the autocorrelation up to maxtau.
    """
    x = np.asarray(x)
    N = len(x)

    if not maxtau:
        maxtau = N
    else:
        maxtau = min(N, maxtau)

    if detrend:
        x = x - np.mean(x)

    y = np.fft.fft(x, 2 * N - 1)
    r = np.real(np.fft.ifft(y * y.conj(), 2 * N - 1))

    if norm:
        return r[:maxtau] / r[0]
    else:
        return r[:maxtau]


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
