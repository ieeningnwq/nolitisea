"""Polynomial-interpolation resampling."""

import numpy as np

__all__ = ["resample"]


def resample(series, sampletime=0.5, order=4):
    """Resample a scalar time series via local polynomial interpolation.

    Fits a polynomial of degree ``order`` to ``order + 1`` consecutive
    samples and evaluates it at equally spaced fractional positions, so the
    output sampling interval is ``sampletime`` (in units of the original
    interval).

    Parameters
    ----------
    series : array_like
        1-D input time series.
    sampletime : float, optional (default = 0.5)
        New sampling time in units of the old one.  Values < 1 up-sample
        (more output points), values > 1 down-sample (fewer points).
    order : int, optional (default = 4)
        Order of the interpolating polynomial.  ``order + 1`` consecutive
        samples are used for each local fit.

    Returns
    -------
    numpy.ndarray
        1-D resampled series.

    Raises
    ------
    ValueError
        If ``order`` is invalid or the series is too short for the
        interpolation window.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = x.size

    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if sampletime <= 0.0:
        raise ValueError(f"sampletime must be > 0, got {sampletime}")

    horder = order + 1                       # number of polynomial coefficients
    horder2 = (horder + 1) // 2 - horder     # integer window offset

    # Window must fit inside the series.
    if n < horder:
        raise ValueError(
            f"series of length {n} is too short for order={order} "
            f"(need at least {horder} samples)"
        )

    # --- Vandermonde matrix with nodes at horder2, ..., horder2+horder-1 ---
    nodes = np.arange(horder, dtype=np.float64) + horder2
    mat = np.vander(nodes, increasing=True)   # mat[i, j] = nodes[i]**j
    imat = np.linalg.inv(mat)

    # --- output time grid (C: time starts at (horder+1)/2., ends < n-horder/2) ---
    start = (horder + 1) / 2.0
    end = n - horder // 2                     # exclusive upper bound (C: integer div)
    times = np.arange(start, end, sampletime, dtype=np.float64)
    if times.size == 0:
        raise ValueError(
            f"no output points: series too short (n={n}) for the requested "
            f"sampletime={sampletime}"
        )

    # --- vectorised window extraction & polynomial evaluation ---
    itimes = times.astype(int) + horder2                      # window start indices
    htimes = times - itimes + horder2                         # fractional offsets

    # windows[i] = series[itimes[i] : itimes[i]+horder]  ->  shape (n_out, horder)
    window_idx = itimes[:, None] + np.arange(horder)[None, :]
    windows = x[window_idx]                                    # (n_out, horder)

    # polynomial coefficients: coef = imat @ vec  ->  windows @ imat.T
    coefs = windows @ imat.T                                   # (n_out, horder)

    # evaluate polynomial at fractional offsets
    powers = htimes[:, None] ** np.arange(horder)[None, :]     # (n_out, horder)
    result = np.sum(coefs * powers, axis=1)

    return result
