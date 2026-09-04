"""Compare two data sets."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def mae(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mean Absolute Error between reference and test 1-D time series.

    Parameters
    ----------
    x : array_like
        Reference / ground-truth 1-D time series.
    y : array_like
        Test / reconstructed / denoised 1-D time series.

    Returns
    -------
    float
        Mean absolute error.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Inputs must be one-dimensional arrays.")
    x = x.ravel()
    y = y.ravel()
    if x.size != y.size:
        raise ValueError(
            f"Length mismatch: reference length={x.size}, test length={y.size}"
        )

    residual = x - y
    return float(np.mean(np.abs(residual)))


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mean Squared Error between reference and test 1-D time series.

    Parameters
    ----------
    x : array_like
        Reference / ground-truth 1-D time series.
    y : array_like
        Test / reconstructed / denoised 1-D time series.

    Returns
    -------
    float
        Mean squared error.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Inputs must be one-dimensional arrays.")
    x = x.ravel()
    y = y.ravel()
    if x.size != y.size:
        raise ValueError(
            f"Length mismatch: reference length={x.size}, test length={y.size}"
        )

    residual = x - y
    return float(np.mean(residual**2))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Root Mean Squared Error between reference and test 1-D time series.

    Parameters
    ----------
    x : array_like
        Reference / ground-truth 1-D time series.
    y : array_like
        Test / reconstructed / denoised 1-D time series.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(np.sqrt(mse(x, y)))


def nrmse_ptp(x: np.ndarray, y: np.ndarray) -> float:
    """
    Peak-to-peak normalized Root Mean Squared Error (NRMSE).
    Normalized by peak-to-peak range (max-min) of reference series x.

    Parameters
    ----------
    x : array_like
        Reference / ground-truth 1-D time series.
    y : array_like
        Test / reconstructed / denoised 1-D time series.

    Returns
    -------
    float
        NRMSE value. Returns np.nan if reference series is constant.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Inputs must be one-dimensional arrays.")
    x = x.ravel()
    y = y.ravel()
    if x.size != y.size:
        raise ValueError(
            f"Length mismatch: reference length={x.size}, test length={y.size}"
        )

    rmse_val = rmse(x, y)
    ptp_x = np.ptp(x)

    if np.isclose(ptp_x, 0.0):
        return float(np.nan)
    return float(rmse_val / ptp_x)


def r2_score(x: np.ndarray, y: np.ndarray) -> float:
    """
    Coefficient of determination R² score.
    R² can be negative, which indicates worse performance than predicting mean of reference x.

    Parameters
    ----------
    x : array_like
        Reference / ground-truth 1-D time series.
    y : array_like
        Test / reconstructed / denoised 1-D time series.

    Returns
    -------
    float
        R-squared value. Returns np.nan if reference series has zero variance.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Inputs must be one-dimensional arrays.")
    x = x.ravel()
    y = y.ravel()
    if x.size != y.size:
        raise ValueError(
            f"Length mismatch: reference length={x.size}, test length={y.size}"
        )

    residual = x - y
    sse = np.sum(residual**2)
    sst = np.sum((x - np.mean(x)) ** 2)

    if np.isclose(sst, 0.0):
        return float(np.nan)
    return float(1.0 - sse / sst)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Pearson linear correlation coefficient between two 1-D time series.

    Parameters
    ----------
    x : array_like
        Reference 1-D time series.
    y : array_like
        Test 1-D time series.

    Returns
    -------
    r : float
        Pearson correlation coefficient, range [-1, 1].
    p_value : float
        Two-tailed p-value for correlation significance test.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Inputs must be one-dimensional arrays.")
    x = x.ravel()
    y = y.ravel()
    if x.size != y.size:
        raise ValueError(
            f"Length mismatch: reference length={x.size}, test length={y.size}"
        )

    r, p_value = pearsonr(x, y)
    return float(r), float(p_value)


def max_cross_correlation(
    x: np.ndarray, y: np.ndarray, max_lag: int | None = None
) -> tuple[float, int]:
    """
    Maximum normalized cross-correlation allowing time shift.
    Computes cross-correlation and returns peak value and corresponding lag index.

    Parameters
    ----------
    x : array_like
        Reference 1-D time series.
    y : array_like
        Test 1-D time series.
    max_lag : int, optional
        Maximum lag range to search. If None, uses full possible lag range.

    Returns
    -------
    max_r : float
        Maximum absolute cross-correlation coefficient.
    best_lag : int
        Lag index which achieves maximum correlation (y shifted relative to x).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Inputs must be one-dimensional arrays.")
    x = x.ravel()
    y = y.ravel()
    n = x.size
    if n != y.size:
        raise ValueError(
            f"Length mismatch: reference length={x.size}, test length={y.size}"
        )

    # Normalize zero-mean
    x_norm = x - np.mean(x)
    y_norm = y - np.mean(y)
    cov_xx = np.sum(x_norm**2)
    cov_yy = np.sum(y_norm**2)
    if np.isclose(cov_xx, 0.0) or np.isclose(cov_yy, 0.0):
        return float(np.nan), 0

    full_cc = np.correlate(x_norm, y_norm, mode="full")
    full_cc = full_cc / np.sqrt(cov_xx * cov_yy)

    if max_lag is not None and max_lag > 0:
        center = n - 1
        low = max(0, center - max_lag)
        high = min(2 * n - 1, center + max_lag + 1)
        cc_slice = full_cc[low:high]
        offset_idx = np.argmax(np.abs(cc_slice))
        best_idx = low + offset_idx
    else:
        best_idx = np.argmax(np.abs(full_cc))

    max_r = float(full_cc[best_idx])
    best_lag = int(best_idx - (n - 1))
    return max_r, best_lag


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D time-series vectors.
    Measures vector direction, magnitude is ignored.

    Parameters
    ----------
    x : array_like
        Reference 1-D time series.
    y : array_like
        Test 1-D time series.

    Returns
    -------
    float
        Cosine similarity, range [-1, 1]. Returns np.nan for zero-norm vectors.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Inputs must be one-dimensional arrays.")
    x = x.ravel()
    y = y.ravel()
    if x.size != y.size:
        raise ValueError(
            f"Length mismatch: reference length={x.size}, test length={y.size}"
        )

    dot = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if np.isclose(norm_x, 0.0) or np.isclose(norm_y, 0.0):
        return float(np.nan)
    return float(dot / (norm_x * norm_y))


def _dtw(x, y):
    """Dynamic time warping of two 1-D series: ``(distance, path)``.

    Replaces ``scipy.spatial.distance.dtw``, which does not exist in
    any released SciPy.  The distance is the accumulated absolute
    difference along an optimal warping path under the classic
    three-step recursion (diagonal, down, right); the path is a list
    of 0-based ``(i, j)`` index pairs into ``x`` and ``y``.
    """
    n = x.size
    m = y.size
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        xi = x[i - 1]
        above = acc[i - 1]
        row = acc[i]
        for j in range(1, m + 1):
            best = above[j - 1]
            best = min(best, above[j])
            best = min(best, row[j - 1])
            row[j] = abs(xi - y[j - 1]) + best
    # Backtrack one optimal warping path (any tie-break is valid).
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            i, j = min(
                ((i - 1, j - 1), (i - 1, j), (i, j - 1)),
                key=lambda p: acc[p[0], p[1]],
            )
        path.append((i - 1, j - 1))
    path.reverse()
    return acc[n, m], path


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Dynamic Time Warping distance.
    Measures shape similarity allowing local time stretching/compression.
    Smaller value indicates higher shape similarity.

    Parameters
    ----------
    x : array_like
        Reference 1-D time series.
    y : array_like
        Test 1-D time series.

    Returns
    -------
    float
        Unnormalized DTW distance.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Inputs must be one-dimensional arrays.")
    x = x.ravel()
    y = y.ravel()

    dist, _ = _dtw(x, y)
    return float(dist)
