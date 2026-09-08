"""Transfer entropy for measuring directed information flow.

Transfer entropy (Schreiber 2000) quantifies the reduction in uncertainty
about the future state of a *target* series gained by learning the past of
a *source* series, beyond what the target's own past already explains.
"""

from __future__ import annotations

import numpy as np

__all__ = ["transfer_entropy"]


def _digitize(arr: np.ndarray, n_bins: int) -> np.ndarray:
    """Map values to integer bin indices ``0..n_bins-1`` (equi-width).

    A constant variable (``min == max``) is mapped entirely to bin 0,
    so the histogram estimator degrades gracefully without division
    by zero.
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.int64)
    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    idx = np.digitize(arr, edges)
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


def _sample_counts(*arrays: np.ndarray) -> np.ndarray:
    """Return per-sample count of each sample's joint combination.

    Packs the integer bin indices of every variable into a single key,
    finds unique keys via ``np.unique``, and looks up each sample's
    combination count.  Because constant variables collapse to a single
    bin, the pack multiplier adapts to the actual cardinality, keeping
    the key space compact.
    """
    arrays = [np.asarray(a, dtype=np.int64) for a in arrays]  # pyright: ignore[reportAssignmentType]
    n = len(arrays[0])
    if n == 0:
        return np.empty(0, dtype=np.float64)
    key = arrays[0].copy()
    for a in arrays[1:]:
        key = key + a * (int(key.max()) + 1)
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inv].astype(np.float64)


def transfer_entropy(
    source,
    target,
    k: int = 1,
    l: int = 1,
    h: int = 1,
    bins: int = 10,
) -> float:
    """Estimate the transfer entropy TE(source -> target).

    Transfer entropy measures the directed information flow from
    ``source`` (X) to ``target`` (Y) [1]_::

        TE_{X->Y} = I(Y_{t+h} ; X_t^{(l)}  |  Y_t^{(k)})
                  = sum  p(y_{t+h}, y_t^{(k)}, x_t^{(l)})
                          * log2[ p(y_{t+h}, y_t^{(k)}, x_t^{(l)}) * p(y_t^{(k)})
                                  / ( p(y_t^{(k)}, x_t^{(l)}) * p(y_{t+h}, y_t^{(k)}) ) ]

    where ``y_t^{(k)} = (y_t, y_{t-1}, ..., y_{t-k+1})`` is the
    k-history of the target and ``x_t^{(l)} = (x_t, x_{t-1}, ...,
    x_{t-l+1})`` is the l-history of the source.  This is the
    conditional mutual information between the source history and the
    target's future, conditioned on the target's own history.

    Estimation uses equi-width histogram discretization with ``bins``
    bins per scalar variable, following the standard binning-based
    estimator (e.g. Schreiber 2000, Marschinski & Kantz 2002).

    Parameters
    ----------
    source : array_like
        Source (driving) 1-D time series X.
    target : array_like
        Target (driven) 1-D time series Y.  Must have the same length
        as ``source``.
    k : int, default 1
        History length of the target (number of past target values
        conditioning the prediction).
    l : int, default 1
        History length of the source (number of past source values
        used as the driving signal).
    h : int, default 1
        Prediction time (steps ahead of the current base time).
    bins : int, default 10
        Number of equi-width bins per scalar variable for the
        histogram probability estimator.

    Returns
    -------
    float
        Transfer entropy in bits (base-2 logarithm).  Non-negative
        by construction; near zero when ``source`` carries no
        directional information about ``target`` beyond what the
        target's own past provides.

    Raises
    ------
    ValueError
        For length mismatch, invalid parameters, or series too short
        for the requested history lengths and prediction time.

    References
    ----------
    .. [1] Schreiber, T. (2000).  Measuring information transfer.
           Physical Review Letters, 85(2), 461-464.
    """
    x = np.asarray(source, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()

    if x.size != y.size:
        raise ValueError(
            f"source and target must have the same length, got "
            f"{x.size} and {y.size}."
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if l < 1:
        raise ValueError(f"l must be >= 1, got {l}")
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    N = y.size
    max_hist = max(k - 1, l - 1)
    n_samples = N - max_hist - h
    if n_samples <= 0:
        raise ValueError(
            f"series of length {N} is too short for k={k}, l={l}, h={h}: "
            f"only {n_samples} usable samples."
        )

    # Build delay vectors for the base time t = max_hist .. N - h - 1.
    # y_hist[:, i] = y_{t-i} (most-recent first, Schreiber convention).
    # x_hist[:, i] = x_{t-i}
    # y_future    = y_{t+h}
    y_hist = np.column_stack(
        [y[max_hist - i : max_hist - i + n_samples] for i in range(k)]
    )
    x_hist = np.column_stack(
        [x[max_hist - i : max_hist - i + n_samples] for i in range(l)]
    )
    y_future = y[max_hist + h : max_hist + h + n_samples]

    # Digitize each scalar variable into integer bin indices.
    y_h_d = [_digitize(y_hist[:, i], bins) for i in range(k)]
    x_h_d = [_digitize(x_hist[:, i], bins) for i in range(l)]
    y_f_d = _digitize(y_future, bins)

    # Per-sample joint counts for each required distribution.
    # All counts are >= 1 for every sample (each sample belongs to at
    # least one joint cell), so ratios are always well-defined.
    c_full = _sample_counts(y_f_d, *y_h_d, *x_h_d)   # n(y_f, y_h, x_h)
    c_yf_yh = _sample_counts(y_f_d, *y_h_d)           # n(y_f, y_h)
    c_yh_xh = _sample_counts(*y_h_d, *x_h_d)         # n(y_h, x_h)
    c_yh = _sample_counts(*y_h_d)                     # n(y_h)

    # TE = (1/n) * sum_s log2[ n_full(s) * n_yh(s)
    #                          / (n_yh_xh(s) * n_yf_yh(s)) ]
    #
    # Grouping samples by their joint cell c with count n_c gives
    # (1/n) * sum_c n_c * log2(ratio_c) = sum_c p_c * log2(ratio_c),
    # which is the standard histogram TE estimator.  The per-sample
    # form is numerically equivalent and avoids an explicit unique loop.
    ratio = (c_full * c_yh) / (c_yh_xh * c_yf_yh)
    te = float(np.sum(np.log2(ratio)) / n_samples)
    return te


if __name__ == "__main__":
    # Quick self-test: a coupled AR(1) where X drives Y, and two
    # independent processes as a null case.
    rng = np.random.default_rng(0)
    n = 5000
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(n - 1):
        x[t + 1] = 0.8 * x[t] + 0.2 * rng.standard_normal()
        y[t + 1] = 0.5 * y[t] + 0.5 * x[t] + 0.2 * rng.standard_normal()

    te_xy = transfer_entropy(x, y, bins=8)
    te_yx = transfer_entropy(y, x, bins=8)
    print("Coupled AR(1)  X -> Y:")
    print(f"  TE(X->Y) = {te_xy:.4f} bits")
    print(f"  TE(Y->X) = {te_yx:.4f} bits")
    if te_yx > 0:
        print(f"  Directionality ratio = {te_xy / te_yx:.2f}")

    x2 = rng.standard_normal(5000)
    y2 = rng.standard_normal(5000)
    te_indep = transfer_entropy(x2, y2, bins=8)
    print("\nIndependent white noise:")
    print(f"  TE = {te_indep:.4f} bits (expected near 0)")
