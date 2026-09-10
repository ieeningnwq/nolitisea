"""Conditional (multivariate) transfer entropy TE(X -> Y | Z).

Transfer entropy (Schreiber 2000) measures the directed information
flow from a *source* X to a *target* Y.  The *conditional* (or
multivariate) transfer entropy further conditions on one or more
system variables Z, isolating the genuine X -> Y flow from spurious
links induced by common drivers (confounders) and indirect paths::

    TE_{X->Y | Z} = I(Y_{t+h} ; X_t^{(l)}  |  Y_t^{(k)}, Z_t^{(l_z)})

This is the conditional mutual information between the source history
and the target's future, given both the target's own past and the
history of the conditioning variable(s) Z.
"""

from __future__ import annotations

import numpy as np

from nolitisea.entropy.transfer_entropy import _digitize, _sample_counts

__all__ = ["multivariate_transfer_entropy"]


def multivariate_transfer_entropy(
    source,
    target,
    condition,
    k: int = 1,
    l: int = 1,
    l_z=1,
    h: int = 1,
    bins: int = 10,
) -> float:
    """Estimate the conditional transfer entropy TE(X -> Y | Z).

    Measures the directed information flow from ``source`` (X) to
    ``target`` (Y) *conditioned on* one or more system variables Z [1]_::

        TE_{X->Y | Z} = I(Y_{t+h} ; X_t^{(l)}  |  Y_t^{(k)}, Z_t^{(l_z)})

    This is the conditional mutual information between the source
    history and the target's future, given both the target's own past
    and the conditioning variable(s)' history.  Conditioning on Z
    removes the spurious X -> Y flow that would otherwise be induced by
    a common driver Z of both X and Y (confounding), or by an indirect
    path X -> Z -> Y.

    Estimation uses equi-width histogram discretization with ``bins``
    bins per scalar variable, the same estimator as the bivariate
    :func:`nolitisea.entropy.transfer_entropy.transfer_entropy`.

    Parameters
    ----------
    source : array_like
        Source (driving) 1-D time series X.
    target : array_like
        Target (driven) 1-D time series Y.  Must have the same length
        as ``source``.
    condition : array_like or sequence of array_like
        Conditioning variable(s) Z.  A single 1-D array is treated as
        one variable; a list/tuple of 1-D arrays (or a 2-D array of
        shape ``(n_cond, n_samples)``) is treated as multiple
        variables.  Every variable must have the same length as
        ``source`` and ``target``.
    k : int, default 1
        History length of the target (number of past target values
        conditioning the prediction).
    l : int, default 1
        History length of the source.
    l_z : int or sequence of int, default 1
        History length(s) of the conditioning variable(s).  An ``int``
        applies the same length to every conditioning variable; a
        sequence must have one entry per variable.
    h : int, default 1
        Prediction time (steps ahead of the current base time).
    bins : int, default 10
        Number of equi-width bins per scalar variable for the
        histogram probability estimator.

    Returns
    -------
    float
        Conditional transfer entropy in bits (base-2 logarithm).
        Non-negative by construction; near zero when ``source``
        carries no *direct* information about ``target`` beyond what
        the target's own past and the conditioning variable(s) Z
        already explain.

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

    # Normalize the conditioning variable(s) into a list of 1-D arrays.
    if isinstance(condition, np.ndarray) and condition.ndim == 2:
        cond_list = [condition[i] for i in range(condition.shape[0])]
    elif isinstance(condition, (list, tuple)):
        cond_list = list(condition)
    else:
        cond_list = [condition]
    cond_list = [np.asarray(c, dtype=np.float64).ravel() for c in cond_list]
    n_cond = len(cond_list)

    N = y.size
    if x.size != N:
        raise ValueError(
            f"source length ({x.size}) does not match target length "
            f"({N})."
        )
    for i, c in enumerate(cond_list):
        if c.size != N:
            raise ValueError(
                f"condition {i} length ({c.size}) does not match "
                f"target length ({N})."
            )

    # Normalize l_z per conditioning variable.
    if np.ndim(l_z) == 0:
        l_z_vals = [int(l_z)] * n_cond
    else:
        l_z_vals = [int(v) for v in l_z]  # pyright: ignore[reportGeneralTypeIssues]
        if len(l_z_vals) != n_cond:
            raise ValueError(
                f"l_z has {len(l_z_vals)} entries but {n_cond} "
                f"conditioning variables were provided."
            )

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if l < 1:
        raise ValueError(f"l must be >= 1, got {l}")
    for v in l_z_vals:
        if v < 1:
            raise ValueError(f"every l_z must be >= 1, got {v}")
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    max_hist = max(k - 1, l - 1, max(l_z_vals) - 1) if l_z_vals else max(
        k - 1, l - 1
    )
    n_samples = N - max_hist - h
    if n_samples <= 0:
        raise ValueError(
            f"series of length {N} is too short for k={k}, l={l}, "
            f"l_z={l_z_vals}, h={h}: only {n_samples} usable samples."
        )

    # Build target history (most-recent first, Schreiber convention).
    y_hist = np.column_stack(
        [y[max_hist - i : max_hist - i + n_samples] for i in range(k)]
    )
    # Build source history.
    x_hist = np.column_stack(
        [x[max_hist - i : max_hist - i + n_samples] for i in range(l)]
    )
    # Build per-conditioning-variable histories.
    z_hists = []
    for c_idx in range(n_cond):
        lzi = l_z_vals[c_idx]
        zh = np.column_stack(
            [cond_list[c_idx][max_hist - i : max_hist - i + n_samples]
             for i in range(lzi)]
        )
        z_hists.append(zh)
    y_future = y[max_hist + h : max_hist + h + n_samples]

    # Digitize each scalar variable into integer bin indices.
    y_h_d = [_digitize(y_hist[:, i], bins) for i in range(k)]
    x_h_d = [_digitize(x_hist[:, i], bins) for i in range(l)]
    z_h_d_all = []
    for c_idx in range(n_cond):
        lzi = l_z_vals[c_idx]
        z_h_d_all.extend(
            [_digitize(z_hists[c_idx][:, i], bins) for i in range(lzi)]
        )
    y_f_d = _digitize(y_future, bins)

    # Conditional MI = I(A; B | C) where
    #   A = Y_{t+h},  B = X_t^{(l)},  C = (Y_t^{(k)}, Z_t^{(l_z)})
    # Per-sample count form:
    #   (1/n) * sum log2[ n(a,b,c) * n(c) / (n(b,c) * n(a,c)) ]
    # so:
    #   n_full     = n(y_f, y_h, x_h, z_h)   = n(a, b, c)
    #   n_yh_zh    = n(y_h, z_h)              = n(c)
    #   n_xh_yh_zh = n(x_h, y_h, z_h)         = n(b, c)
    #   n_yf_yh_zh = n(y_f, y_h, z_h)         = n(a, c)
    c_full = _sample_counts(y_f_d, *y_h_d, *x_h_d, *z_h_d_all)
    c_yh_zh = _sample_counts(*y_h_d, *z_h_d_all)
    c_xh_yh_zh = _sample_counts(*x_h_d, *y_h_d, *z_h_d_all)
    c_yf_yh_zh = _sample_counts(y_f_d, *y_h_d, *z_h_d_all)

    ratio = (c_full * c_yh_zh) / (c_xh_yh_zh * c_yf_yh_zh)
    te = float(np.sum(np.log2(ratio)) / n_samples)
    return te


if __name__ == "__main__":
    from nolitisea.entropy.transfer_entropy import transfer_entropy

    rng = np.random.default_rng(0)
    n = 8000

    # --- Case 1: spurious (common-cause) link removed by conditioning ---
    # Z is strongly autocorrelated and drives both X and Y at the same
    # lag.  X[t] ~= Z[t-1] predicts Y[t+1] ~= Z[t] through Z's memory,
    # so TE(X->Y) > 0 (spurious).  Conditioning on Z[t] removes it:
    # once Z[t] is known, X[t] = Z[t-1] carries no extra information
    # about Y[t+1] = Z[t] + noise.
    z = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)
    z[0] = rng.standard_normal()
    for t in range(n - 1):
        z[t + 1] = 0.95 * z[t] + 0.1 * rng.standard_normal()
        x[t + 1] = z[t] + 0.02 * rng.standard_normal()
        y[t + 1] = z[t] + 0.02 * rng.standard_normal()

    te_biv = transfer_entropy(x, y, bins=8)
    te_cond = multivariate_transfer_entropy(x, y, z, bins=8)
    print("Common-cause (Z autocorrelated; Z -> X, Z -> Y):")
    print(f"  TE(X->Y)      = {te_biv:.4f} bits  (spurious)")
    print(f"  TE(X->Y | Z)  = {te_cond:.4f} bits  (conditioning removes it)")

    # --- Case 2: genuine direct link survives conditioning ---
    # X -> Y directly, plus Z drives Y as well.
    x2 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    y2 = np.zeros(n)
    for t in range(n - 1):
        y2[t + 1] = 0.5 * y2[t] + 0.5 * x2[t] + 0.3 * z2[t] \
            + 0.2 * rng.standard_normal()

    te_biv2 = transfer_entropy(x2, y2, bins=8)
    te_cond2 = multivariate_transfer_entropy(x2, y2, z2, bins=8)
    print("\nDirect link X -> Y plus Z -> Y:")
    print(f"  TE(X->Y)      = {te_biv2:.4f} bits")
    print(f"  TE(X->Y | Z)  = {te_cond2:.4f} bits  (direct link survives)")

    # --- Case 3: multiple conditioning variables ---
    z3a = rng.standard_normal(n)
    z3b = rng.standard_normal(n)
    y3 = np.zeros(n)
    for t in range(n - 1):
        y3[t + 1] = 0.4 * y3[t] + 0.3 * z3a[t] + 0.3 * z3b[t] \
            + 0.2 * rng.standard_normal()
    te_cond3 = multivariate_transfer_entropy(
        x2, y3, [z3a, z3b], bins=6
    )
    print("\nMultiple conditioning variables [Z1, Z2]:")
    print(f"  TE(X->Y | Z1, Z2) = {te_cond3:.4f} bits")
