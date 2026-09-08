"""End-to-end mismatch measure.

Given a (possibly multivariate) time series, the function scans every
sub-sequence length from the full length down to three samples and every
starting offset, looking for the sub-sequence whose wrap-around is the
smoothest.  This is used to pick a stationary slice before generating
surrogate data (the AAFT and IAAFT algorithms assume periodic boundary
conditions, so a large jump between the last and first sample corrupts
the surrogate spectrum).

For a sub-series of length ``L`` the mismatch has two contributions,
each normalised by ``L * sd**2`` where ``sd`` is the population standard
deviation of the sub-series:

* **jump**  ``(x[0] - x[L-1])**2 / (L * sd**2)`` — discontinuity of the
  value itself at the wrap-around,
* **slip**  ``((x[L-1] - x[L-2]) - (x[1] - x[0]))**2 / (L * sd**2)`` —
  discontinuity of the first finite difference at the wrap-around.

For multivariate input the two contributions are summed across
components.  The weighted total

    etot = wjump * jump + (1 - wjump) * slip

is the quantity minimised over lengths and offsets; ``wjump``
defaults to ``0.5``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["endtoend"]

_TINY = 1e-30


def _sliding_pop_std(arr: np.ndarray, L: int) -> np.ndarray:
    """Population standard deviation of every length-``L`` window of ``arr``.

    Uses cumulative sums so the cost is ``O(n)`` per call rather than
    ``O(n * L)``.  Numerical floor at zero guards against tiny negative
    variances from cancellation.
    """
    n = arr.size
    n_win = n - L + 1
    if n_win <= 0:
        return np.empty(0, dtype=np.float64)

    cumsum = np.concatenate(([0.0], np.cumsum(arr.astype(np.float64))))
    cumsum_sq = np.concatenate(([0.0], np.cumsum((arr.astype(np.float64)) ** 2)))

    s = cumsum[L : L + n_win] - cumsum[:n_win]
    sq = cumsum_sq[L : L + n_win] - cumsum_sq[:n_win]
    mean = s / L
    var = sq / L - mean * mean
    np.maximum(var, 0.0, out=var)
    return np.sqrt(var)


def endtoend(series, wjump: float = 0.5) -> dict:
    """Find the stationary sub-sequence with the smallest wrap-around mismatch.

    Parameters
    ----------
    series : array_like
        Input data.  A 1-D array is a single component; a 2-D array must
        have shape ``(n_times, n_vars)`` with one column per component.
    wjump : float, default 0.5
        Weight of the *jump* (value-discontinuity) contribution relative
        to the *slip* (slope-discontinuity) contribution.
        ``etot = wjump * jump + (1 - wjump) * slip``.

    Returns
    -------
    dict
        ``"length"``   : int, optimal sub-sequence length ``L``.
        ``"offset"``   : int, starting index of the optimal sub-sequence.
        ``"lost"``     : float, fraction of samples discarded
        ``(n - L) / n``.
        ``"jump"``     : float, jump contribution (fraction).
        ``"slip"``     : float, slip contribution (fraction).
        ``"weighted"`` : float, weighted mismatch ``etot`` (fraction).
        Multiply by 100 to obtain percentages.

    Raises
    ------
    ValueError
        If ``series`` is not 1- or 2-D, has fewer than three samples, or
        if ``wjump`` is outside ``[0, 1]``.
    """
    data = np.asarray(series, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim != 2:
        raise ValueError(
            "series must be a 1-D array or a 2-D array of shape "
            "(n_times, n_vars)"
        )

    n, n_vars = data.shape
    if n < 3:
        raise ValueError(
            f"series must have at least 3 samples, got {n}"
        )
    if not (0.0 <= wjump <= 1.0):
        raise ValueError(f"wjump must be in [0, 1], got {wjump}")

    best_etot = np.inf
    best_length = n
    best_offset = 0
    best_jump = 0.0
    best_slip = 0.0

    # Scan every length from the full series down to 3 samples.
    for L in range(n, 2, -1):
        n_win = n - L + 1

        # Per-offset first/last/second/second-last values for each comp.
        first = data[:n_win]                       # shape (n_win, n_vars)
        last = data[L - 1 : L - 1 + n_win]
        second = data[1 : 1 + n_win]
        second_last = data[L - 2 : L - 2 + n_win]

        jump_val = (first - last) ** 2               # (n_win, n_vars)
        slip_val = ((last - second_last) - (second - first)) ** 2

        # Population std of each length-L window, per component.
        sd = np.empty((n_win, n_vars), dtype=np.float64)
        for c in range(n_vars):
            sd[:, c] = _sliding_pop_std(data[:, c], L)

        denom = L * sd * sd
        # Guard against constant windows (sd == 0): the jump/slip
        # contributions are 0 when sd is zero, so those components
        # contribute nothing.
        with np.errstate(divide="ignore", invalid="ignore"):
            xj_comp = np.where(denom > 0, jump_val / denom, 0.0)
            sj_comp = np.where(denom > 0, slip_val / denom, 0.0)

        xj = xj_comp.sum(axis=1)                      # (n_win,)
        sj = sj_comp.sum(axis=1)
        etot = wjump * xj + (1.0 - wjump) * sj

        idx = int(np.argmin(etot))
        if etot[idx] < best_etot:
            best_etot = float(etot[idx])
            best_length = L
            best_offset = idx
            best_jump = float(xj[idx])
            best_slip = float(sj[idx])

        # Stop as soon as a near-perfect match is found.
        if best_etot < 1e-5:
            break

    return {
        "length": int(best_length),
        "offset": int(best_offset),
        "lost": float((n - best_length) / n),
        "jump": best_jump,
        "slip": best_slip,
        "weighted": best_etot,
    }
