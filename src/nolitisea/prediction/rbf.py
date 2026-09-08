"""Radial basis function prediction (TISEAN ``rbf``).

Fits a Gaussian radial basis function model to a scalar time series
using delay embedding.  The fitted model has the form::

    y_{n+step} = c_0 + sum_{i=1}^{n_centers} c_i * phi( x_n - center_i )

where ``phi(r) = exp(-r^2 / (2 * eps^2))`` and ``x_n`` is the
delay-embedded vector ``[x(n), x(n-delay), ..., x(n-(dim-1)*delay)]``.

This module follows the normalisation convention used by the TISEAN
C program: the series is first mapped linearly to ``[0, 1]`` so
that all computed quantities (centres, weights, errors) live on
the rescaled attractor, and physical units are restored at the
output stage.
"""

from __future__ import annotations

import numpy as np

from ..utils.rescale import rescale_data

__all__ = ["fit_rbf", "predict_rbf", "rbf_forecast_error"]


# ---------------------------------------------------------------------------
# 1.  Centre initialisation  (mirrors TISEAN main loop lines 301-304)
# ---------------------------------------------------------------------------

def _init_centers(series: np.ndarray, dim: int, delay: int,
                  n_centers: int) -> np.ndarray:
    """Initialise RBF centres by sampling the delay-embedded trajectory.

    The first centre is placed at index ``(dim-1)*delay`` and the last
    at ``LENGTH-1``; intermediate centres are spaced uniformly by
    integer division of ``cstep = LENGTH-1-(dim-1)*delay``, exactly
    matching the C code's ``(i*cstep)/(CENTER-1)`` pattern.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        *Rescaled* series in ``[0, 1]``.
    dim, delay : int
        Embedding parameters.
    n_centers : int
        Number of centres.

    Returns
    -------
    centers : np.ndarray, shape (n_centers, dim)
    """
    n = len(series)
    offset = (dim - 1) * delay
    cstep = n - 1 - offset  # total span from first to last embedding point

    centers = np.empty((n_centers, dim), dtype=np.float64)
    for i in range(n_centers):
        # Integer division, exactly as C (i*cstep)/(CENTER-1)
        base = offset + (i * cstep) // (n_centers - 1)
        for j in range(dim):
            centers[i, j] = series[base - j * delay]
    return centers


# ---------------------------------------------------------------------------
# 2.  Drift / repulsive centre movement  (mirrors C drift(), 20 iterations)
# ---------------------------------------------------------------------------

def _drift_centers(centers: np.ndarray, n_iter: int = 20,
                   step0: float = 1e-2) -> np.ndarray:
    """Push centres apart so no two overlap (optional TISEAN step).

    The force on centre ``i`` in dimension ``j`` is::

        F_j = sum_{k != i}  sign(h) / h^2,   h = centers[i,j] - centers[k,j]

    (the C code writes ``h / sqr(h) / fabs(h)`` which reduces to the
    same expression).  Centres are kept inside a loose ``[-0.1, 1.1]``
    bounding box to avoid degenerate drift.

    Parameters
    ----------
    centers : np.ndarray, shape (n_centers, dim)
        *Rescaled* centres in ``[0, 1]``.  Modified in place.
    n_iter : int, default 20
        Number of drift sweeps (TISEAN uses 20).
    step0 : float, default 1e-2
        Base step size.

    Returns
    -------
    centers : np.ndarray
        Same object as input, modified in place.
    """
    nc, d = centers.shape
    for _ in range(n_iter):
        for i in range(nc):
            force = np.zeros(d, dtype=np.float64)
            ci = centers[i]
            for k in range(nc):
                if k == i:
                    continue
                h = ci - centers[k]
                # h / h^2 / |h| = sign(h) / h^2  (vectorised)
                force += np.sign(h) / (h * h + 1e-300)  # +eps prevents div-by-zero
            h_total = np.sqrt(np.dot(force, force))
            if h_total < 1e-300:
                continue
            step1 = step0 / h_total
            delta = step1 * force
            # Per-dimension boundary check (C code handles each dim independently)
            for j in range(d):
                new_val = ci[j] + delta[j]
                if -0.1 < new_val < 1.1:
                    ci[j] = new_val
    return centers


# ---------------------------------------------------------------------------
# 3.  RBF width estimation via average centre distance  (mirrors C avdistance())
# ---------------------------------------------------------------------------

def _avdistance(centers: np.ndarray) -> float:
    """Mean Euclidean distance between all centre pairs, per dimension.

    The C implementation accumulates ``sum_{i!=j} sum_k (c_ik - c_jk)^2``
    and divides by ``(nc-1) * nc * dim`` before taking the square root.

    Parameters
    ----------
    centers : np.ndarray, shape (n_centers, dim)

    Returns
    -------
    float
        Average centre distance used as the RBF width.
    """
    nc, d = centers.shape
    total = 0.0
    for i in range(nc):
        diffs = centers[i + 1:] - centers[i]  # shape (nc-1-i, d)
        total += np.sum(diffs * diffs)
    # Each pair counted twice implicitly — C's double loop does the
    # same but divides by (nc-1)*nc instead of 2*(nc-1 choose 2).
    # Let's verify: C sums over all i!=j, so total_pairs_contribution = 2*above
    # and denominator = (nc-1)*nc.  So total * 2 / ((nc-1)*nc*d)
    return float(np.sqrt(2.0 * total / ((nc - 1) * nc * d)))


# ---------------------------------------------------------------------------
# 4.  Gaussian RBF evaluation
# ---------------------------------------------------------------------------

def _rbf(x_vec: np.ndarray, center: np.ndarray, eps: float) -> float:
    """Evaluate ``exp(- ||x_vec - center||^2 / (2 * eps^2))``."""
    diff = x_vec - center
    return float(np.exp(-np.dot(diff, diff) / (2.0 * eps * eps)))


def _delay_vector(series: np.ndarray, n: int, dim: int, delay: int) -> np.ndarray:
    """Return ``[x(n), x(n-delay), ..., x(n-(dim-1)*delay)]``."""
    idx = n - np.arange(dim) * delay
    return series[idx]


# ---------------------------------------------------------------------------
# 5.  Main fit  (mirrors C make_fit() + main orchestration)
# ---------------------------------------------------------------------------

def fit_rbf(series, dim, delay, n_centers, eps=None, step=1,
            insample=None, drift=True):
    """Fit a Gaussian radial basis function model.

    Parameters
    ----------
    series : array_like
        1-D scalar input series (physical units).
    dim : int
        Embedding dimension ``m`` (TISEAN ``-m``).
    delay : int
        Time delay ``d`` (TISEAN ``-d``).
    n_centers : int
        Number of RBF centres ``p`` (TISEAN ``-p``).
    eps : float or None, default None
        RBF width.  When ``None`` it is set to the average centre
        distance (TISEAN default ``avdistance``).
    step : int, default 1
        Forecast horizon ``s`` (TISEAN ``-s``).
    insample : int or None, default None
        Number of data points used for fitting.  ``None`` means the
        whole series (TISEAN ``-n``).
    drift : bool, default True
        Whether to apply the repulsive drift step to centres
        (TISEAN default; pass ``False`` to mirror ``-X``).

    Returns
    -------
    dict
        Fitted model with keys:

        - ``"centers"`` : np.ndarray, shape (n_centers, dim) — rescaled centres
        - ``"coeffs"`` : np.ndarray, shape (n_centers + 1,) — OLS coefficients
          (``coeffs[0]`` is the bias / constant term, the rest are
          RBF weights).
        - ``"eps"`` : float — RBF width used.
        - ``"minv"`` : float — original minimum (for inverse scaling).
        - ``"interval"`` : float — original range (max-min).
        - ``"in_sample_rmse"`` : float — in-sample RMSE (rescaled units).
        - ``"out_of_sample_rmse"`` : float or None.
        - ``"in_sample_fce"`` : float — normalised in-sample error.
        - ``"out_of_sample_fce"`` : float or None.
    """
    s = np.asarray(series, dtype=np.float64).ravel()
    if s.ndim != 1:
        raise ValueError("series must be 1-D")

    n = len(s)
    offset = (dim - 1) * delay
    if n <= offset + step + 1:
        raise ValueError(
            f"series of length {n} is too short for dim={dim}, "
            f"delay={delay}, step={step}"
        )

    if n_centers < 2:
        raise ValueError("n_centers must be >= 2")
    if n_centers > n - offset:
        raise ValueError(
            f"n_centers={n_centers} is too large for length={n}, offset={offset}"
        )

    if insample is None or insample > n:
        insample = n
    if insample <= offset + step:
        raise ValueError(
            f"insample={insample} is too short for dim={dim}, delay={delay}, step={step}"
        )

    # --- rescale to [0, 1] (exact match with TISEAN rescale_data) --------
    s_resc, minv, interval = rescale_data(s)

    # --- centre initialisation -------------------------------------------
    centers = _init_centers(s_resc, dim, delay, n_centers)

    # --- drift step (optional) -------------------------------------------
    if drift:
        _drift_centers(centers)

    # --- RBF width -------------------------------------------------------
    if eps is None:
        eps = _avdistance(centers)
        if eps < 1e-300:
            eps = 1.0  # safety fallback

    # --- build design matrix and solve normal equations ------------------
    # Effective sample indices: n from offset to insample-step-1 inclusive
    n_base_start = offset
    n_base_end = insample - step  # exclusive
    n_samples = n_base_end - n_base_start
    if n_samples <= 0:
        raise ValueError("no valid samples after embedding offset and step")

    # Precompute RBF values for all centre pairs
    hcen = np.empty((n_samples, n_centers), dtype=np.float64)
    for k, n_idx in enumerate(range(n_base_start, n_base_end)):
        x_vec = _delay_vector(s_resc, n_idx, dim, delay)
        for j in range(n_centers):
            hcen[k, j] = _rbf(x_vec, centers[j], eps)

    y_target = s_resc[n_base_start + step: n_base_end + step]  # length n_samples

    # Build Gram matrix and rhs — TISEAN's exact normal equation layout:
    #   mat[0,0] = sum(1); mat[1:,0] = sum(hcen[:,j]);
    #   mat[i,j] for i,j>=1 = sum(hcen[:,i-1] * hcen[:,j-1])
    #   coefs[0] = sum(y); coefs[1:] = sum(y * hcen[:,j])
    p = n_centers + 1
    mat = np.empty((p, p), dtype=np.float64)
    rhs = np.empty(p, dtype=np.float64)

    mat[0, 0] = n_samples
    rhs[0] = np.sum(y_target)
    mat[1:, 0] = np.sum(hcen, axis=0)
    rhs[1:] = hcen.T @ y_target

    # Gram block (vectorised — replaces the double loop over i, j)
    mat[1:, 1:] = hcen.T @ hcen
    mat[0, 1:] = mat[1:, 0]  # symmetrise first row/column

    # Solve via numpy.linalg.solve (matches TISEAN solvele LU decomp)
    try:
        coefs = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "RBF normal matrix is singular — increase eps or use fewer centres"
        ) from exc

    # --- errors (in-sample + optional out-of-sample) ----------------------
    in_rmse = rbf_forecast_error(s_resc, coefs, centers, eps,
                                 dim, delay, step, 0, insample)
    out_rmse = None
    if insample < n:
        out_rmse = rbf_forecast_error(s_resc, coefs, centers, eps,
                                      dim, delay, step, insample, n)

    # FCE = RMSE / sigma — C code computes sigma from *rescaled* in-sample
    # (variance was already overwritten by avdistance, but in the
    #  RMSE denominator it recomputes sigma from the insample slice).
    _in = s_resc[:insample]
    in_sigma = float(np.sqrt(np.mean((_in - np.mean(_in)) ** 2)))
    in_fce = in_rmse / in_sigma if in_sigma > 0 else float("nan")
    out_fce = None
    if out_rmse is not None:
        _out = s_resc[insample:]
        out_sigma = float(np.sqrt(np.mean((_out - np.mean(_out)) ** 2)))
        out_fce = out_rmse / out_sigma if out_sigma > 0 else float("nan")

    return {
        "centers": centers,
        "coeffs": coefs,
        "eps": eps,
        "minv": minv,
        "interval": interval,
        "dim": dim,
        "delay": delay,
        "step": step,
        "in_sample_rmse": in_rmse,
        "out_of_sample_rmse": out_rmse,
        "in_sample_fce": in_fce,
        "out_of_sample_fce": out_fce,
    }


# ---------------------------------------------------------------------------
# 6.  Forecast error helper
# ---------------------------------------------------------------------------

def rbf_forecast_error(series: np.ndarray, coefs: np.ndarray,
                       centers: np.ndarray, eps: float,
                       dim: int, delay: int, step: int,
                       i0: int, i1: int) -> float:
    """Root-mean-square ``step``-step forecast error in ``[i0, i1)``.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        *Rescaled* series.
    coefs : np.ndarray, shape (n_centers + 1,)
    centers : np.ndarray, shape (n_centers, dim)
    eps : float
    dim, delay, step : int
    i0, i1 : int
        Start and exclusive end of the interval to evaluate.

    Returns
    -------
    float
        RMSE (rescaled units), or NaN if there are no valid samples.
    """
    offset = (dim - 1) * delay
    n_start = i0 + offset
    n_end = i1 - step
    if n_start >= n_end:
        return float("nan")
    nc = centers.shape[0]

    error = 0.0
    for n_idx in range(n_start, n_end):
        x_vec = _delay_vector(series, n_idx, dim, delay)
        pred = coefs[0]
        for j in range(nc):
            pred += coefs[j + 1] * _rbf(x_vec, centers[j], eps)
        target = series[n_idx + step]
        error += (target - pred) ** 2

    denom = i1 - i0 - step - offset
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(error / denom))


# ---------------------------------------------------------------------------
# 7.  Iterated forecasting  (mirrors C make_cast())
# ---------------------------------------------------------------------------

def predict_rbf(model, series, n_steps):
    """Iterate a fitted RBF model ``n_steps`` into the future.

    The forecast is seeded with the last ``(dim-1)*delay + 1`` values
    of the input series.  Each predicted value is fed back as the
    newest entry of the delay window.

    Parameters
    ----------
    model : dict
        Model returned by :func:`fit_rbf`.  Must contain keys
        ``"coeffs"``, ``"centers"``, ``"eps"``, ``"minv"``,
        ``"interval"``.
    series : array_like
        1-D scalar series in *physical units* (any length >= 1).
    n_steps : int
        Number of future values to produce.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_steps,)`` with forecast values in the
        original (physical) units.
    """
    if n_steps < 1:
        return np.array([], dtype=np.float64)

    coeffs = model["coeffs"]
    centers = model["centers"]
    eps = model["eps"]
    minv = model["minv"]
    interval = model["interval"]
    dim = model["dim"]
    delay = model["delay"]

    # Rescale seed series using the same parameters as fit
    s = np.asarray(series, dtype=np.float64).ravel()
    s_resc = (s - minv) / interval

    # Working buffer: last (dim-1)*delay+1 values, oldest at index 0,
    # newest at index (dim-1)*delay (TISEAN convention).
    buf_len = (dim - 1) * delay + 1
    buf = np.empty(buf_len, dtype=np.float64)
    if len(s_resc) < buf_len:
        # Pad with zeros (scaled space) — this matches C behaviour when
        # the cast buffer is larger than the input; we simply take what's available
        buf[:] = 0.0
        buf[buf_len - len(s_resc):] = s_resc
    else:
        buf[:] = s_resc[-buf_len:]

    dim_offset = (dim - 1) * delay
    casted = np.empty(n_steps, dtype=np.float64)
    nc = centers.shape[0]

    for step_idx in range(n_steps):
        # Extract delay vector [buf[dim_offset], buf[dim_offset-delay], ..., buf[0]]
        x_vec = np.empty(dim, dtype=np.float64)
        for d in range(dim):
            x_vec[d] = buf[dim_offset - d * delay]

        pred = coeffs[0]
        for j in range(nc):
            pred += coeffs[j + 1] * _rbf(x_vec, centers[j], eps)

        # Save in physical units, update buffer in rescaled units
        casted[step_idx] = pred * interval + minv

        # Shift buffer left by one, append pred
        buf[:-1] = buf[1:]
        buf[-1] = pred

    return casted
