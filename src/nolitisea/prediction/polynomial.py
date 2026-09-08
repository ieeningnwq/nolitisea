"""Polynomial model fitting."""

from __future__ import annotations

import numpy as np

__all__ = [
    "fit_polynom",
    "fit_polynom_terms",
    "monomial_value",
    "polyback",
    "polypar",
]


# ---------------------------------------------------------------------------
# 1.  Monomial generation  (polypar)
# ---------------------------------------------------------------------------

def polypar(dim: int, degree: int) -> list[tuple[int, ...]]:
    """Generate every monomial exponent tuple with total degree <= ``degree``.

    returns

        [(0,0,0), (1,0,0), (2,0,0), (0,1,0), (1,1,0),
         (0,2,0), (0,0,1), (1,0,1), (0,1,1), (0,0,2)]

    Parameters
    ----------
    dim : int
        Embedding dimension (number of exponents per tuple).  Must be
        >= 1.
    degree : int
        Maximum total degree ``sum(e_i) <= degree``.  Must be >= 0.

    Returns
    -------
    list[tuple[int, ...]]
        Ordered list of length ``p = C(dim + degree, dim)``.

    Raises
    ------
    ValueError
        If ``dim < 1`` or ``degree < 0``.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")

    terms: list[tuple[int, ...]] = []

    def _recurse(cur: list[int], d: int, running_sum: int) -> None:
        if d < 0:
            terms.append(tuple(cur))
            return
        # Try exponent e for dimension d, with 0 <= e <= degree - running_sum.
        for e in range(degree - running_sum + 1):
            cur[d] = e
            _recurse(cur, d - 1, running_sum + e)

    _recurse([0] * dim, dim - 1, 0)
    return terms


# ---------------------------------------------------------------------------
# 2.  Monomial evaluation on a delay vector
# ---------------------------------------------------------------------------

def monomial_value(x: np.ndarray, exponents: tuple[int, ...]) -> float:
    """Evaluate a monomial ``prod_d x[d] ** exponents[d]``.

    Parameters
    ----------
    x : np.ndarray
        1-D delay vector of length ``dim``.
    exponents : tuple[int, ...]
        Exponent tuple of the same length.

    Returns
    -------
    float
        The monomial value.
    """
    result = 1.0
    for xd, e in zip(x, exponents):
        if e:
            result *= xd**e
    return result


def _build_design(series: np.ndarray, dim: int, delay: int,
                  terms: list[tuple[int, ...]], step: int,
                  n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the design matrix and target vector for OLS.

    The delay embedding convention: for base index
    ``n`` the embedding vector is ``[x(n), x(n-delay), ...,
    x(n-(dim-1)*delay)]``.  The target is ``x(n + step)``.

    Parameters
    ----------
    series : np.ndarray
        1-D float array.
    dim, delay, step : int
        Model parameters.
    terms : list[tuple[int, ...]]
        Monomial exponents.
    n_samples : int
        Number of rows to use (starting from index ``(dim-1)*delay``).

    Returns
    -------
    X : np.ndarray, shape (n_samples, len(terms))
        Design matrix.
    y : np.ndarray, shape (n_samples,)
        Target vector.
    bases : np.ndarray, shape (n_samples,)
        Original time index corresponding to each row of ``X``.
    """
    p = len(terms)
    X = np.empty((n_samples, p), dtype=np.float64)
    y = np.empty(n_samples, dtype=np.float64)
    offset = (dim - 1) * delay
    bases = np.arange(offset, offset + n_samples)

    # Pre-build the delay-embedded matrix E of shape (n_samples, dim)
    # where E[k, d] = series[bases[k] - d*delay]
    E = np.empty((n_samples, dim), dtype=np.float64)
    for d in range(dim):
        E[:, d] = series[bases - d * delay]
    # Target
    y = series[bases + step]

    # Fill each column of X using vectorised power products
    for j, exp in enumerate(terms):
        col = np.ones(n_samples, dtype=np.float64)
        for d, e in enumerate(exp):
            if e:
                col *= E[:, d] ** e
        X[:, j] = col

    return X, y, bases


def _solve_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ``X @ c = y`` for ``c``, returning the coefficient vector.

    Uses the normal equations ``X.T @ X @ c = X.T @ y`` with
    :func:`numpy.linalg.solve` (LU factorisation of the Gram matrix).
    """
    G = X.T @ X
    b = X.T @ y
    return np.linalg.solve(G, b)


def _rmse(series: np.ndarray, dim: int, delay: int, step: int,
          terms: list[tuple[int, ...]], coeffs: np.ndarray,
          start: int, end: int) -> float:
    """Root-mean-square one-step forecast error in ``[start, end)``.

    Uses the denominator convention
    ``(end - start) - (dim - 1) * delay - step``.
    """
    offset = (dim - 1) * delay
    if end <= start + offset + step:
        return float("nan")
    n_samples = end - start - offset - step
    X, y, _ = _build_design(series[start:end], dim, delay, terms, step,
                            n_samples)
    resid = y - X @ coeffs
    return float(np.sqrt(np.mean(resid * resid)))


# ---------------------------------------------------------------------------
# 3.  fit_polynom  (full monomial set, normalised)
# ---------------------------------------------------------------------------

def fit_polynom(series, dim, delay, degree, step=1, insample=None, cast_steps=0):
    """Fit a polynomial model using every monomial up to ``degree``.

    The procedure:

    1. Removes the sample mean and divides by the sample standard
       deviation.
    2. Fits the polynomial on the normalised series via OLS.
    3. Converts the coefficients back to the original scale so that
       predictions and forecasts are in physical units.

    Parameters
    ----------
    series : array_like
        1-D scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    degree : int
        Maximum polynomial degree.
    step : int, default 1
        Forecast horizon .
    insample : int or None, default None
        Number of data points used for fitting (``-n``).  ``None``
        (the default) means "use the whole series".
    cast_steps : int, default 0
        If > 0, perform ``cast_steps`` steps of iterated prediction
        after the fit and include them in the returned dict under the
        key ``"casted"``.

    Returns
    -------
    dict
        Keys:

        - ``"terms"`` : list[tuple[int, ...]] — the monomial tuples.
        - ``"coeffs"`` : np.ndarray — OLS coefficients (original scale).
        - ``"mean"`` : float — sample mean subtracted.
        - ``"std_dev"`` : float — sample standard deviation divided out.
        - ``"in_sample_rmse"`` : float — in-sample RMSE (normalised).
        - ``"out_of_sample_rmse"`` : float or None — out-of-sample
          RMSE when ``insample`` is smaller than the series length.
        - ``"casted"`` : np.ndarray, shape (cast_steps,) — only when
          ``cast_steps > 0``.
    """
    s = np.asarray(series, dtype=np.float64).ravel()
    if s.ndim != 1:
        raise ValueError("series must be 1-D")

    n = len(s)
    if n <= (dim - 1) * delay + step + 1:
        raise ValueError(
            f"series of length {n} is too short for dim={dim}, "
            f"delay={delay}, step={step}"
        )

    # --- number of points used for fitting ---------------------------------
    if insample is None or insample > n:
        insample = n
    if insample <= (dim - 1) * delay + step:
        raise ValueError(
            f"insample={insample} is too short for dim={dim}, delay={delay}, step={step}"
        )

    # --- normalisation -----------------------------------------------------
    # std_dev = sqrt(mean((x - mean)^2)), then series[i] /= std_dev.
    # Mean is NOT removed.
    mean = float(np.mean(s))
    var_total = float(np.mean((s - mean) ** 2))
    std_dev = np.sqrt(var_total) if var_total > 0 else 1.0
    if std_dev < 1e-300:
        std_dev = 1.0
    sn = s / std_dev

    # --- terms and design --------------------------------------------------
    terms = polypar(dim, degree)
    offset = (dim - 1) * delay
    n_fit = insample - offset - step
    X, y_norm, _ = _build_design(sn, dim, delay, terms, step, n_fit)
    coeffs_norm = _solve_ols(X, y_norm)

    # --- coefficient de-normalisation --------------------------------------
    # y_norm = x_orig / std_dev,  X_norm_j = prod_d (x_orig_d / std_dev) ** e_{j,d}
    #                                     = prod x_orig ** e / std_dev ** sum(e_j)
    # y_norm ≈ sum_j c_norm_j * X_norm_j
    #   ⇒ x_orig ≈ std_dev * sum_j c_norm_j * X_norm_j
    #             = sum_j c_norm_j / std_dev ** (sum(e_j) - 1)
    #               * prod_d x_orig ** e_{j,d}
    # which is exactly c_orig_j = c_norm_j / std_dev ** (sum(e_j) - 1).
    coeffs = np.empty_like(coeffs_norm)
    for j, exp in enumerate(terms):
        total_exp = sum(exp)
        coeffs[j] = coeffs_norm[j] / (std_dev ** (total_exp - 1))

    # --- in-sample RMSE (in normalised units) -------------------------------
    resid = y_norm - X @ coeffs_norm
    in_rmse = float(np.sqrt(np.mean(resid * resid)))

    # --- out-of-sample RMSE -----------------------------------------------
    out_rmse = None
    if insample < n:
        out_rmse = _rmse(sn, dim, delay, step, terms, coeffs_norm, insample, n)

    # --- optional casted series --------------------------------------------
    casted = None
    if cast_steps > 0:
        casted = _do_cast(s, dim, delay, terms, coeffs, cast_steps)

    result = {
        "terms": terms,
        "coeffs": coeffs,
        "mean": mean,
        "std_dev": std_dev,
        "in_sample_rmse": in_rmse,
        "out_of_sample_rmse": out_rmse,
    }
    if casted is not None:
        result["casted"] = casted
    return result


# ---------------------------------------------------------------------------
# 4.  fit_polynom_terms  (specified terms, no normalisation)
# ---------------------------------------------------------------------------

def fit_polynom_terms(series, dim, delay, terms, step=1,
                      insample=None, cast_steps=0, variance=None):
    """Fit a polynomial model using a caller-supplied term list.

    No mean removal, no standard-deviation normalisation — 
    raw OLS on the physical series.

    Parameters
    ----------
    series : array_like
        1-D scalar series.
    dim, delay : int
        Embedding parameters.
    terms : list[tuple[int, ...]]
        Monomial exponent tuples (length ``p``).
    step : int, default 1
        Forecast horizon.
    insample : int or None, default None
        Number of data points used for fitting.  ``None`` means whole
        series.
    cast_steps : int, default 0
        If > 0, include ``cast_steps`` steps of iterated prediction in
        the result under ``"casted"``.
    variance : float or None, default None
        Optional series variance used only for the FCE-norm output in
        the returned dict.  If ``None`` it is computed from the series.

    Returns
    -------
    dict
        Keys:

        - ``"terms"`` : list[tuple[int, ...]] — same as input.
        - ``"coeffs"`` : np.ndarray — OLS coefficients.
        - ``"in_sample_rmse"`` : float
        - ``"out_of_sample_rmse"`` : float or None
        - ``"in_sample_fce"`` : float — in-sample RMSE / sqrt(variance)
        - ``"out_of_sample_fce"`` : float or None
        - ``"variance"`` : float — series variance used for FCE.
        - ``"casted"`` : np.ndarray, shape (cast_steps,) — only when
          ``cast_steps > 0``.
    """
    s = np.asarray(series, dtype=np.float64).ravel()
    if s.ndim != 1:
        raise ValueError("series must be 1-D")
    if not terms:
        raise ValueError("terms must be a non-empty list")

    n = len(s)
    if n <= (dim - 1) * delay + step + 1:
        raise ValueError(
            f"series of length {n} is too short for dim={dim}, "
            f"delay={delay}, step={step}"
        )

    if insample is None or insample > n:
        insample = n
    if insample <= (dim - 1) * delay + step:
        raise ValueError(
            f"insample={insample} is too short for dim={dim}, delay={delay}, step={step}"
        )

    offset = (dim - 1) * delay
    n_fit = insample - offset - step
    X, y, _ = _build_design(s, dim, delay, terms, step, n_fit)
    coeffs = _solve_ols(X, y)

    # RMSE in physical units
    resid = y - X @ coeffs
    in_rmse = float(np.sqrt(np.mean(resid * resid)))
    out_rmse = None
    if insample < n:
        out_rmse = _rmse(s, dim, delay, step, terms, coeffs, insample, n)

    if variance is None:
        variance = float(np.mean((s - np.mean(s)) ** 2))
    stdv = float(np.sqrt(variance))
    in_fce = in_rmse / stdv if stdv > 0 else float("nan")
    out_fce = None
    if out_rmse is not None:
        out_fce = out_rmse / stdv if stdv > 0 else float("nan")

    casted = None
    if cast_steps > 0:
        casted = _do_cast(s, dim, delay, terms, coeffs, cast_steps)

    result = {
        "terms": terms,
        "coeffs": coeffs,
        "in_sample_rmse": in_rmse,
        "out_of_sample_rmse": out_rmse,
        "in_sample_fce": in_fce,
        "out_of_sample_fce": out_fce,
        "variance": variance,
    }
    if casted is not None:
        result["casted"] = casted
    return result


# ---------------------------------------------------------------------------
# 5.  Iterated forecasting used by both fitters
# ---------------------------------------------------------------------------

def _do_cast(series: np.ndarray, dim: int, delay: int,
             terms: list[tuple[int, ...]], coeffs: np.ndarray,
             n_steps: int) -> np.ndarray:
    """Produce ``n_steps`` of iterated single-step predictions.

    The forecast is seeded with the last ``(dim - 1) * delay + 1``
    values of the input series and each predicted value is pushed
    back onto the left end of the delay window.
    """
    cast = np.empty(n_steps, dtype=np.float64)
    # Working buffer: hold the last (dim-1)*delay+1 values, oldest at
    # index 0, newest at index (dim-1)*delay.  After each prediction
    # we shift left and append the new value.
    buf = np.empty((dim - 1) * delay + 1, dtype=np.float64)
    buf[:] = series[-(dim - 1) * delay - 1:]

    for step_idx in range(n_steps):
        # Delay vector x_d = buf[offset - d*delay] where offset = (dim-1)*delay
        offset = (dim - 1) * delay
        x = np.empty(dim, dtype=np.float64)
        for d in range(dim):
            x[d] = buf[offset - d * delay]
        pred = 0.0
        for j, exp in enumerate(terms):
            mon = 1.0
            for xd, e in zip(x, exp):
                if e:
                    mon *= xd**e
            pred += coeffs[j] * mon
        cast[step_idx] = pred
        # Shift buffer left by one, insert pred at the rightmost slot
        buf[:-1] = buf[1:]
        buf[-1] = pred

    return cast


# ---------------------------------------------------------------------------
# 6.  polyback  (backward elimination)
# ---------------------------------------------------------------------------

def polyback(series, dim, delay, degree, down_to=1, step=1,
             insample=None):
    """Backward-elimination polynomial fitting.

    Starts with every monomial up to ``degree`` and greedily removes
    one term at a time.  At each step the candidate term whose
    deletion yields the **smallest increase** (or largest decrease)
    in the chosen error metric is dropped.  When an out-of-sample
    split is available (``insample < len(series)``) the decision is
    based on the out-of-sample RMSE.  Otherwise the in-sample RMSE is
    used.

    Parameters
    ----------
    series : array_like
        1-D scalar series.
    dim, delay : int
        Embedding parameters.
    degree : int
        Maximum polynomial degree (initial full-model generator).
    down_to : int, default 1
        Stop when only ``down_to`` terms remain.  Must be >= 1 and
        <= the initial number of terms.
    step : int, default 1
        Forecast horizon.
    insample : int or None, default None
        Size of the fitting window.  When ``None`` the whole series is
        used and only the in-sample RMSE guides elimination.

    Returns
    -------
    dict
        Keys:

        - ``"initial"`` : result of the full fit (from
          :func:`fit_polynom_terms`).
        - ``"path"`` : list[dict] — one entry per elimination step,
          each with ``"n_terms"``, ``"removed"`` (tuple),
          ``"in_rmse"``, ``"out_rmse"`` (or None).
        - ``"final"`` : result of the final reduced fit.
        - ``"decision_error"`` : ``"out_of_sample"`` or
          ``"in_sample"`` indicating which RMSE drove elimination.
    """
    s = np.asarray(series, dtype=np.float64).ravel()
    if s.ndim != 1:
        raise ValueError("series must be 1-D")

    n = len(s)
    if insample is None or insample > n:
        insample = n
    out_set = insample < n
    decision_err = "out_of_sample" if out_set else "in_sample"

    full_terms = polypar(dim, degree)
    if not full_terms:
        raise ValueError("no terms generated (check dim and degree)")
    if down_to < 1 or down_to > len(full_terms):
        raise ValueError(
            f"down_to must satisfy 1 <= down_to <= {len(full_terms)}, got {down_to}"
        )

    initial_result = fit_polynom_terms(
        s, dim, delay, full_terms, step=step, insample=insample
    )

    current_terms = list(full_terms)
    path: list[dict] = []

    for _ in range(len(full_terms) - down_to):
        best_idx = -1
        best_err = float("inf")
        best_in_rmse = None
        best_out_rmse = None

        for idx in range(len(current_terms)):
            trial_terms = current_terms[:idx] + current_terms[idx + 1:]
            try:
                trial = fit_polynom_terms(
                    s, dim, delay, trial_terms, step=step, insample=insample
                )
            except np.linalg.LinAlgError:
                # Singular Gram matrix — skip this candidate
                continue

            if out_set:
                err = trial["out_of_sample_rmse"]
            else:
                err = trial["in_sample_rmse"]
            if err is None or not np.isfinite(err):
                continue

            if best_idx == -1 or err < best_err:
                best_err = err
                best_idx = idx
                best_in_rmse = trial["in_sample_rmse"]
                best_out_rmse = trial["out_of_sample_rmse"]

        if best_idx == -1:
            raise RuntimeError(
                "backward elimination stalled: all candidate deletions "
                "produced singular Gram matrices"
            )

        removed_term = current_terms[best_idx]
        current_terms = current_terms[:best_idx] + current_terms[best_idx + 1:]

        path.append({
            "n_terms": len(current_terms),
            "removed": removed_term,
            "in_rmse": best_in_rmse,
            "out_rmse": best_out_rmse,
        })

    final_result = fit_polynom_terms(
        s, dim, delay, current_terms, step=step, insample=insample
    )

    return {
        "initial": initial_result,
        "path": path,
        "final": final_result,
        "decision_error": decision_err,
    }
