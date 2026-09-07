"""Multivariate ARIMA model fitting and iteration."""

import numpy as np

__all__ = ["difference", "fit_arima", "iterate_arima_model"]


def difference(series, d):
    """Apply ``d``-th order differencing along time (the I step).

    Parameters
    ----------
    series : array_like
        Input data of shape ``(N,)`` or ``(N, dim)``.
    d : int
        Differencing order (>= 0).  ``d == 0`` returns a copy.

    Returns
    -------
    numpy.ndarray
        Differenced series of shape ``(N - d,)`` or ``(N - d, dim)``.

    Raises
    ------
    ValueError
        If ``d`` is negative or the series is too short.
    """
    s = np.asarray(series, dtype=np.float64)
    if s.ndim not in (1, 2):
        raise ValueError("series must be a 1-D or 2-D array")
    if d < 0:
        raise ValueError(f"d must be >= 0, got {d}")
    if s.shape[0] <= d:
        raise ValueError(f"series of length {s.shape[0]} is too short for d={d}")
    if d == 0:
        return s.copy()
    return np.diff(s, n=d, axis=0)


def _regressor_index(ar_order, ma_order, dim):
    """Regressor table ``(size, 3)`` with rows ``[kind, source, lag]``.

    ``kind = 0`` marks lagged data (AR part, sources ``0..dim-1``),
    ``kind = 1`` marks lagged innovation estimates (MA part, sources
    ``dim..2*dim-1`` of the augmented history).
    """
    rows = [(0, c, j) for c in range(dim) for j in range(ar_order)]
    rows += [(1, dim + c, j) for c in range(dim) for j in range(ma_order)]
    return np.array(rows, dtype=np.intp).reshape(-1, 3)


def _solve_model(h, index, n_start, length, dim):
    """One least-squares pass.

    Regressors are read from the augmented history ``h`` (first ``dim``
    rows: data, last ``dim`` rows: innovation estimates) at times
    ``n - lag`` for ``n`` in ``[n_start, length - 2]``; the target is
    the data at ``n + 1``.  Returns coefficients of shape ``(dim, size)``
    (row = output component, column = regressor).
    """
    size = index.shape[0]
    n_count = length - 1 - n_start
    if n_count < 1:
        raise ValueError(
            f"series of length {length} is too short for this ARIMA fit "
            f"(the regression sample range is empty); reduce the model "
            f"orders or the number of iterations"
        )
    design = np.empty((n_count, size))
    for i in range(size):
        src, lag = index[i, 1], index[i, 2]
        design[:, i] = h[src, n_start - lag : n_start - lag + n_count]
    target = h[:dim, n_start + 1 : n_start + 1 + n_count].T
    solved = np.linalg.solve(design.T @ design, design.T @ target)  # (size, dim)
    return solved.T


def _residuals(h, index, coeff, poles, length, dim):
    """One-step residuals.

    Returns ``(diff, forecast_errors)`` where ``diff`` has shape
    ``(dim, length)`` (zeros before index ``poles``) and
    ``forecast_errors`` is the per-component residual RMS over
    ``[poles, length)``.
    """
    n_res = length - poles
    diff = np.zeros((dim, length))
    pred = np.zeros((dim, n_res))
    for i in range(index.shape[0]):
        src, lag = index[i, 1], index[i, 2]
        pred += (
            coeff[:, i : i + 1]
            * h[src, poles - 1 - lag : poles - 1 - lag + n_res][None, :]
        )
    diff[:, poles:] = h[:dim, poles:] - pred
    forecast_errors = np.sqrt(np.sum(diff[:, poles:] ** 2, axis=1) / n_res)
    return diff, forecast_errors


def fit_arima(
    series,
    poles=10,
    ar_order=0,
    diff_order=0,
    ma_order=0,
    max_iter=50,
    convergence=1e-3,
):
    """Fit a multivariate ARIMA model.

    The model works on the ``diff_order``-th difference of the input
    with per-component means subtracted.  An initial AR fit of order
    ``poles`` is always computed; if ``ar_order + ma_order > 0`` it is
    refined by the iterative ARMA scheme: lagged innovation
    estimates (the residuals of the previous pass) enter the
    regression as extra regressors and the fit is repeated until the
    residual RMS change falls below ``convergence`` or ``max_iter``
    iterations are reached (Hannan-Rissanen style).  The one-step model
    is::

        y_d(t) = sum_(i, j) a_(d,i,j) * y_i(t-1-j)
               + sum_(i, j) m_(d,i,j) * e_i(t-1-j) + e_d(t)

    with ``e`` the innovation (residual) estimates.

    Parameters
    ----------
    series : array_like
        Input data of shape ``(N,)`` (univariate) or ``(N, dim)``
        (multivariate; components are columns).
    poles : int, default 10
        Order of the initial AR fit .  Must satisfy
        ``1 <= poles < N - diff_order``.
    ar_order : int, default 0
        AR order of the ARIMA refinement .
    diff_order : int, default 0
        Differencing order .
    ma_order : int, default 0
        MA order of the ARIMA refinement.
    max_iter : int, default 50
        Maximum number of ARIMA refinement iterations.
    convergence : float, default 1e-3
        Stop when the maximum per-component RMS change of the residuals
        between two iterations drops below this value.

    Returns
    -------
    dict
        ``"coeffs"`` : coefficient matrix, shape ``(dim, size)`` with
        ``size = poles * dim`` (pure AR mode) or ``(ar_order +
        ma_order) * dim`` (refinement); column order is given by
        ``"regressors"``.
        ``"regressors"`` : int array ``(size, 3)`` of ``[kind, source,
        lag]``; ``kind = 0`` is a lagged data regressor ``x_source(n-1-lag)``,
        ``kind = 1`` a lagged innovation regressor ``e_(source-dim)(n-1-lag)``.
        ``"residuals"`` : one-step prediction errors of the differenced
        series, shape ``(N - diff_order - poles_out, dim)``.
        ``"forecast_errors"`` : residual RMS per component, shape ``(dim,)``.
        ``"average_forecast_error"`` : RMS across components (float).
        ``"log_likelihood"`` : Gaussian log-likelihood of the fit.
        ``"aic"`` : ``2 * k - 2 * log_likelihood`` with ``k = ar_order +
        ma_order`` (refinement) or ``poles`` (pure AR).
        ``"mean"`` : per-component mean of the differenced series.
        ``"diff_order"`` : differencing order ``d`` the model was
        fitted with.
        ``"diff_anchors"`` : array of shape ``(d, dim)``; row ``k``
        holds the last observed value of the ``k``-th difference of
        the input (row 0 is the last point of the original series).
        Used by :func:`iterate_arima_model` with ``restore_level=True``
        to undifference the simulation; empty for ``d == 0``.
        ``"poles"`` : ``max(ar_order, ma_order)`` (refinement) or
        ``poles`` (pure AR); state length used by
        :func:`iterate_arima_model` and first valid residual index.
        ``"n_iterations"`` : number of refinement iterations performed
        (0 in pure-AR mode).
        ``"residual_rms_change"`` : per-component RMS change of the
        residuals per iteration, shape ``(n_iterations, dim)``.
        ``"coeff_rms_change"`` : RMS change of all coefficients per
        iteration, shape ``(n_iterations,)``.

    Raises
    ------
    ValueError
        For invalid parameters, a series too short for the requested
        orders/iterations, or a vanishing forecast error (the
        log-likelihood would be undefined).
    numpy.linalg.LinAlgError
        If a regression normal-equation matrix is singular.

    References
    ----------
    .. [1] Hegger, R., Kantz, H., & Schreiber, T. (1999).  Practical
           implementation of nonlinear time series methods: The TISEAN
           package.  *Chaos*, 9(2), 413-435.
    """
    s = np.asarray(series, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    if s.ndim != 2:
        raise ValueError("series must be a 1-D or 2-D array")
    if s.shape[0] < 1 or s.shape[1] < 1:
        raise ValueError("series must not be empty")
    if diff_order < 0:
        raise ValueError(f"diff_order must be >= 0, got {diff_order}")
    if diff_order >= s.shape[0]:
        raise ValueError(
            f"series of length {s.shape[0]} is too short for diff_order={diff_order}"
        )
    # Differencing (the I step); record the last value of each
    # difference level (0 .. diff_order - 1) so that the simulation
    # can be undifferenced back to the original level.
    diff_anchors = np.empty((diff_order, s.shape[1]))
    cur = s
    for k in range(diff_order):
        diff_anchors[k] = cur[-1]
        cur = np.diff(cur, axis=0)
    s = cur
    length, dim = s.shape

    if poles < 1:
        raise ValueError(f"poles must be >= 1, got {poles}")
    if poles >= length:
        raise ValueError(
            f"poles ({poles}) must be smaller than the differenced series "
            f"length ({length})"
        )
    if ar_order < 0:
        raise ValueError(f"ar_order must be >= 0, got {ar_order}")
    if ma_order < 0:
        raise ValueError(f"ma_order must be >= 0, got {ma_order}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if not np.isfinite(convergence) or convergence <= 0:
        raise ValueError(f"convergence must be finite and > 0, got {convergence}")

    refine = (ar_order + ma_order) > 0
    if refine and (ar_order >= length or ma_order >= length):
        raise ValueError(
            f"ar_order ({ar_order}) and ma_order ({ma_order}) must be "
            f"smaller than the differenced series length ({length})"
        )

    mean = s.mean(axis=0)
    x = s - mean

    # --- initial AR fit (always; final model in pure-AR mode) -----------
    index = _regressor_index(poles, 0, dim)
    h = np.vstack([x.T, np.zeros((dim, length))])
    coeff = _solve_model(h, index, poles - 1, length, dim)
    diff, forecast_errors = _residuals(h, index, coeff, poles, length, dim)

    if refine:
        index = _regressor_index(ar_order, ma_order, dim)
        size = index.shape[0]
        poles_out = max(ar_order, ma_order)
        offset = poles
        h = np.vstack([x.T, diff])  # innovations start from the AR fit
        oldcoeff = np.zeros((dim, size))
        resid_trace = []
        coeff_trace = []
        for n_iterations, _ in enumerate(range(max_iter), start=1):
            offset += poles_out
            coeff = _solve_model(h, index, offset + poles_out - 1, length, dim)
            diff, forecast_errors = _residuals(h, index, coeff, poles_out, length, dim)
            # RMS change of the innovation estimates over the fitted range
            rms = np.sqrt(
                np.sum((h[dim:] - diff)[:, offset:] ** 2, axis=1) / (length - offset)
            )
            h[dim:] = diff
            resid_trace.append(rms)
            coeff_trace.append(float(np.sqrt(np.sum((coeff - oldcoeff) ** 2) / size)))
            oldcoeff = coeff
            if np.max(rms) < convergence:
                break
        aic_k = ar_order + ma_order
        resid_trace = np.array(resid_trace) if resid_trace else np.zeros((0, dim))
        coeff_trace = np.array(coeff_trace) if coeff_trace else np.zeros((0,))
    else:
        poles_out = poles
        aic_k = poles
        n_iterations = 0
        resid_trace = np.zeros((0, dim))
        coeff_trace = np.zeros((0,))

    if np.any(forecast_errors <= 0.0):
        raise ValueError(
            "a forecast error vanished (perfect fit); the log-likelihood is undefined"
        )
    average_forecast_error = float(np.sqrt(np.mean(forecast_errors**2)))
    log_likelihood = float(
        -length * np.sum(np.log(forecast_errors))
        - length * dim * (1.0 + np.log(2.0 * np.pi)) / 2.0
    )
    aic = float(2.0 * aic_k - 2.0 * log_likelihood)

    return {
        "coeffs": coeff,
        "regressors": index,
        "residuals": diff[:, poles_out:].T,
        "forecast_errors": forecast_errors,
        "average_forecast_error": average_forecast_error,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "mean": mean,
        "diff_order": diff_order,
        "diff_anchors": diff_anchors,
        "poles": poles_out,
        "n_iterations": n_iterations,
        "residual_rms_change": resid_trace,
        "coeff_rms_change": coeff_trace,
    }


def iterate_arima_model(fit, n_steps, seed=None, initial=None,
                        restore_level=False):
    """Iterate a fitted ARIMA model with bootstrapped innovations.

    Generates a realisation of the fitted one-step recursion, drawing
    the innovations by resampling the fitted residuals:

        y_d(t) = sum_i coeffs[d, i] * u_i(t - 1) + eps_d(t)

    where ``u_i`` are the lagged data/innovation regressors described
    by ``fit["regressors"]`` and ``eps_d(t)`` is the fresh innovation,
    which also becomes the innovation history used by the MA part.

    Parameters  
    ----------
    fit : dict
        Result of :func:`fit_arima`.
    n_steps : int
        Number of samples to generate (>= 1).
    seed : int or None
        Optional seed for the random generator (reproducibility).
    initial : array_like or None
        Initial data history, shape ``(fit["poles"], dim)`` in
        chronological order (oldest row first); a 1-D array of shape
        ``(fit["poles"],)`` is accepted when ``dim == 1``.  The
        innovation history is always drawn from the bootstrap pool
        (``None`` also draws the data history from it).  The history must be given in
        mean-subtracted differenced units (the model domain),
        regardless of ``restore_level``.
    restore_level : bool, default False
        If True, add back ``fit["mean"]`` and invert the
        ``diff_order``-th differencing (cumulative sums anchored at
        the last observed point of each difference level) so that the
        output is in the original series units and continues from the
        end of the fitted series.  Requires ``fit`` to contain the
        ``"diff_order"``, ``"diff_anchors"`` and ``"mean"`` entries
        produced by :func:`fit_arima`.

    Returns
    -------
    numpy.ndarray
        Generated series of shape ``(n_steps, dim)``.  With
        ``restore_level=False`` (default) it is in mean-subtracted
        units of the ``diff_order``-th difference of the original
        data; with ``restore_level=True`` it is in the original
        series units.

    Raises
    ------
    ValueError
        For an inconsistent ``fit`` or invalid ``n_steps``/``initial``,
        or if ``restore_level`` is True but ``fit`` lacks the
        differencing information.
    KeyError
        If ``fit`` lacks one of the required entries.
    """
    coeffs = np.asarray(fit["coeffs"], dtype=np.float64)
    regressors = np.asarray(fit["regressors"])
    residuals = np.asarray(fit["residuals"], dtype=np.float64)
    if coeffs.ndim != 2 or coeffs.shape[0] < 1 or coeffs.shape[1] < 1:
        raise ValueError(
            f"fit['coeffs'] must have shape (dim, size), got {coeffs.shape}"
        )
    dim, size = coeffs.shape
    if regressors.shape != (size, 3):
        raise ValueError(
            "fit['regressors'] must have shape (size, 3) matching "
            f"coeffs, got {regressors.shape}"
        )
    if residuals.ndim != 2 or residuals.shape[1] != dim or residuals.shape[0] < 1:
        raise ValueError(
            "fit['residuals'] must have shape (n_resid, dim) with "
            f"n_resid >= 1, got {residuals.shape}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    poles = int(regressors[:, 2].max()) + 1

    rng = np.random.default_rng(seed)
    noise = np.empty((poles + n_steps, dim))
    for d in range(dim):
        noise[:, d] = rng.choice(residuals[:, d], size=poles + n_steps, replace=True)

    # Augmented circular state: columns [0, dim) hold the data history,
    # columns [dim, 2*dim) the innovation history (oldest row first).
    buf = np.empty((poles, 2 * dim))
    if initial is None:
        buf[:, :dim] = noise[:poles]
    else:
        init = np.asarray(initial, dtype=np.float64)
        if dim == 1 and init.ndim == 1:
            init = init[:, None]
        if init.shape != (poles, dim):
            raise ValueError(
                f"initial must have shape ({poles}, {dim}), got {init.shape}"
            )
        buf[:, :dim] = init
    buf[:, dim:] = noise[:poles]

    out = np.empty((n_steps, dim))
    pos = 0  # circular-buffer index of the oldest state row
    for t in range(n_steps):
        eps = noise[poles + t]
        val = eps.copy()
        for i in range(size):
            src, lag = regressors[i, 1], regressors[i, 2]
            val += coeffs[:, i] * buf[(pos + poles - 1 - lag) % poles, src]
        out[t] = val
        buf[pos, :dim] = val
        buf[pos, dim:] = eps
        pos = (pos + 1) % poles

    if restore_level:
        missing = [
            k for k in ("diff_order", "diff_anchors", "mean") if k not in fit
        ]
        if missing:
            raise ValueError(
                f"fit is missing {missing}; restore_level=True requires a "
                "fit produced by fit_arima"
            )
        d_order = int(fit["diff_order"])
        anchors = np.asarray(fit["diff_anchors"], dtype=np.float64)
        level_mean = np.asarray(fit["mean"], dtype=np.float64)
        if anchors.shape != (d_order, dim) or level_mean.shape != (dim,):
            raise ValueError(
                "fit['diff_anchors']/fit['mean'] are inconsistent with the "
                f"model dimension ({dim})"
            )
        out = out + level_mean
        for k in range(d_order - 1, -1, -1):
            out = anchors[k] + np.cumsum(out, axis=0)
    return out
