"""Multivariate autoregressive model fitting and iteration."""

import numpy as np

__all__ = ["fit_ar_model", "iterate_ar_model"]


def fit_ar_model(series, order=1):
    """Fit a multivariate autoregressive (AR) model.

    The model predicts every component from all components at the
    previous ``order`` time steps::

        y_d(t) = sum_{i=0}^{dim-1} sum_{j=0}^{order-1}
                 coeffs[d, i, j] * y_i(t - 1 - j) + residual_d(t)

    The per-component mean is subtracted internally and the
    coefficients are obtained from the ordinary least-squares normal
    equations with the TISEAN normalisation ``1/(N - order)``.  All
    outputs are in mean-subtracted units; add ``mean`` to restore the
    original level.

    Parameters
    ----------
    series : array_like
        Input data of shape ``(N,)`` (univariate) or ``(N, dim)``
        (multivariate; components are columns).
    order : int, default 1
        AR model order.  Must satisfy ``1 <= order < N``.

    Returns
    -------
    dict
        ``"coeffs"`` : coefficient tensor, shape ``(dim, dim, order)``;
        ``coeffs[d, i, j]`` multiplies ``y_i(t-1-j)`` in the equation
        for ``y_d(t)``.
        ``"residuals"`` : one-step prediction errors, shape
        ``(N - order, dim)``; row ``k`` corresponds to time ``t =
        order + k`` (the first ``order`` samples have no prediction).
        ``"forecast_errors"`` : residual RMS per component, shape
        ``(dim,)``.
        ``"average_forecast_error"`` : RMS of the forecast errors
        across components (float).
        ``"mean"`` : per-component mean that was subtracted, shape
        ``(dim,)``.

    Raises
    ------
    ValueError
        For invalid parameters or an empty/too-short series.
    numpy.linalg.LinAlgError
        If the normal-equation matrix is singular (e.g. a constant
        series).

    Notes
    -----
    With ``N`` samples and model order ``p`` the residuals use the
    population normalisation ``sigma_d^2 = sum_t residual_d(t)^2 / (N - p)``
    (no degrees-of-freedom correction), matching TISEAN exactly.

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
    length, dim = s.shape
    if length < 1 or dim < 1:
        raise ValueError("series must not be empty")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if order >= length:
        raise ValueError(
            f"order ({order}) must be smaller than the series length ({length})"
        )

    n_valid = length - order
    mean = s.mean(axis=0)
    x = s - mean

    # Regressor matrix: row k corresponds to time t = order - 1 + k and
    # holds x_i(t - j) for i = 0..dim-1, j = 0..order-1, with column
    # index i * order + j.
    lags = np.stack([x[order - 1 - j : length - 1 - j, :] for j in range(order)])
    design = np.transpose(lags, (2, 0, 1)).reshape(dim * order, n_valid).T
    target = x[order:, :]

    norm = 1.0 / n_valid
    gram = (design.T @ design) * norm  # build_matrix
    cross = (design.T @ target) * norm  # build_vector

    solved = np.linalg.solve(gram, cross)  # (dim * order, dim), row i*order+j
    coeffs = solved.reshape(dim, order, dim).transpose(2, 0, 1)  # (d, i, j)

    residuals = target - design @ solved
    forecast_errors = np.sqrt(np.sum(residuals**2, axis=0) / n_valid)
    average_forecast_error = float(np.sqrt(np.mean(forecast_errors**2)))

    return {
        "coeffs": coeffs,
        "residuals": residuals,
        "forecast_errors": forecast_errors,
        "average_forecast_error": average_forecast_error,
        "mean": mean,
    }


def iterate_ar_model(coeffs, sigma, n_steps, seed=None, initial=None):
    """Iterate a fitted multivariate AR model driven by Gaussian noise.

    Generates a realisation of the recursion:

        y_d(t) = sum_{i, j} coeffs[d, i, j] * y_i(t - 1 - j) + e_d(t),
        e_d ~ N(0, sigma_d^2)

    Parameters
    ----------
    coeffs : array_like
        Coefficient tensor of shape ``(dim, dim, order)`` as returned
        by :func:`fit_ar_model`.
    sigma : float or array_like
        Standard deviation of the driving noise: a scalar (same for
        every component) or a vector of shape ``(dim,)``.  Typically
        the fitted ``"forecast_errors"``.  ``0`` yields a deterministic
        iteration.
    n_steps : int
        Number of samples to generate (>= 1).
    seed : int or None
        Optional seed for the random generator (reproducibility).
    initial : array_like or None
        Initial history of shape ``(order, dim)`` in chronological
        order (oldest row first); a 1-D array of shape ``(order,)``
        is accepted when ``dim == 1``.  ``None`` (default) draws the
        history from the same Gaussian noise, mirroring the TISEAN
        warm start.

    Returns
    -------
    numpy.ndarray
        Generated series of shape ``(n_steps, dim)``, in mean-
        subtracted units (add the ``"mean"`` of the fit to restore the
        original level).

    Raises
    ------
    ValueError
        For invalid ``coeffs``, ``sigma``, ``n_steps`` or ``initial``.
    """
    a = np.asarray(coeffs, dtype=np.float64)
    if a.ndim != 3 or a.shape[0] != a.shape[1] or a.shape[2] < 1:
        raise ValueError(
            f"coeffs must have shape (dim, dim, order) with order >= 1, got {a.shape}"
        )
    dim, order = a.shape[0], a.shape[2]

    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if sigma_arr.ndim == 0:
        sigma_arr = np.full(dim, float(sigma_arr))
    if sigma_arr.shape != (dim,):
        raise ValueError(
            f"sigma must be a scalar or have shape ({dim},), got {sigma_arr.shape}"
        )
    if not np.all(np.isfinite(sigma_arr)) or np.any(sigma_arr < 0):
        raise ValueError("sigma must be finite and >= 0")

    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    rng = np.random.default_rng(seed)

    buf = np.empty((order, dim))
    if initial is None:
        buf[:] = rng.normal(0.0, sigma_arr, size=(order, dim))
    else:
        init = np.asarray(initial, dtype=np.float64)
        if dim == 1 and init.ndim == 1:
            init = init[:, None]
        if init.shape != (order, dim):
            raise ValueError(
                f"initial must have shape ({order}, {dim}), got {init.shape}"
            )
        buf[:] = init

    # Pre-extract per-lag coupling matrices for the recursion loop.
    mats = [np.ascontiguousarray(a[:, :, j]) for j in range(order)]

    out = np.empty((n_steps, dim))
    pos = 0  # circular-buffer index of the oldest history row
    for t in range(n_steps):
        val = rng.normal(0.0, sigma_arr)
        for j, mat in enumerate(mats):
            val += mat @ buf[(pos + order - 1 - j) % order]
        out[t] = val
        buf[pos] = val
        pos = (pos + 1) % order
    return out
