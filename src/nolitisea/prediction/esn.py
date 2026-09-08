"""Echo State Network (ESN) prediction.

A reservoir-computing baseline that uses a fixed random reservoir
with a trained linear readout (ridge regression).  The reservoir
state evolves as::

    h(t) = tanh(W_in @ x(t) + W_res @ h(t-1))

and the readout maps states to future values::

    y(t+step) = W_out @ [1, h(t)]

Training is a single matrix solve (no backprop), making it extremely
fast and well-suited for chaotic dynamics prediction.

The series is rescaled to ``[0, 1]`` per component before fitting,
matching the convention used by :func:`fit_rbf` and the local-linear
predictors.
"""

from __future__ import annotations

import numpy as np

from ..core.embed import lag_block_delay_embed
from ..utils.rescale import rescale_data

__all__ = ["esn_forecast_error", "fit_esn", "predict_esn"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_inputs(series):
    """Return ``(n_times, n_vars, s2d)`` where ``s2d`` is ``(n_times, n_vars)``."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("series must be a 1-D or 2-D array")
    n_times, n_vars = arr.shape
    return n_times, n_vars, arr


def _init_reservoir(n_reservoir, in_dim, n_vars, spectral_radius,
                    sparsity, rng):
    """Initialize ESN weight matrices.

    Parameters
    ----------
    n_reservoir : int
        Number of reservoir nodes.
    in_dim : int
        Input dimension (n_vars * embed).
    n_vars : int
        Number of output variables.
    spectral_radius : float
        Desired spectral radius of the reservoir matrix.
    sparsity : float
        Fraction of zero entries in the reservoir matrix.
    rng : np.random.Generator

    Returns
    -------
    W_in : np.ndarray, shape (n_reservoir, in_dim)
    W_res : np.ndarray, shape (n_reservoir, n_reservoir)
    W_fb : np.ndarray, shape (n_reservoir, n_vars) or None
        Feedback weights (not used in this implementation).
    """
    # Input weights: uniform [-1, 1]
    W_in = rng.uniform(-1.0, 1.0, size=(n_reservoir, in_dim))

    # Reservoir matrix: sparse random, then scaled to spectral radius
    W_res = rng.uniform(-1.0, 1.0, size=(n_reservoir, n_reservoir))
    # Apply sparsity mask
    mask = rng.random(size=W_res.shape) > sparsity
    W_res[mask] = 0.0

    # Scale to desired spectral radius
    eigenvalues = np.linalg.eigvals(W_res)
    current_radius = np.max(np.abs(eigenvalues))
    if current_radius > 1e-30:
        W_res *= spectral_radius / current_radius

    return W_in, W_res


def _harvest_states(W_in, W_res, E, n_train, n_reservoir):
    """Run the reservoir over the training data and collect states.

    Parameters
    ----------
    W_in : np.ndarray, shape (n_reservoir, in_dim)
    W_res : np.ndarray, shape (n_reservoir, n_reservoir)
    E : np.ndarray, shape (n_points, in_dim)
        Delay-embedding matrix.
    n_train : int
        Number of points to harvest.
    n_reservoir : int

    Returns
    -------
    H : np.ndarray, shape (n_train, n_reservoir + 1)
        State matrix with a bias column prepended.
    """
    h = np.zeros(n_reservoir, dtype=np.float64)
    H = np.empty((n_train, n_reservoir + 1), dtype=np.float64)
    H[:, 0] = 1.0  # bias term

    for t in range(n_train):
        h = np.tanh(W_in @ E[t] + W_res @ h)
        H[t, 1:] = h

    return H


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_esn(series, dim=2, delay=1, n_reservoir=500, spectral_radius=0.9,
            sparsity=0.2, ridge_alpha=1e-6, step=1, insample=None,
            seed=None):
    """Fit an Echo State Network to a delay-embedded series.

    The model uses a fixed random reservoir with a trained linear
    readout (ridge regression).  The series is rescaled to ``[0, 1]``
    per component before fitting, and the returned model carries
    ``(minv, interval)`` per component so :func:`predict_esn` can map
    forecasts back to physical units.

    Parameters
    ----------
    series : array_like
        1-D (univariate) or 2-D ``(n_times, n_vars)`` input.
    dim : int, default 2
        Embedding dimension per component.
    delay : int, default 1
        Time delay between successive embedding coordinates.
    n_reservoir : int, default 500
        Number of reservoir nodes.
    spectral_radius : float, default 0.9
        Spectral radius of the reservoir matrix (controls memory).
    sparsity : float, default 0.2
        Fraction of zero entries in the reservoir matrix.
    ridge_alpha : float, default 1e-6
        Ridge regression regularization parameter.
    step : int, default 1
        Forecast horizon.
    insample : int or None, default None
        Number of points used for fitting (rest held out for
        out-of-sample error).  ``None`` means the whole series.
    seed : int or None, default None
        Random seed for reproducible reservoir initialization.

    Returns
    -------
    dict
        Keys:

        - ``"W_in"`` : np.ndarray, shape (n_reservoir, n_vars*dim)
        - ``"W_res"`` : np.ndarray, shape (n_reservoir, n_reservoir)
        - ``"W_out"`` : np.ndarray, shape (n_vars, n_reservoir+1)
        - ``"minv"`` : np.ndarray, shape (n_vars,)
        - ``"interval"`` : np.ndarray, shape (n_vars,)
        - ``"dim"``, ``"delay"``, ``"step"`` : int
        - ``"n_vars"`` : int
        - ``"n_reservoir"`` : int
        - ``"in_sample_rmse"`` : float
        - ``"out_of_sample_rmse"`` : float or None
        - ``"in_sample_fce"`` : float
        - ``"out_of_sample_fce"`` : float or None
        - ``"seed"`` : int or None
        - ``"last_state"`` : np.ndarray, shape (n_reservoir,)
          Last reservoir state (for warm-starting prediction).

    Raises
    ------
    ValueError
        For invalid parameters or a series too short.
    RuntimeError
        If a component is constant (zero range).
    """
    # --- validation -------------------------------------------------------
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if n_reservoir < 1:
        raise ValueError(f"n_reservoir must be >= 1, got {n_reservoir}")
    if spectral_radius <= 0:
        raise ValueError(f"spectral_radius must be > 0, got {spectral_radius}")
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if ridge_alpha < 0:
        raise ValueError(f"ridge_alpha must be >= 0, got {ridge_alpha}")

    n_times, n_vars, s2d = _resolve_inputs(series)

    valid_start = (dim - 1) * delay
    if n_times <= step + valid_start:
        raise ValueError(
            f"series of length {n_times} is too short for dim={dim}, "
            f"delay={delay}, step={step}"
        )

    if insample is None or insample > n_times:
        insample = n_times
    if insample <= step + valid_start:
        raise ValueError(
            f"insample={insample} is too short for dim={dim}, "
            f"delay={delay}, step={step}"
        )

    # --- per-component rescale -------------------------------------------
    minv = np.empty(n_vars, dtype=np.float64)
    interval = np.empty(n_vars, dtype=np.float64)
    rescaled = np.empty_like(s2d)
    for c in range(n_vars):
        sr, mn, iv = rescale_data(s2d[:, c])
        minv[c] = mn
        interval[c] = iv
        rescaled[:, c] = sr

    # --- delay embedding -------------------------------------------------
    E = lag_block_delay_embed(rescaled, embed=dim, delay=delay)
    # E[r] corresponds to original time t = r + valid_start
    in_dim = n_vars * dim

    # --- training set construction ---------------------------------------
    # E rows 0..n_train-1 correspond to times valid_start..insample-step-1
    # Targets: rescaled[valid_start+step .. insample-1+step] → but we need
    # targets at t+step where t = r + valid_start, so:
    #   r ranges from 0 to (insample - step - valid_start - 1)
    n_train = insample - step - valid_start
    if n_train <= 0:
        raise ValueError("no valid training samples after embedding and step")

    X_train = E[:n_train]                              # (n_train, in_dim)
    y_train = rescaled[valid_start + step: valid_start + step + n_train]

    # --- reservoir init --------------------------------------------------
    rng = np.random.default_rng(seed)
    W_in, W_res = _init_reservoir(
        n_reservoir, in_dim, n_vars, spectral_radius, sparsity, rng
    )

    # --- harvest states --------------------------------------------------
    H = _harvest_states(W_in, W_res, X_train, n_train, n_reservoir)

    # --- ridge regression readout ----------------------------------------
    # W_out = (H^T H + αI)^{-1} H^T y_train
    # Solve: (H^T H + αI) W_out^T = H^T y_train
    HtH = H.T @ H
    HtY = H.T @ y_train
    reg = ridge_alpha * np.eye(n_reservoir + 1, dtype=np.float64)
    reg[0, 0] = 0.0  # don't regularize the bias
    W_out = np.linalg.solve(HtH + reg, HtY).T  # shape (n_vars, n_reservoir+1)

    # --- in-sample RMSE --------------------------------------------------
    pred_in = H @ W_out.T  # (n_train, n_vars)
    in_rmse = float(np.sqrt(np.mean((pred_in - y_train) ** 2)))

    # --- out-of-sample RMSE ---------------------------------------------
    out_rmse = None
    out_fce = None
    if insample < n_times:
        n_out = n_times - insample
        if n_out > 0:
            # E[n_train + k] is the state at time t = insample - step + k;
            # its step-ahead target is rescaled[t + step] = rescaled[insample + k].
            X_out = E[n_train: n_train + n_out]
            y_out = rescaled[insample: insample + n_out]
            H_out = _harvest_states(W_in, W_res, X_out, n_out, n_reservoir)
            pred_out = H_out @ W_out.T
            out_rmse = float(np.sqrt(np.mean((pred_out - y_out) ** 2)))

    # --- FCE (normalized by sample std) --------------------------------
    _in = rescaled[:insample]
    in_sigma = float(np.std(_in, ddof=1)) if insample > 1 else 0.0
    in_fce = in_rmse / in_sigma if in_sigma > 0 else float("nan")
    if out_rmse is not None:
        _out = rescaled[insample:]
        out_sigma = float(np.std(_out, ddof=1)) if len(_out) > 1 else 0.0
        out_fce = out_rmse / out_sigma if out_sigma > 0 else float("nan")

    # --- last reservoir state (for prediction warm-start) ---------------
    # Run reservoir through all training data to get final state
    h_final = np.zeros(n_reservoir, dtype=np.float64)
    for t in range(n_train):
        h_final = np.tanh(W_in @ X_train[t] + W_res @ h_final)

    return {
        "W_in": W_in,
        "W_res": W_res,
        "W_out": W_out,
        "minv": minv,
        "interval": interval,
        "dim": dim,
        "delay": delay,
        "step": step,
        "n_vars": n_vars,
        "n_reservoir": n_reservoir,
        "in_sample_rmse": in_rmse,
        "out_of_sample_rmse": out_rmse,
        "in_sample_fce": in_fce,
        "out_of_sample_fce": out_fce,
        "seed": seed,
        "last_state": h_final,
    }


# ---------------------------------------------------------------------------
# Forecast error
# ---------------------------------------------------------------------------

def esn_forecast_error(series, model, dim, delay, step, i0, i1):
    """Root-mean-square ESN forecast error over ``[i0, i1)``.

    Parameters
    ----------
    series : array_like
        *Rescaled* series (per component, in ``[0, 1]``).
    model : dict
        Model returned by :func:`fit_esn`.
    dim, delay, step : int
        Embedding parameters (must match the model's values).
    i0, i1 : int
        Start and exclusive end of the interval to evaluate.

    Returns
    -------
    float
        RMSE in rescaled units, or NaN if no valid samples.
    """
    W_in = model["W_in"]
    W_res = model["W_res"]
    W_out = model["W_out"]
    n_reservoir = model["n_reservoir"]
    n_vars = model["n_vars"]

    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("series must be 1-D or 2-D")

    valid_start = (dim - 1) * delay
    n_start = i0 + valid_start
    n_end = i1 - step
    if n_start >= n_end:
        return float("nan")

    denom = i1 - i0 - step - valid_start
    if denom <= 0:
        return float("nan")

    # Build embedding for the evaluation range
    sub = arr[i0: i1]
    E = lag_block_delay_embed(sub, embed=dim, delay=delay)
    n_eval = E.shape[0]

    # Targets
    targets = arr[n_start + step: n_end + step]

    # Harvest states
    h = np.zeros(n_reservoir, dtype=np.float64)
    error = 0.0
    count = 0
    for t in range(n_eval):
        h = np.tanh(W_in @ E[t] + W_res @ h)
        pred = W_out @ np.concatenate([[1.0], h])
        if t < len(targets):
            error += np.sum((pred - targets[t]) ** 2)
            count += n_vars

    if count == 0:
        return float("nan")
    return float(np.sqrt(error / count))


# ---------------------------------------------------------------------------
# Predict (iterated forecast)
# ---------------------------------------------------------------------------

def predict_esn(model, series, n_steps):
    """Iterate a fitted ESN model ``n_steps`` into the future.

    The forecast is seeded with the last ``(dim-1)*delay + 1`` values
    of the input series (per component).  The reservoir state is
    warm-started from the model's ``last_state``.  Each predicted
    value is fed back as the newest entry of the delay window.

    Parameters
    ----------
    model : dict
        Model returned by :func:`fit_esn`.
    series : array_like
        Seed series in physical units.
    n_steps : int
        Number of future values to produce.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_steps,)`` for univariate, ``(n_steps, n_vars)``
        for multivariate, in physical units.
    """
    if n_steps < 1:
        n_vars = model["n_vars"]
        return np.empty((0, n_vars), dtype=np.float64) if n_vars > 1 \
            else np.array([], dtype=np.float64)

    W_in = model["W_in"]
    W_res = model["W_res"]
    W_out = model["W_out"]
    minv = model["minv"]
    interval = model["interval"]
    dim = model["dim"]
    delay = model["delay"]
    step = model["step"]
    n_vars = model["n_vars"]
    n_reservoir = model["n_reservoir"]
    h = model["last_state"].copy()

    # Rescale seed
    s = np.asarray(series, dtype=np.float64)
    if s.ndim == 1:
        if n_vars != 1:
            raise ValueError(
                f"model expects {n_vars} components but got 1-D series"
            )
        s = s[:, None]
    if s.shape[1] != n_vars:
        raise ValueError(
            f"model expects {n_vars} components but got {s.shape[1]}"
        )
    s_resc = (s - minv) / interval

    # Build rolling buffer
    buf_len = (dim - 1) * delay + 1
    buf = np.zeros((buf_len, n_vars), dtype=np.float64)
    if len(s_resc) < buf_len:
        buf[buf_len - len(s_resc):] = s_resc
    else:
        buf[:] = s_resc[-buf_len:]

    dim_offset = (dim - 1) * delay
    out = np.empty((n_steps, n_vars), dtype=np.float64)

    for step_idx in range(n_steps):
        # Build delay vector in lag_block_delay_embed column order:
        # x_vec[k*n_vars + c] = buf[dim_offset - k*delay, c]
        x_vec = np.empty(n_vars * dim, dtype=np.float64)
        for k in range(dim):
            for c in range(n_vars):
                x_vec[k * n_vars + c] = buf[dim_offset - k * delay, c]

        # Reservoir update + readout
        h = np.tanh(W_in @ x_vec + W_res @ h)
        pred = W_out @ np.concatenate([[1.0], h])

        # Store physical units
        out[step_idx] = pred * interval + minv

        # Slide buffer
        buf[:-1] = buf[1:]
        buf[-1] = pred

    if n_vars == 1:
        return out[:, 0]
    return out
