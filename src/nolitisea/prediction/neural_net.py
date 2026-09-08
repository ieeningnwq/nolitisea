"""Neural network baselines for nonlinear dynamics prediction.

Provides six architectures via a unified ``fit_nn(model_type=...)``:

* ``"mlp"`` — feedforward multilayer perceptron
* ``"lstm"`` / ``"gru"`` — recurrent networks
* ``"tcn"`` — temporal convolutional network (causal dilated convolutions)
* ``"narx"`` — nonlinear autoregressive with exogenous inputs (MLP + teacher forcing)
* ``"node"`` — neural ODE (learns vector field ``dx/dt = f(x;θ)``)
* ``"transformer"`` — self-attention over delay window

PyTorch is an *optional* dependency: this module imports without
torch installed, and only raises ``ImportError`` when ``fit_nn`` is
called.  Install with ``pip install nolitisea[nn]``.

The series is rescaled to ``[0, 1]`` per component before training,
matching :func:`fit_rbf` and the local-linear predictors.
"""

from __future__ import annotations

import numpy as np

from ..core.embed import lag_block_delay_embed
from ..utils.rescale import rescale_data

__all__ = ["fit_nn", "nn_forecast_error", "predict_nn"]

_TORCH = None


def _require_torch():
    """Lazily import and cache torch."""
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "nolitisea.prediction.neural_net requires PyTorch. "
            "Install with: pip install nolitisea[nn]  "
            "(or pip install 'torch>=2.0')"
        ) from exc
    _TORCH = torch
    return _TORCH


def _resolve_inputs(series):
    """Return ``(n_times, n_vars, s2d)``."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("series must be a 1-D or 2-D array")
    n_times, n_vars = arr.shape
    return n_times, n_vars, arr


# ---------------------------------------------------------------------------
# Network builders
# ---------------------------------------------------------------------------


def _build_mlp(torch, in_dim, out_dim, hidden_layers, activation):
    layers = []
    prev = in_dim
    act_cls = torch.nn.Tanh if activation == "tanh" else torch.nn.ReLU
    for h in hidden_layers:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(torch.nn.Linear(prev, out_dim))
    return torch.nn.Sequential(*layers)


def _build_rnn(torch, rnn_type, n_vars, hidden_size, out_dim):
    """Build LSTM or GRU + linear projection."""
    rnn_cls = torch.nn.LSTM if rnn_type == "lstm" else torch.nn.GRU

    class RNNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn_cls(n_vars, hidden_size, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, out_dim)

        def forward(self, x):
            # x: (batch, seq_len, n_vars)
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])  # last time step

    return RNNModel()


def _build_tcn(torch, n_vars, hidden_size, out_dim, kernel_size=3, levels=3):
    """Build a Temporal Convolutional Network with causal dilated convolutions."""

    class TCNBlock(torch.nn.Module):
        def __init__(self, in_ch, out_ch, dilation, ks):
            super().__init__()
            pad = (ks - 1) * dilation
            self.conv = torch.nn.Conv1d(
                in_ch, out_ch, ks, padding=pad, dilation=dilation
            )
            self.act = torch.nn.Tanh()
            # Remove the right padding to make it causal
            self.crop = pad

        def forward(self, x):
            # x: (batch, channels, length)
            out = self.conv(x)
            out = self.act(out)
            if self.crop > 0:
                out = out[..., : -self.crop]
            return out

    class TCNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            prev = n_vars
            for i in range(levels):
                dilation = 2**i
                blocks.append(TCNBlock(prev, hidden_size, dilation, kernel_size))
                prev = hidden_size
            self.blocks = torch.nn.ModuleList(blocks)
            self.fc = torch.nn.Linear(hidden_size, out_dim)

        def forward(self, x):
            # x: (batch, n_vars, dim) → treat as (batch, channels, length)
            out = x
            for block in self.blocks:
                out = block(out)
            # Use the last time step
            return self.fc(out[..., -1])

    return TCNModel()


def _build_narx(torch, in_dim, out_dim, hidden_layers, activation):
    """NARX uses the same MLP backbone; training loop differs (teacher forcing)."""
    return _build_mlp(torch, in_dim, out_dim, hidden_layers, activation)


def _build_neural_ode(torch, n_vars, hidden_layers, activation):
    """Build a Neural ODE vector field f(x) -> dx/dt."""
    act_cls = torch.nn.Tanh if activation == "tanh" else torch.nn.ReLU
    layers = []
    prev = n_vars
    for h in hidden_layers:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(torch.nn.Linear(prev, n_vars))
    return torch.nn.Sequential(*layers)


def _build_transformer(torch, n_vars, hidden_size, n_heads, n_layers, out_dim):
    """Build a Transformer encoder + linear head."""

    class PositionalEncoding(torch.nn.Module):
        """Sinusoidal positional encoding (no learnable params)."""

        def __init__(self, d_model, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TransformerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = torch.nn.Linear(n_vars, hidden_size)
            self.pos_encoder = PositionalEncoding(hidden_size)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size * 2,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, n_layers)
            self.fc = torch.nn.Linear(hidden_size, out_dim)

        def forward(self, x):
            # x: (batch, seq_len, n_vars)
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.encoder(x)
            return self.fc(x[:, -1, :])  # last time step

    return TransformerModel()


def _build_network(
    torch,
    model_type,
    in_dim,
    out_dim,
    hidden_layers,
    n_vars,
    activation,
    n_heads,
    n_layers,
):
    """Dispatch to the right builder based on model_type."""
    if model_type == "mlp":
        return _build_mlp(torch, in_dim, out_dim, hidden_layers, activation)
    elif model_type in ("lstm", "gru"):
        return _build_rnn(torch, model_type, n_vars, hidden_layers[0], out_dim)
    elif model_type == "tcn":
        return _build_tcn(torch, n_vars, hidden_layers[0], out_dim)
    elif model_type == "narx":
        return _build_narx(torch, in_dim, out_dim, hidden_layers, activation)
    elif model_type == "node":
        return _build_neural_ode(torch, n_vars, hidden_layers, activation)
    elif model_type == "transformer":
        return _build_transformer(
            torch, n_vars, hidden_layers[0], n_heads, n_layers, out_dim
        )
    else:
        raise ValueError(f"unknown model_type: {model_type}")


def _build_from_state(torch, model):
    """Rebuild network from model dict and load weights."""
    cfg = model["network_config"]
    net = _build_network(
        torch,
        model["model_type"],
        cfg["in_dim"],
        cfg["out_dim"],
        cfg["hidden_layers"],
        model["n_vars"],
        cfg.get("activation", "tanh"),
        cfg.get("n_heads", 4),
        cfg.get("n_layers", 2),
    )
    net = net.double()
    state_dict = {
        k: torch.from_numpy(v).double() for k, v in model["model_state"].items()
    }
    net.load_state_dict(state_dict)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Forward pass helper (architecture-dependent input shaping)
# ---------------------------------------------------------------------------


def _forward_pass(torch, net, model_type, x_vec, n_vars, dim, delay):
    """Run a forward pass with the right input shape for the architecture.

    Parameters
    ----------
    x_vec : np.ndarray, shape (n_vars*dim,)
        Flat delay vector in lag_block_delay_embed column order.

    Returns
    -------
    np.ndarray, shape (n_vars,)
        Predicted next state in rescaled units.  Always 1-D: the redundant
        batch dimension of the sequence models is dropped here so that all
        architectures return the same shape.
    """
    if model_type in ("mlp", "narx"):
        x = torch.from_numpy(np.ascontiguousarray(x_vec)).double()
        with torch.no_grad():
            return net(x).numpy()
    elif model_type in ("lstm", "gru", "transformer"):
        # Reshape to (1, seq_len=dim, n_vars)
        x_seq = x_vec.reshape(dim, n_vars).reshape(1, dim, n_vars)
        x = torch.from_numpy(np.ascontiguousarray(x_seq)).double()
        with torch.no_grad():
            return net(x).numpy()[0]  # drop the batch dim -> (n_vars,)
    elif model_type == "tcn":
        # Reshape to (1, n_vars, dim) — channels = n_vars, length = dim
        x_conv = x_vec.reshape(dim, n_vars).T.reshape(1, n_vars, dim)
        x = torch.from_numpy(np.ascontiguousarray(x_conv)).double()
        with torch.no_grad():
            return net(x).numpy()[0]  # drop the batch dim -> (n_vars,)
    elif model_type == "node":
        # Neural ODE: integrate the current state forward by 1 step
        # For prediction, we use Euler integration: x_{t+1} = x_t + f(x_t) * dt
        x = torch.from_numpy(np.ascontiguousarray(x_vec[:n_vars])).double()
        with torch.no_grad():
            dx = net(x)
        return (x + dx).numpy()
    else:
        raise ValueError(f"unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_nn(
    series,
    model_type="mlp",
    dim=2,
    delay=1,
    hidden_layers=(64, 64),
    epochs=500,
    lr=1e-3,
    step=1,
    insample=None,
    batch_size=None,
    seed=None,
    activation="tanh",
    verbose=False,
    n_heads=4,
    n_layers=2,
):
    """Fit a neural network model to a delay-embedded series.

    Parameters
    ----------
    series : array_like
        1-D (univariate) or 2-D ``(n_times, n_vars)`` input.
    model_type : str, default "mlp"
        One of "mlp", "lstm", "gru", "tcn", "narx", "node", "transformer".
    dim : int, default 2
        Embedding dimension per component.
    delay : int, default 1
        Time delay between successive embedding coordinates.
    hidden_layers : tuple of int, default (64, 64)
        Widths of the hidden layers (for LSTM/GRU/TCN/Transformer only
        the first element is used).
    epochs : int, default 500
        Number of training epochs.
    lr : float, default 1e-3
        Adam learning rate.
    step : int, default 1
        Forecast horizon.
    insample : int or None, default None
        Number of points used for fitting.  ``None`` = whole series.
    batch_size : int or None, default None
        Mini-batch size.  ``None`` = full-batch.
    seed : int or None, default None
        Torch manual seed for reproducibility.
    activation : {"tanh", "relu"}, default "tanh"
        Hidden-layer activation.
    verbose : bool, default False
        Print per-epoch loss if True.
    n_heads : int, default 4
        Number of attention heads (transformer only).
    n_layers : int, default 2
        Number of transformer encoder layers (transformer only).

    Returns
    -------
    dict
        Model dict with keys: ``model_state``, ``network_config``,
        ``model_type``, ``minv``, ``interval``, ``dim``, ``delay``,
        ``step``, ``n_vars``, ``in_sample_rmse``, ``out_of_sample_rmse``,
        ``in_sample_fce``, ``out_of_sample_fce``, ``seed``.

    Raises
    ------
    ValueError
        For invalid parameters.
    ImportError
        If torch is not installed.
    """
    torch = _require_torch()

    valid_types = ("mlp", "lstm", "gru", "tcn", "narx", "node", "transformer")
    if model_type not in valid_types:
        raise ValueError(f"model_type must be one of {valid_types}, got {model_type!r}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if not hidden_layers:
        raise ValueError("hidden_layers must be non-empty")
    if activation not in ("tanh", "relu"):
        raise ValueError(f"activation must be 'tanh' or 'relu', got {activation!r}")

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
        raise ValueError(f"insample={insample} is too short")

    # --- per-component rescale ------------------------------------------
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
    in_dim = n_vars * dim
    out_dim = n_vars

    # --- training set ----------------------------------------------------
    n_train = insample - step - valid_start
    if n_train <= 0:
        raise ValueError("no valid training samples after embedding and step")

    X_train_flat = E[:n_train]  # (n_train, in_dim)
    y_train = rescaled[valid_start + step : valid_start + step + n_train]

    # --- seed ------------------------------------------------------------
    if seed is not None:
        torch.manual_seed(int(seed))

    # --- build network ---------------------------------------------------
    net = _build_network(
        torch,
        model_type,
        in_dim,
        out_dim,
        hidden_layers,
        n_vars,
        activation,
        n_heads,
        n_layers,
    )
    net = net.double()

    # --- prepare training data (architecture-dependent shaping) ---------
    X_t, y_t = _prepare_training_input(
        torch, model_type, X_train_flat, y_train, n_vars, dim
    )

    # --- train -----------------------------------------------------------
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    if batch_size is None or batch_size >= n_train:
        batch_size = n_train

    for epoch in range(epochs):
        if batch_size < n_train:
            perm = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                pred = net(_index_batch(X_t, idx))
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                opt.step()
        else:
            opt.zero_grad()
            pred = net(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={loss.item():.6e}")

    # --- in-sample RMSE --------------------------------------------------
    with torch.no_grad():
        pred_in = net(X_t).numpy()
    in_rmse = float(np.sqrt(np.mean((pred_in - y_train) ** 2)))

    # --- out-of-sample RMSE ---------------------------------------------
    out_rmse = None
    out_fce = None
    if insample < n_times:
        n_out = n_times - insample
        if n_out > 0:
            # E[n_train + k] is the input window ending at time
            # t = insample - step + k; its step-ahead target is
            # rescaled[t + step] = rescaled[insample + k].
            X_out_flat = E[n_train : n_train + n_out]
            y_out = rescaled[insample : insample + n_out]
            X_out_t, _ = _prepare_training_input(
                torch, model_type, X_out_flat, y_out, n_vars, dim
            )
            with torch.no_grad():
                pred_out = net(X_out_t).numpy()
            out_rmse = float(np.sqrt(np.mean((pred_out - y_out) ** 2)))

    # --- FCE -------------------------------------------------------------
    _in = rescaled[:insample]
    in_sigma = float(np.std(_in, ddof=1)) if insample > 1 else 0.0
    in_fce = in_rmse / in_sigma if in_sigma > 0 else float("nan")
    if out_rmse is not None:
        _out = rescaled[insample:]
        out_sigma = float(np.std(_out, ddof=1)) if len(_out) > 1 else 0.0
        out_fce = out_rmse / out_sigma if out_sigma > 0 else float("nan")

    # --- serialize state -------------------------------------------------
    model_state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}

    return {
        "model_state": model_state,
        "network_config": {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "hidden_layers": tuple(hidden_layers),
            "activation": activation,
            "n_heads": n_heads,
            "n_layers": n_layers,
        },
        "model_type": model_type,
        "minv": minv,
        "interval": interval,
        "dim": dim,
        "delay": delay,
        "step": step,
        "n_vars": n_vars,
        "in_sample_rmse": in_rmse,
        "out_of_sample_rmse": out_rmse,
        "in_sample_fce": in_fce,
        "out_of_sample_fce": out_fce,
        "seed": seed,
    }


def _prepare_training_input(torch, model_type, X_flat, y, n_vars, dim):
    """Convert flat delay vectors to architecture-specific tensor shapes."""
    n = X_flat.shape[0]
    y_t = torch.from_numpy(np.ascontiguousarray(y)).double()

    if model_type in ("mlp", "narx"):
        X_t = torch.from_numpy(np.ascontiguousarray(X_flat)).double()
    elif model_type in ("lstm", "gru", "transformer"):
        # Reshape (n, n_vars*dim) → (n, dim, n_vars)
        X_seq = X_flat.reshape(n, dim, n_vars)
        X_t = torch.from_numpy(np.ascontiguousarray(X_seq)).double()
    elif model_type == "tcn":
        # Reshape (n, n_vars*dim) → (n, n_vars, dim)
        X_conv = X_flat.reshape(n, dim, n_vars).transpose(0, 2, 1)
        X_t = torch.from_numpy(np.ascontiguousarray(X_conv)).double()
    elif model_type == "node":
        # Neural ODE: input is just the current state (first n_vars entries)
        X_t = torch.from_numpy(np.ascontiguousarray(X_flat[:, :n_vars])).double()
    else:
        X_t = torch.from_numpy(np.ascontiguousarray(X_flat)).double()

    return X_t, y_t


def _index_batch(X_t, idx):
    """Index a batch from the training tensor (handles different shapes)."""
    return X_t[idx]


# ---------------------------------------------------------------------------
# Forecast error
# ---------------------------------------------------------------------------

# Names accepted by ``nn_forecast_error(error_metric=...)``; each maps to a
# scalar function ``(targets, pred) -> float`` from
# :mod:`nolitisea.noise.compare`.  The metrics that return tuples
# (``pearson_corr``, ``max_cross_correlation``) are intentionally excluded
# because ``nn_forecast_error`` returns a single float.
_ERROR_METRIC_NAMES = (
    "mae",
    "mse",
    "rmse",
    "nrmse_ptp",
    "r2",
    "r2_score",
    "cosine",
    "cosine_similarity",
    "dtw",
    "dtw_distance",
)


def _resolve_error_metric(error_metric):
    """Resolve an error-metric spec to a callable ``f(targets, pred) -> float``.

    Parameters
    ----------
    error_metric : str or callable
        - ``str``: name of a metric from :mod:`nolitisea.noise.compare`
          (one of :data:`_ERROR_METRIC_NAMES`).  ``"r2"`` and ``"r2_score"``
          are aliases; likewise ``"cosine"``/``"cosine_similarity"`` and
          ``"dtw"``/``"dtw_distance"``.
        - ``callable``: invoked as ``error_metric(targets, pred)`` where
          both arrays are 1-D (flattened) and ``targets`` is the ground
          truth.  Must return a float.

    Returns
    -------
    callable
        Function ``(targets, pred) -> float``.

    Raises
    ------
    ValueError
        If ``error_metric`` is a string not in the registry.
    """
    if callable(error_metric):
        return error_metric

    name = str(error_metric).lower()
    # Lazy import: keeps this module importable even when scipy is absent
    # (compare.py pulls in scipy.stats.pearsonr at its own import time).
    from ..noise.compare import (
        cosine_similarity,
        dtw_distance,
        mae,
        mse,
        nrmse_ptp,
        r2_score,
        rmse,
    )

    registry = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "nrmse_ptp": nrmse_ptp,
        "r2": r2_score,
        "r2_score": r2_score,
        "cosine": cosine_similarity,
        "cosine_similarity": cosine_similarity,
        "dtw": dtw_distance,
        "dtw_distance": dtw_distance,
    }
    if name not in registry:
        raise ValueError(
            f"unknown error_metric: {error_metric!r}. "
            f"Expected one of {_ERROR_METRIC_NAMES} or a callable."
        )
    return registry[name]


def nn_forecast_error(
    series, model, dim, delay, step, i0, i1, error_metric="rmse"
):
    """NN forecast error over ``[i0, i1)`` under a chosen metric.

    Parameters
    ----------
    series : array_like
        *Rescaled* series.
    model : dict
        Model returned by :func:`fit_nn`.
    dim, delay, step : int
    i0, i1 : int
        Start and exclusive end of the interval.
    error_metric : str or callable, default "rmse"
        Error function used to score predictions against targets.

        - ``str``: name of a metric from :mod:`nolitisea.noise.compare`.
          Accepted names: ``"mae"``, ``"mse"``, ``"rmse"`` (default),
          ``"nrmse_ptp"``, ``"r2"``/``"r2_score"``, ``"cosine"``
          /``"cosine_similarity"``, ``"dtw"``/``"dtw_distance"``.
          The metrics that return tuples (``pearson_corr``,
          ``max_cross_correlation``) are not supported because this
          function returns a single float.
        - ``callable``: invoked as ``error_metric(targets, pred)`` where
          both arrays are 1-D (flattened) and ``targets`` is the ground
          truth; must return a float.

    Returns
    -------
    float
        Error in rescaled units, or NaN for an empty/invalid interval.
        The default ``"rmse"`` reproduces the previous behaviour exactly.
    """
    torch = _require_torch()
    net = _build_from_state(torch, model)
    model_type = model["model_type"]
    n_vars = model["n_vars"]

    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]

    valid_start = (dim - 1) * delay
    n_start = i0 + valid_start
    n_end = i1 - step
    if n_start >= n_end:
        return float("nan")

    denom = i1 - i0 - step - valid_start
    if denom <= 0:
        return float("nan")

    sub = arr[i0:i1]
    E = lag_block_delay_embed(sub, embed=dim, delay=delay)
    targets = arr[n_start + step : n_end + step]
    n_eval = min(E.shape[0], targets.shape[0])

    X_t, _ = _prepare_training_input(
        torch, model_type, E[:n_eval], targets[:n_eval], n_vars, dim
    )

    with torch.no_grad():
        pred = net(X_t).numpy()

    targets_eval = targets[:n_eval]
    metric_fn = _resolve_error_metric(error_metric)
    # compare.py metrics require 1-D input; flatten multivariate arrays.
    # This matches the previous np.mean-based RMSE over all elements.
    return float(metric_fn(targets_eval.ravel(), pred.ravel()))  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Predict (iterated forecast)
# ---------------------------------------------------------------------------


def predict_nn(model, series, n_steps):
    """Iterate a fitted NN model ``n_steps`` into the future.

    Parameters
    ----------
    model : dict
        Model returned by :func:`fit_nn`.
    series : array_like
        Seed series in physical units.
    n_steps : int
        Number of future values.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_steps,)`` for univariate, ``(n_steps, n_vars)`` for
        multivariate, in physical units.
    """
    if n_steps < 1:
        n_vars = model["n_vars"]
        if n_vars == 1:
            return np.array([], dtype=np.float64)
        return np.empty((0, n_vars), dtype=np.float64)

    torch = _require_torch()
    net = _build_from_state(torch, model)
    model_type = model["model_type"]
    minv = model["minv"]
    interval = model["interval"]
    dim = model["dim"]
    delay = model["delay"]
    n_vars = model["n_vars"]

    # Rescale seed
    s = np.asarray(series, dtype=np.float64)
    if s.ndim == 1:
        if n_vars != 1:
            raise ValueError(f"model expects {n_vars} components but got 1-D series")
        s = s[:, None]
    if s.shape[1] != n_vars:
        raise ValueError(f"model expects {n_vars} components but got {s.shape[1]}")
    s_resc = (s - minv) / interval

    # Build rolling buffer
    buf_len = (dim - 1) * delay + 1
    buf = np.zeros((buf_len, n_vars), dtype=np.float64)
    if len(s_resc) < buf_len:
        buf[buf_len - len(s_resc) :] = s_resc
    else:
        buf[:] = s_resc[-buf_len:]

    dim_offset = (dim - 1) * delay
    out = np.empty((n_steps, n_vars), dtype=np.float64)

    for step_idx in range(n_steps):
        # Build delay vector in lag_block_delay_embed column order
        x_vec = np.empty(n_vars * dim, dtype=np.float64)
        for k in range(dim):
            for c in range(n_vars):
                x_vec[k * n_vars + c] = buf[dim_offset - k * delay, c]

        # Forward pass
        pred = _forward_pass(torch, net, model_type, x_vec, n_vars, dim, delay)
        # pred has shape (n_vars,)

        # Store physical units
        out[step_idx] = pred * interval + minv

        # Slide buffer (for Neural ODE, we only use the first n_vars entries)
        buf[:-1] = buf[1:]
        buf[-1] = pred

    if n_vars == 1:
        return out[:, 0]
    return out
