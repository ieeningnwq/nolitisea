"""Add measurement noise to a multivariate series (TISEAN ``addnoise`` style)."""

import numpy as np

__all__ = ["make_noise"]


def make_noise(series, level=0.05, noise_type="gaussian", absolute=False, seed=None):
    """Add noise to every column of a multivariate series.

    Each column of ``series`` is treated as an independent 1-D time
    series; the noise of one column neither depends on nor leaks into
    the others.  In every case ``level`` denotes the noise *standard
    deviation* of each column: ``level * std(series[:, j])`` by default
    (the convention of the TISEAN ``addnoise -%`` option), or the
    absolute standard deviation when ``absolute=True``.

    Parameters
    ----------
    series : array_like
        Input data.  A 1-D array is a single time series; a 2-D array
        must have shape ``(n_times, n_vars)`` with one column per
        variable.
    level : float
        Noise level (standard deviation), relative to the per-column
        standard deviation or absolute when ``absolute=True``.
    noise_type : {"gaussian", "uniform"}, default "gaussian"
        Distribution of the added noise:

        - ``"gaussian"``: white noise drawn from ``N(0, sigma^2)``.
        - ``"uniform"``: white noise drawn uniformly from
          ``[-sqrt(3) * sigma, sqrt(3) * sigma]``, so that its standard
          deviation also equals ``sigma``.

        Here ``sigma`` is the per-column noise standard deviation
        implied by ``level``.
    absolute : bool, default False
        If ``True``, ``level`` is the absolute noise standard
        deviation shared by every column; otherwise ``level`` is
        relative to the standard deviation of each column.
    seed : int or None, optional
        Seed of the random generator for reproducibility.  ``None``
        (default) draws fresh entropy on every call.

    Returns
    -------
    numpy.ndarray
        Noisy copy of the input with the same shape, dtype float64.

    Raises
    ------
    ValueError
        If the input is empty, has more than two dimensions, ``level``
        is negative, or ``noise_type`` is unknown.
    """
    original_ndim = np.asarray(series).ndim
    data = np.asarray(series, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(
            "series must be a 1-D array or a 2-D array of shape "
            "(n_times, n_vars)"
        )
    if data.size == 0:
        raise ValueError("series must contain at least one sample")
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    noise_type = str(noise_type).lower()
    if noise_type not in ("gaussian", "uniform"):
        raise ValueError(
            "noise_type must be 'gaussian' or 'uniform', got "
            f"'{noise_type}'"
        )

    if absolute:
        sigmas = np.full(data.shape[1], float(level))
    else:
        sigmas = level * data.std(axis=0)

    rng = np.random.default_rng(seed)
    if noise_type == "gaussian":
        noise = rng.standard_normal(data.shape) * sigmas
    else:
        # Uniform on [-sqrt(3), sqrt(3)] has standard deviation 1.
        half_width = np.sqrt(3.0) * sigmas
        noise = rng.uniform(-half_width, half_width, size=data.shape)

    noisy = data + noise
    if original_ndim == 1:
        noisy = noisy[:, 0]
    return noisy
