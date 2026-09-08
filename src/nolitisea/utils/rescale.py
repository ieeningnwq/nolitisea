"""Scaling and statistics helpers."""

import numpy as np


def rescale_data(x: np.ndarray):
    """
    Performs global affine Min-Max scaling to [0, 1].

    Applies linear transformation:
        x_scaled[i] = (x[i] - minv) / interval
    where interval = max(x) - minv.
    A RuntimeError will be raised if the input sequence is nearly constant (zero range),
    matching the abort behaviour in the original C source code.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional input time-series array.

    Returns
    -------
    x_scaled : np.ndarray
        Scaled array mapped to the range [0, 1].
    minv : float
        Minimum value of original input series, used for inverse transformation.
    interval : float
        Range (max-min) of original input series, used for inverse transformation.

    Raises
    ------
    RuntimeError
        Raised when the value range of input is near zero, indicating a constant sequence.

    Notes
    -----
    This is a global affine linear transformation (shift + uniform scaling).
    The topological property of phase-space attractor will be preserved.
    Inverse transform formula: x_original = x_scaled * interval + minv
    """
    minv = np.min(x)
    interval = np.max(x) - minv
    if abs(interval) < 1e-30:
        raise RuntimeError("rescale_data: zero interval, constant sequence, abort.")
    x_scaled = (x - minv) / interval
    return x_scaled, minv, interval

