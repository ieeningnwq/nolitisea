"""Linear filters: notch, Wiener, 1-2-1 low-pass, Savitzky-Golay."""

from math import factorial

import numpy as np
from scipy.signal import filtfilt, iirnotch


def _as_1d(series):
    """Convert ``series`` to a 1-D float array, validating shape and size."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input series must be 1-D.")
    if arr.size == 0:
        raise ValueError("Input series is empty.")
    return arr


def notch_filter(series, freq, q=30.0, fs=1.0):
    """Apply a notch filter rejecting ``freq``.

    A second-order IIR notch (biquad) centred at ``freq`` is designed with
    quality factor ``q`` and applied twice (forward-backward), giving a
    zero-phase response with unity gain everywhere except the rejected
    band.

    Parameters
    ----------
    series : array_like
        Input series.
    freq : float
        Frequency to reject (in cycles per unit time).
    q : float, default 30.0
        Quality factor.  Higher ``q`` gives a narrower notch.
    fs : float, default 1.0
        Sampling frequency.

    Returns
    -------
    numpy.ndarray
        Filtered series (same length as the input).
    """
    x = _as_1d(series)
    if fs <= 0.0:
        raise ValueError("fs must be positive.")
    if not 0.0 < freq < fs / 2.0:
        raise ValueError(f"freq must lie in (0, fs/2) = (0, {fs / 2.0}); got {freq}.")
    if q <= 0.0:
        raise ValueError("Quality factor q must be positive.")

    # Forward-backward application needs padding strictly shorter than x.
    if x.size < 3:
        return x.copy()

    b, a = iirnotch(freq, q, fs=fs)
    padlen = min(3 * max(len(a), len(b)), x.size - 1)
    return filtfilt(b, a, x, padlen=padlen)


def wiener_filter(series, noise_var):
    """Apply a frequency-domain Wiener filter.

    Each Fourier bin is attenuated by the non-causal Wiener gain

    .. math::

        H_k = \\max\\!\\left(1 - \\frac{N\\,\\sigma_n^2}{|X_k|^2},\\ 0\\right),

    where ``N`` is the series length and ``noise_var`` is the estimated
    noise variance.  Bins dominated by noise are suppressed while bins
    carrying signal power are kept.

    Parameters
    ----------
    series : array_like
        Input series.
    noise_var : float
        Estimated noise variance (non-negative).  ``0`` leaves the series
        unchanged; very large values drive the output to zero.

    Returns
    -------
    numpy.ndarray
        Filtered series (same length as the input).
    """
    x = _as_1d(series)
    if noise_var < 0.0:
        raise ValueError("noise_var must be non-negative.")

    n = x.size
    spectrum = np.fft.fft(x)
    power = np.abs(spectrum) ** 2

    # For white noise of variance ``noise_var`` the unnormalised periodogram
    # |X_k|^2 has expectation n * noise_var per bin.
    gain = np.ones(n)
    active = power > 0.0
    gain[active] = np.clip(1.0 - n * noise_var / power[active], 0.0, 1.0)

    return np.real(np.fft.ifft(gain * spectrum))


def low121(series):
    """Apply a simple 1-2-1 low-pass filter.

    The three-tap binomial kernel ``[1, 2, 1] / 4`` is applied once.  It
    preserves constant and linear sequences exactly and reduces the
    variance of white noise by a factor ``3/8`` (interior samples).
    Boundary samples use edge padding.

    Parameters
    ----------
    series : array_like
        Input series.

    Returns
    -------
    numpy.ndarray
        Filtered series (same length as the input).
    """
    x = _as_1d(series)
    kernel = np.array([0.25, 0.5, 0.25])
    padded = np.pad(x, 1, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _savgol_weights(window, order, deriv):
    """Least-squares Savitzky-Golay weight vectors for every window slot.

    Returns an array of shape ``(window, window)`` whose row ``p`` gives
    the weights combining the samples of a fitted window into the value
    (or ``deriv``-th derivative) of the fitted polynomial evaluated at the
    window slot ``p`` (local coordinate ``p - (window - 1) / 2``).
    """
    half = (window - 1) // 2
    pos = np.arange(-half, half + 1, dtype=float)
    # Vandermonde design matrix: A[i, j] = pos[i] ** j.
    design = np.power(pos[:, None], np.arange(order + 1)[None, :])
    pinv = np.linalg.pinv(design)  # (order + 1, window)

    weights = np.zeros((window, window))
    for p in range(window):
        for j in range(deriv, order + 1):
            # d/dp^deriv of p**j is j! / (j - deriv)! * p**(j - deriv);
            # c_j = pinv[j] . y are the fitted polynomial coefficients.
            factor = factorial(j) // factorial(j - deriv)
            weights[p] += factor * pos[p] ** (j - deriv) * pinv[j]
    return weights


def savitzky_golay(series, window, order, deriv=0):
    """Apply a Savitzky-Golay smoothing/derivative filter.

    A local polynomial of degree ``order`` is fitted by least squares over
    a sliding window of ``window`` points; the fitted value (or its
    ``deriv``-th derivative, in units per sample) is returned at each
    sample.  Boundary samples are evaluated from the polynomial fitted on
    the first/last window (interpolation mode), so the output has the same
    length as the input.

    Parameters
    ----------
    series : array_like
        Input series.
    window : int
        Window length (must be odd).
    order : int
        Polynomial order (must be smaller than ``window``).
    deriv : int, default 0
        Derivative order (0 = smoothing).  Must not exceed ``order``.

    Returns
    -------
    numpy.ndarray
        Filtered series (same length as the input).
    """
    x = _as_1d(series)
    if not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError("window must be a positive integer.")
    window = int(window)
    if window % 2 == 0:
        raise ValueError("window must be odd.")
    if not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("order must be a non-negative integer.")
    order = int(order)
    if order >= window:
        raise ValueError("order must be smaller than window.")
    if not isinstance(deriv, (int, np.integer)) or deriv < 0:
        raise ValueError("deriv must be a non-negative integer.")
    deriv = int(deriv)
    if deriv > order:
        raise ValueError("deriv must not exceed order.")
    if x.size < window:
        raise ValueError(f"Input length ({x.size}) must be at least window ({window}).")

    weights = _savgol_weights(window, order, deriv)
    half = (window - 1) // 2
    n = x.size
    y = np.empty(n)

    # Left edge: polynomial fitted on the first window, evaluated at slots.
    for i in range(half):
        y[i] = np.dot(weights[i], x[:window])
    # Interior: central slot weights applied as a convolution.
    interior = np.convolve(x, weights[half][::-1], mode="valid")
    y[half : n - half] = interior
    # Right edge: polynomial fitted on the last window.
    for i in range(n - half, n):
        slot = i - (n - window)
        y[i] = np.dot(weights[slot], x[n - window :])
    return y
