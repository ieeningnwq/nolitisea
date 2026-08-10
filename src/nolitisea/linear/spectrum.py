"""FFT-based power spectrum estimation (TISEAN ``spectrum``)."""

from __future__ import annotations

import numpy as np
from scipy.signal import periodogram, welch


def power_spectrum(
    x: np.ndarray,
    *,
    method: str = "welch",
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend="constant",
    scaling: str = "density",
    return_onesided: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum of a 1D real-valued signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal (will be flattened to 1D).
    method : {"periodogram", "welch", "fft"}
        Spectrum estimation method.
    fs : float
        Sampling frequency.
    nperseg : int, optional
        Length of each segment (Welch only).
    noverlap : int, optional
        Overlap between segments (Welch only).
    detrend : {"constant", "linear", function}
        Detrending applied before spectrum estimation.
    scaling : {"density", "spectrum"}
        Scaling mode.
    return_onesided : bool
        If True, return one-sided spectrum for real signals.

    Returns
    -------
    freq : np.ndarray
        Frequency bins.
    psd : np.ndarray
        Power spectral density or power spectrum.
    """
    x = x.ravel()

    if x.size == 0:
        raise ValueError("Input vector is empty.")

    if x.size < 2:
        raise ValueError("Input vector too short (need at least 2 samples).")

    method = method.lower()

    if method == "periodogram":
        freq, psd = periodogram(
            x,
            fs=fs,
            detrend=detrend,
            scaling=scaling,
            return_onesided=return_onesided,
        )

    elif method == "welch":
        freq, psd = welch(
            x,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling=scaling,
            return_onesided=return_onesided,
        )

    elif method == "fft":
        x = x.copy()
        if detrend == "constant":
            x -= np.mean(x)
        elif detrend == "linear":
            x -= np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
        elif callable(detrend):
            x = detrend(x)
        elif detrend is None or detrend is False:
            pass  # no detrending
        else:
            raise ValueError(f"Invalid detrend option '{detrend}' for method='fft'.")
        n = len(x)
        fft_vals = np.fft.rfft(x)
        freq = np.fft.rfftfreq(n, d=1.0 / fs)

        if scaling == "density":
            psd = np.abs(fft_vals) ** 2 / (fs * n)
        else:  # "spectrum"
            psd = np.abs(fft_vals) ** 2 / n

        if not return_onesided:
            raise ValueError("return_onesided=False not supported for method='fft'")
    else:
        raise ValueError(f"Unknown method '{method}'.")

    return freq, psd
