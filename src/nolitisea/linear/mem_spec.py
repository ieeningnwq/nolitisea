"""Maximum-entropy (AR-based) spectrum estimation."""

import numpy as np

__all__ = ["mem_spectrum"]


def _burg_coefs(x, order):
    """Estimate AR coefficients via Burg's method.

    Returns ``(coeffs, sigma2)`` where ``coeffs`` has length ``order``
    and ``sigma2`` is the residual variance (prediction-error power).
    """
    n = x.size

    # Forward and backward prediction errors.
    # fwd[j] = x[j], bwd[j] = x[j+1] for j = 0..n-2.
    fwd = x[:-1].copy()
    bwd = x[1:].copy()

    coeffs = np.zeros(order)
    help_arr = np.zeros(order)
    sigma2 = np.dot(x, x) / n

    for m in range(order):
        h1 = np.dot(fwd, bwd)
        h2 = np.dot(fwd, fwd) + np.dot(bwd, bwd)
        k = 2.0 * h1 / h2
        coeffs[m] = k

        if m > 0:
            coeffs[:m] = help_arr[:m] - k * help_arr[m - 1::-1]

        sigma2 *= 1.0 - k * k

        if m == order - 1:
            break

        help_arr = coeffs.copy()

        f_new = fwd[:-1] - k * bwd[:-1]
        b_new = bwd[1:] - k * fwd[1:]
        fwd = f_new
        bwd = b_new

    return coeffs, sigma2


def _power_spectrum(coeffs, sigma2, n_freq, n):
    """Evaluate the MEM spectrum at ``n_freq`` points in [0, 0.5).

    The spectrum is ``sigma2 / |A(f)|^2 / sqrt(n)`` where
    ``A(f) = 1 - sum_i coeffs[i] * exp(j*2*pi*f*(i+1))``.
    """
    freqs = np.arange(n_freq) / (2.0 * n_freq)
    omdt = 2.0 * np.pi * freqs
    hr = np.cos(omdt)
    hi = np.sin(omdt)

    zr = np.ones(n_freq)
    zi = np.zeros(n_freq)
    sr = np.ones(n_freq)
    si = np.zeros(n_freq)

    for c in coeffs:
        zr, zi = zr * hr - zi * hi, zr * hi + zi * hr
        sr -= c * zr
        si -= c * zi

    return freqs, sigma2 / (sr * sr + si * si) / np.sqrt(n)


def mem_spectrum(series, order, n_freq=512):
    """Estimate the power spectrum via the maximum-entropy method.

    Parameters
    ----------
    series : array_like
        Input series (1-D).  The mean is subtracted internally.
    order : int
        AR model order (number of poles).  Must be smaller than the
        series length.
    n_freq : int, default 512
        Number of frequency bins, spanning ``[0, 0.5)`` in cycles per
        sample.

    Returns
    -------
    tuple
        ``(freqs, power)`` arrays of length ``n_freq``.  ``freqs`` are
        normalised frequencies in ``[0, 0.5)``; ``power`` is the
        maximum-entropy spectral estimate.

    Raises
    ------
    ValueError
        If ``order >= len(series)`` or ``series`` is not 1-D.

    Notes
    -----
    The AR coefficients are estimated with Burg's method.  The spectrum is

    .. math:: P(f) = \\frac{\\sigma^2}{|A(f)|^2 \\sqrt{N}}

    where :math:`A(f) = 1 - \\sum_{i=1}^{p} a_i \\, e^{j 2\\pi f i}`
    and :math:`\\sigma^2` is the residual variance.

    References
    ----------
    .. [1] Burg, J. P. (1975). *Maximum Entropy Spectral Analysis*.
           Ph.D. thesis, Stanford University.
    .. [2] Hegger, R., Kantz, H., & Schreiber, T. (1999).  Practical
           implementation of nonlinear time series methods: The TISEAN
           package.  *Chaos*, 9(2), 413-435.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("series must be a 1-D array")
    n = x.size

    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if order >= n:
        raise ValueError(
            f"order ({order}) must be smaller than series length ({n})"
        )
    if n_freq < 1:
        raise ValueError(f"n_freq must be >= 1, got {n_freq}")

    x = x - np.mean(x)

    coeffs, sigma2 = _burg_coefs(x, order)

    return _power_spectrum(coeffs, sigma2, n_freq, n)
