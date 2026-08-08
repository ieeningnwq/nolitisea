from scipy.signal import periodogram, welch
import numpy as np

# ============================================================
#  Noise generation helpers
# ============================================================


def _make_colored_noise(rng, n, alpha, sigma):
    """
    Generate 1/f^alpha (colored) noise with desired standard deviation.

    Uses the Fourier-domain method (rfft/irfft for efficiency):
        S(f) ~ 1/|f|^alpha   (with f=0 term set to 0 to avoid divergence)
    The resulting noise is zero-mean and has std approximately `sigma`.

    Parameters
    ----------
    rng : numpy.random.Generator
    n : int
        Number of samples.
    alpha : float
        Exponent in 1/f^alpha. Typical values:
          alpha = 0   -> white noise
          alpha = 1   -> pink (flicker) noise
          alpha = 2   -> red (Brownian / random-walk) noise
    sigma : float
        Target standard deviation of the output.

    Returns
    -------
    ndarray of shape (n,)
    """
    if n <= 1:
        return np.zeros(n)

    mag = np.zeros(n // 2 + 1)
    k = np.arange(1, n // 2 + 1)
    mag[1:] = 1.0 / (k ** (alpha / 2.0))
    # mag[0] (DC) stays 0

    half = n // 2 + 1
    phases = rng.normal(size=half) + 1j * rng.normal(size=half)
    phases[0] = phases[0].real
    if n % 2 == 0:
        phases[-1] = phases[-1].real

    noise_rfft = mag * phases
    noise = np.fft.irfft(noise_rfft, n=n)

    noise = noise - np.mean(noise)
    cur_std = np.std(noise)
    if cur_std > 0:
        noise = noise * (sigma / cur_std)

    return noise


def _make_harmonic_noise(rng, n, amplitudes, frequencies, phases=None, sigma=None):
    """
    Generate harmonic (sum-of-sinusoids) noise.

    x[n] = sum_k  A_k * sin(2*pi*f_k*n + phi_k)

    Parameters
    ----------
    rng : numpy.random.Generator
    n : int
        Number of samples.
    amplitudes : array-like
        Amplitudes A_k for each harmonic component.
    frequencies : array-like
        Normalized frequencies f_k in [0, 0.5] (cycles per sample).
    phases : array-like or None
        Initial phases phi_k in radians. If None, randomly drawn from [0, 2*pi).
    sigma : float or None
        If given, rescale the resulting signal to have this standard deviation.

    Returns
    -------
    ndarray of shape (n,)
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if phases is None:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=amplitudes.shape)
    else:
        phases = np.asarray(phases, dtype=float)

    if not (amplitudes.shape == frequencies.shape == phases.shape):
        raise ValueError(
            "amplitudes, frequencies, and phases must have the same shape.")

    t = np.arange(n)
    noise = np.zeros(n)
    for A, f, phi in zip(amplitudes, frequencies, phases):
        noise += A * np.sin(2.0 * np.pi * f * t + phi)

    noise = noise - np.mean(noise)
    cur_std = np.std(noise)
    if sigma is not None and cur_std > 0:
        noise = noise * (sigma / cur_std)

    return noise


def _make_impulse_noise(rng, n, prob, amplitude, bipolar=True):
    """
    Generate impulse (salt-and-pepper) noise.

    Each sample has probability `prob` of being replaced by a large
    impulse value.  Impulses are zero-mean overall.

    Parameters
    ----------
    rng : numpy.random.Generator
    n : int
        Number of samples.
    prob : float
        Probability that any given sample is corrupted (0 <= prob <= 1).
    amplitude : float
        Magnitude of each impulse.  The actual value is either
        +amplitude or -amplitude (if bipolar=True), giving zero mean.
    bipolar : bool
        If True, impulses take values +/- amplitude with equal probability
        (zero-mean).  If False, impulses are all +amplitude (positive only),
        mimicking "salt" noise.

    Returns
    -------
    ndarray of shape (n,)
    """
    noise = np.zeros(n)

    if prob <= 0 or n == 0:
        return noise

    mask = rng.random(n) < prob

    if bipolar:
        signs = rng.choice([-1.0, 1.0], size=n)
        noise[mask] = amplitude * signs[mask]
    else:
        noise[mask] = amplitude

    return noise


def add_noise(
    data,
    noise_type="gaussian",
    level=0.05,
    absolute=False,
    seed=None,
    # ---- colored noise ----
    alpha=1.0,
    # ---- harmonic noise ----
    amplitudes=(1.0,),
    frequencies=(0.05,),
    phases=None,
    # ---- impulse noise ----
    impulse_prob=0.05,
    impulse_amplitude=3.0,
    impulse_bipolar=True,
):
    """
    Add noise to a numpy array (1D or 2D).

    For 2D data, noise is added to ALL columns independently.

    Parameters
    ----------
    data : array-like
        Input data, converted to float numpy array. 1D or 2D.
    noise_type : str
        One of:
          "gaussian"  - zero-mean normal noise
          "uniform"   - uniform in [-a, a]
          "colored"   - 1/f^alpha (colored) noise
          "harmonic"  - sum of sinusoidal components
          "impulse"   - salt-and-pepper impulse noise
    level : float
        Noise level:
          - gaussian/uniform/colored: as described above
          - harmonic: scales amplitudes
          - impulse (absolute=False): probability of impulse per sample
          - impulse (absolute=True): impulse amplitude (half-width)
    absolute : bool
        If True, `level` is an absolute scale.
    seed : int or None
        Random seed for reproducibility.
    alpha : float
        Exponent for colored noise (1/f^alpha). 0=white, 1=pink, 2=red.
        Only used when noise_type="colored".
    amplitudes : array-like
        Amplitudes of harmonic components. Only for noise_type="harmonic".
    frequencies : array-like
        Normalized frequencies (cycles per sample, 0 < f < 0.5).
        Only for noise_type="harmonic".
    phases : array-like or None
        Phases in radians. None = random. Only for noise_type="harmonic".
    impulse_prob : float
        Probability that a given sample is replaced by an impulse.
        Only for noise_type="impulse".  Overrides `level` when absolute=False.
    impulse_amplitude : float
        Amplitude of each impulse (+/-).  Only for noise_type="impulse".
        Overrides `level` when absolute=True.
    impulse_bipolar : bool
        If True (default), impulses are +/- amplitude (zero-mean).
        If False, impulses are all +amplitude ("salt" noise).

    Returns
    -------
    noisy : ndarray
        Data with noise added (same shape as input).

    Raises
    ------
    ValueError
        If data is empty, has wrong shape, or parameters are invalid.
    """

    # ---- Validate input ----
    if data is None:
        raise ValueError("Input data is None. Please provide valid data.")

    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError(
            "Input data is empty (zero elements). Cannot add noise.")

    if data.ndim not in (1, 2):
        raise ValueError(
            f"Input data must be 1D or 2D, got {data.ndim}D array."
        )

    noise_type = noise_type.lower()
    valid_types = ("gaussian", "uniform", "colored", "harmonic", "impulse")
    if noise_type not in valid_types:
        raise ValueError(
            f"noise_type must be one of {valid_types}, got '{noise_type}'."
        )

    if level < 0:
        raise ValueError(f"Noise level must be non-negative, got {level}.")

    # ---- Initialize RNG ----
    rng = np.random.default_rng(seed)

    # ---- Work on a copy ----
    noisy = data.copy()

    # ---- Determine shape for iteration ----
    if data.ndim == 1:
        n_rows = data.shape[0]
        n_cols = 1
        # wrap 1D as 2D for uniform processing
        data_view = data.reshape(-1, 1)
        noisy_view = noisy.reshape(-1, 1)
    else:
        n_rows, n_cols = data.shape
        data_view = data
        noisy_view = noisy

    # ---- Add noise column by column ----
    for col_idx in range(n_cols):
        signal = data_view[:, col_idx]
        out = noisy_view[:, col_idx]
        n = signal.shape[0]

        # Determine noise scale
        if noise_type == "impulse":
            prob = impulse_prob
            if absolute:
                amp = level
            else:
                s = np.std(signal)
                if s < 1e-12:
                    raise ValueError(
                        "impulse noise with absolute=False requires signal with non‑zero standard deviation. "
                        "For constant signals, set absolute=True and specify impulse amplitude via level."
                    )
                amp = impulse_amplitude * s
            sigma = None
        elif absolute:
            if noise_type in ("gaussian", "colored"):
                sigma = level
            elif noise_type == "uniform":
                sigma = level / np.sqrt(3.0)
            else:
                sigma = None
        else:
            sigma_data = np.std(signal)
            sigma = level * sigma_data

        # ---- Generate noise ----
        if noise_type == "gaussian":
            noise = rng.normal(loc=0.0, scale=sigma, size=n)

        elif noise_type == "uniform":
            if absolute:
                a, b = -level, level
            else:
                a, b = -level * np.std(signal), level * np.std(signal)
            noise = rng.uniform(low=a, high=b, size=n)

        elif noise_type == "colored":
            noise = _make_colored_noise(rng, n, alpha=alpha, sigma=sigma)

        elif noise_type == "harmonic":
            if absolute:
                amps = np.asarray(amplitudes, dtype=float)
                noise = _make_harmonic_noise(
                    rng, n,
                    amplitudes=amps,
                    frequencies=frequencies,
                    phases=phases,
                    sigma=None,
                )
            else:
                amps = np.asarray(amplitudes, dtype=float) * level * sigma_data
                noise = _make_harmonic_noise(
                    rng, n,
                    amplitudes=amps,
                    frequencies=frequencies,
                    phases=phases,
                    sigma=None,
                )

        elif noise_type == "impulse":
            noise = _make_impulse_noise(
                rng, n,
                prob=prob,
                amplitude=amp,
                bipolar=impulse_bipolar,
            )
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        out[:] = signal + noise

    return noisy

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
            x -= np.polyval(np.polyfit(np.arange(len(x)), x, 1),
                            np.arange(len(x)))
        elif callable(detrend):
            x = detrend(x)
        elif detrend is None or detrend is False:
            pass  # no detrending
        else:
            raise ValueError(
                f"Invalid detrend option '{detrend}' for method='fft'.")
        n = len(x)
        fft_vals = np.fft.rfft(x)
        freq = np.fft.rfftfreq(n, d=1.0 / fs)

        if scaling == "density":
            psd = np.abs(fft_vals) ** 2 / (fs * n)
        else:  # "spectrum"
            psd = np.abs(fft_vals) ** 2 / n

        if not return_onesided:
            raise ValueError(
                "return_onesided=False not supported for method='fft'")

    else:
        raise ValueError(f"Unknown method '{method}'.")

    return freq, psd
