"""Space-time separation plot (TISEAN ``stp``).

Python rewrite of the TISEAN Fortran program ``stp.f`` (Hegger, Kantz
and Schreiber, *Chaos* 9, 413 (1999)).  For every time shift ``t`` it
accumulates the Chebyshev distance between each embedding vector and
its ``t``-shifted copy into a fixed-bin histogram, then reads off the
distance at several cumulative fractions.  The result is the curve of
space-time separation used to choose a Theiler window.
"""

import numpy as np

__all__ = ["stp"]

# Histogram bin count (Fortran parameter ``meps``).
_N_BINS = 1000
# Cap on the number of time steps (Fortran parameter ``mdt``).
_MAX_TIME = 500
# Cap on the number of fraction levels (Fortran parameter ``mfrac``).
_MAX_FRAC = 100


def stp(series, dim, delay, max_time=100, resolution=1, fraction=0.05,
        n_bins=_N_BINS):
    """Space-time separation plot via the Chebyshev distance histogram.

    Parameters
    ----------
    series : array_like
        Input scalar series (1-D).
    dim : int
        Embedding dimension (Fortran ``m``, option ``-m``); must be >= 1.
    delay : int
        Time delay (Fortran ``id``, option ``-d``); must be >= 1.
    max_time : int, default 100
        Number of time shifts evaluated (Fortran ``ndt``, option ``-t``);
        capped at 500 (Fortran ``mdt``).
    resolution : int, default 1
        Time resolution: the ``t``-th shift is ``t * resolution`` samples
        (Fortran ``idt``, option ``-#``); must be >= 1.
    fraction : float, default 0.05
        Step between cumulative fraction levels (Fortran ``perc``,
        option ``-%``).  The number of levels is
        ``min(100, int(1 / fraction))`` and the levels are
        ``1/n_frac, 2/n_frac, ..., 1.0``.
    n_bins : int, default 1000
        Number of distance histogram bins over ``[0, epsmax]`` (Fortran
        ``meps``); the returned distances are quantised to
        ``epsmax / n_bins``.

    Returns
    -------
    dict
        ``"times"`` : the time shifts ``t * resolution`` for
        ``t = 1 .. max_time``, shape ``(max_time,)``.
        ``"fractions"`` : the cumulative fraction levels, shape
        ``(n_frac,)``.
        ``"stp"`` : the separation distance at every fraction and time
        shift, shape ``(n_frac, max_time)``; ``nan`` where no pair was
        available for that shift.

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the requested
        ``dim``/``delay``/``max_time``/``resolution``.

    Notes
    -----
    The Fortran bins distances with ``int(meps * dis / epsmax) + 1``
    (clamped to ``meps``); the separation at fraction ``f`` is the upper
    edge of the first bin whose cumulative count reaches
    ``f * n_pairs``.  This rewrite reproduces that bin-based quantile
    exactly, so it agrees with ``stp.f`` to the bin resolution.

    References
    ----------
    .. [1] Kantz, H., & Schreiber, T. (1997, 2004). *Nonlinear Time
           Series Analysis*.  Cambridge University Press.
    .. [2] Hegger, R., Kantz, H., & Schreiber, T. (1999). Practical
           implementation of nonlinear time series methods: The TISEAN
           package.  *Chaos*, 9(2), 413-435.
    """
    y = np.asarray(series, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("series must be a 1-D array")
    nmax = y.size

    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if resolution < 1:
        raise ValueError(f"resolution must be >= 1, got {resolution}")
    if max_time < 1:
        raise ValueError(f"max_time must be >= 1, got {max_time}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    max_time = min(max_time, _MAX_TIME)
    n_frac = min(_MAX_FRAC, int(1.0 / fraction))
    n_frac = max(n_frac, 1)
    frac_levels = np.arange(1, n_frac + 1, dtype=np.float64) / n_frac

    epsmax = y.max() - y.min()
    if epsmax < 1e-30:
        raise RuntimeError("stp: zero data interval, constant sequence.")

    pstart = (dim - 1) * delay
    times = np.arange(1, max_time + 1, dtype=np.int64) * resolution
    stp = np.full((n_frac, max_time), np.nan)

    # Precompute the absolute lag-difference array for every shift; the
    # Chebyshev distance between the embeddings at ``n`` and ``n-t`` is
    # the max of |y[k] - y[k-t]| over the ``dim`` coordinates
    # k = n, n-delay, ..., n-(dim-1)*delay.
    coord_offsets = pstart - np.arange(dim) * delay  # [pstart, ..., 0]

    for it in range(max_time):
        lag = times[it]
        if lag + pstart >= nmax:
            continue  # no embedding pair survives this shift
        # d[j] = |y[lag + j] - y[j]|, j = 0 .. nmax-1-lag, i.e. the
        # lag-difference at base index k = lag + j.
        d = np.abs(y[lag:] - y[:nmax - lag])
        # Reference rows n in [lag + pstart, nmax-1]; m_idx = n - lag - pstart.
        n_pairs = nmax - lag - pstart
        m_idx = np.arange(n_pairs)[:, None]  # (n_pairs, 1)
        # Indices into d for the dim coordinates: n - lag - me*delay
        # = pstart + m_idx - me*delay = m_idx + (pstart - me*delay).
        dis = d[m_idx + coord_offsets[None, :]].max(axis=1)  # (n_pairs,)

        # Bin distances: bin = int(n_bins * dis / epsmax) + 1, clamped.
        bins = np.minimum(
            (n_bins * dis / epsmax).astype(np.int64) + 1, n_bins
        )
        ihist = np.bincount(bins, minlength=n_bins + 1)[1:n_bins + 1]
        cum = np.cumsum(ihist, dtype=np.float64)

        # First bin whose cumulative count reaches need = n_pairs * f.
        # The Fortran evaluates ``need = (n_pairs * ifrac) / nfrac``
        # (integer product first, then real division); mirror that
        # order so the threshold lands on the same side of every integer
        # cumulative count as the reference.
        ifrac = np.arange(1, n_frac + 1, dtype=np.int64)
        needs = (n_pairs * ifrac).astype(np.float64) / n_frac
        ieps = np.searchsorted(cum, needs, side="left") + 1  # 1-based
        ieps = np.clip(ieps, 1, n_bins)
        stp[:, it] = ieps * (epsmax / n_bins)

    return {"times": times, "fractions": frac_levels, "stp": stp}
