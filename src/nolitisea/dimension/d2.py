"""Correlation sum, dimension and entropy estimates."""

import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import lag_block_delay_embed

__all__ = ["d2"]


def _default_ladder(data, howoften, eps_min, eps_max):
    """Geometric epsilon ladder."""
    maxrange = float(np.ptp(data, axis=0).max())
    if eps_max is None:
        eps_max = maxrange
    else:
        eps_max = min(abs(float(eps_max)), maxrange)
    if maxrange <= 0.0 or eps_max <= 0.0:
        raise ValueError(
            "default eps bounds require a non-constant series; "
            "pass an explicit eps array instead"
        )
    if eps_min is None:
        eps_min = maxrange / 1000.0
    else:
        # EPSMIN = |EPSMIN| < EPSMAX ? |EPSMIN| : EPSMAX / 2
        eps_min = abs(float(eps_min))
        if eps_min >= eps_max:
            eps_min = eps_max / 2.0
    if eps_min <= 0.0:
        raise ValueError(f"eps_min must be > 0, got {eps_min}")
    epsfactor = (eps_max / eps_min) ** (1.0 / (howoften - 1))
    return eps_max / epsfactor ** np.arange(howoften)


def _shift_running_max(E, theiler):
    """Running coordinate maxima of shifted differences for the Theiler
    correction.

    Entry ``k - 1`` holds, for shift ``k``, the matrix
    ``max_{c <= m} |E[i, c] - E[i + k, c]|`` accumulated over columns,
    so column ``m - 1`` is the order-``m`` embedding distance of the
    time-``k`` neighbour pair ``(i, i + k)``.
    """
    running = []
    for k in range(1, theiler + 1):
        if k >= E.shape[0]:
            break
        diff = np.abs(E[:-k] - E[k:])
        running.append(np.maximum.accumulate(diff, axis=1))
    return running


def _eligible_norm(n_points, theiler):
    """Number of unordered pairs with time separation > ``theiler``."""
    norm = n_points * (n_points - 1) // 2
    for k in range(1, theiler + 1):
        norm -= max(n_points - k, 0)
    return norm


def _pair_counts(E_m, eps_asc):
    """Unordered pairs within each radius (ascending ladder).

    ``count_neighbors`` returns ordered counts including the ``n``
    self pairs (distance 0) and both orientations of every non-self
    pair, so the unordered count is ``(raw - n) / 2``.
    """
    n_points = E_m.shape[0]
    tree = cKDTree(E_m)
    raw = np.asarray(tree.count_neighbors(tree, eps_asc, p=np.inf),
                     dtype=np.float64)
    return (raw - n_points) / 2.0


def _subtract_theiler(counts, E_m, eps_asc, shift_running):
    """Subtract Theiler-violating pairs from per-radius counts.

    ``shift_running[k - 1]`` holds the running coordinate maxima of the
    shift-``k`` differences over all ``m_full`` columns, so column
    ``m - 1`` is the order-``m`` distance of every pair with time
    separation exactly ``k``.
    """
    if not shift_running:
        return counts
    m = E_m.shape[1]
    for k, shift in enumerate(shift_running, start=1):
        if k >= E_m.shape[0]:
            break
        col = np.sort(shift[:, m - 1])
        counts = counts - np.searchsorted(col, eps_asc, side="right")
    return counts


def d2(
    series,
    embed=10,
    delay=1,
    theiler=0,
    eps=None,
    howoften=100,
    eps_min=None,
    eps_max=None,
):
    """Correlation sum, D2 and H2 for embeddings up to ``n_vars * embed``.

    Parameters
    ----------
    series : array_like
        Input data.  A 1-D array is a single component; a 2-D array
        must have shape ``(n_times, n_vars)`` with one column per
        component.
    embed : int
        Maximum embedding dimension per component;
        the full phase-space dimension is ``n_vars * embed``.  Correlation
        sums are reported for every prefix order ``1 .. n_vars * embed``.
    delay : int
        Time delay.
    theiler : int
        Theiler window: pairs with time separation ``<= theiler`` are excluded.
    eps : array_like or None
        Explicit radii.  When given, ``howoften``, ``eps_min`` and
        ``eps_max`` are ignored; at least two radii are required for
        the slope output.
    howoften : int
        Number of ladder radii when ``eps`` is not given. The ladder descends geometrically from ``eps_max``
        to ``eps_min``.
    eps_min, eps_max : float or None
        Ladder bounds in data units.
        ``None`` uses: the largest component interval and its thousandth part.

    Returns
    -------
    dict
        ``"eps"`` : descending radii, shape ``(n_eps,)``.
        ``"c2"`` : correlation sums, shape ``(n_vars * embed, n_eps)``;
        row ``i`` is embedding order ``i + 1``.
        ``"h2"`` : order-2 Renyi entropy estimates
        (``-log C2`` for order 1, ``log C2[m-1]/C2[m]`` for higher
        orders), same shape as ``c2``; ``nan`` where undefined.
        ``"d2"`` : local slopes ``log(C2(eps_j)/C2(eps_{j+1})) /
        log(eps_j/eps_{j+1})``, shape ``(n_vars * embed, n_eps - 1)``.
        ``"norm"`` : number of eligible (Theiler-corrected) pairs.

    Raises
    ------
    ValueError
        For invalid parameters, a series too short for the embedding,
        or degenerate epsilon bounds.
    """
    data = np.asarray(series, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim != 2:
        raise ValueError(
            "series must be a 1-D array or a 2-D array of shape "
            "(n_times, n_vars)"
        )
    length, n_vars = data.shape

    if embed < 1:
        raise ValueError(f"embed must be >= 1, got {embed}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if theiler < 0:
        raise ValueError(f"theiler must be >= 0, got {theiler}")

    n_points = length - (embed - 1) * delay
    if n_points < 2:
        raise ValueError(
            f"series of length {length} is too short for embed={embed}, "
            f"delay={delay}: only {n_points} delay vectors"
        )

    if eps is None:
        if howoften < 2:
            raise ValueError(f"howoften must be >= 2, got {howoften}")
        eps_arr = _default_ladder(data, howoften, eps_min, eps_max)
    else:
        eps_arr = np.sort(np.asarray(eps, dtype=np.float64).ravel())[::-1]
        if eps_arr.size < 2:
            raise ValueError("eps must contain at least two radii")
        if (eps_arr <= 0.0).any():
            raise ValueError("all radii must be > 0")

    E = lag_block_delay_embed(data, embed, delay)
    m_full = n_vars * embed
    shift_running = _shift_running_max(E, theiler) if theiler else []

    norm = _eligible_norm(n_points, theiler)
    if norm <= 0:
        raise ValueError(
            f"theiler={theiler} excludes every pair among {n_points} "
            "delay vectors"
        )

    eps_asc = eps_arr[::-1]
    c2 = np.empty((m_full, eps_arr.size), dtype=np.float64)
    for m in range(1, m_full + 1):
        E_m = np.ascontiguousarray(E[:, :m])
        counts = _pair_counts(E_m, eps_asc)
        counts = _subtract_theiler(counts, E_m, eps_asc, shift_running)
        # counts follow the ascending ladder; store in eps_arr (descending).
        c2[m - 1] = counts[::-1] / norm

    # Order-2 Renyi entropy estimates.
    h2 = np.full_like(c2, np.nan)
    pos = c2[0] > 0.0
    h2[0, pos] = -np.log(c2[0, pos])
    if m_full > 1:
        num = c2[:-1]
        den = c2[1:]
        pos = (num > 0.0) & (den > 0.0)
        h2[1:][pos] = np.log(num[pos] / den[pos])

    # Local slopes of the correlation dimension.
    deps = np.log(eps_arr[:-1] / eps_arr[1:])
    deps = np.broadcast_to(deps, (m_full, eps_arr.size - 1))
    slopes = np.full((m_full, eps_arr.size - 1), np.nan)
    pos = (c2[:, :-1] > 0.0) & (c2[:, 1:] > 0.0) & (deps > 0.0)
    slopes[pos] = np.log(c2[:, :-1][pos] / c2[:, 1:][pos]) / deps[pos]

    return {
        "eps": eps_arr,
        "c2": c2,
        "h2": h2,
        "d2": slopes,
        "norm": int(norm),
    }
