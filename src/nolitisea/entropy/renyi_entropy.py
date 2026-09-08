"""Renyi entropies of order q."""

import numpy as np

from nolitisea.utils.rescale import rescale_data

__all__ = ["renyi_entropy"]

# Sanity limit for the integer grid count 1/eps (int64 index arithmetic).
_MAX_BOXES = 2 ** 62


def _prepare_series(series):
    """Validate the input and rescale every component to ``[0, 1]``.

    Each component passes through ``rescale_data`` and ``maxinterval``
    is the largest component interval of the *original* data, used to
    report box sizes in data units.

    Returns
    -------
    scaled : numpy.ndarray
        ``(n_times, n_vars)`` array with every column in ``[0, 1]``.
    maxinterval : float
        Largest component interval of the original data.
    """
    data = np.asarray(series, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim != 2:
        raise ValueError(
            "series must be a 1-D array or a 2-D array of shape "
            "(n_times, n_vars)"
        )
    scaled = np.empty_like(data)
    maxinterval = 0.0
    for c in range(data.shape[1]):
        scaled[:, c], _, interval = rescale_data(data[:, c])
        maxinterval = max(maxinterval, interval)
    return scaled, maxinterval


def _forward_embed(data, embed, delay):
    """Interleaved forward delay embedding.

    Column ``e * n_vars + c`` holds component ``c`` at forward delay
    index ``e`` (time ``t + e * delay`` for base time ``t``); the row
    index is ``wd = embed * n_vars + comp``.
    """
    n_times, n_vars = data.shape
    n_points = n_times - (embed - 1) * delay
    E = np.empty((n_points, n_vars * embed), dtype=np.float64)
    for e in range(embed):
        E[:, e * n_vars:(e + 1) * n_vars] = data[e * delay:e * delay + n_points, :]
    return E


def _epsilon_ladder(eps_min, eps_max, eps_count):
    """Geometric epsilon ladder.

    The ladder descends from ``eps_max`` by the factor
    ``(eps_max / eps_min) ** (1 / (eps_count - 1))``; steps whose
    integer grid count ``int(1/eps)`` does not exceed the previous one
    are skipped (duplicate grid counts).

    Returns
    -------
    deps : numpy.ndarray
        Relative box sizes (fractions of ``maxinterval``), descending.
    epsis : numpy.ndarray
        Integer grid counts ``epsi = int(1/eps)`` used at each step.
    """
    if eps_count < 1:
        raise ValueError(f"eps_count must be >= 1, got {eps_count}")
    if not 0.0 < eps_min < eps_max:
        raise ValueError(
            f"epsilon bounds must satisfy 0 < eps_min < eps_max, got "
            f"eps_min={eps_min}, eps_max={eps_max}"
        )
    if eps_count > 1:
        epsfactor = (eps_max / eps_min) ** (1.0 / (eps_count - 1))
    else:
        epsfactor = 1.0
    heps = eps_max * epsfactor
    epsi_old = 0
    deps = np.empty(eps_count, dtype=np.float64)
    epsis = np.empty(eps_count, dtype=np.int64)
    for k in range(eps_count):
        while True:
            heps /= epsfactor
            epsi_test = int(1.0 / heps)
            if epsi_test > _MAX_BOXES:
                raise ValueError(
                    f"eps={heps} is too small: the grid count exceeds the "
                    f"int64 range"
                )
            if epsi_test > epsi_old:
                break
        epsi_old = epsi_test
        deps[k] = heps
        epsis[k] = epsi_test
    return deps, epsis


def _partition_entropies(idx, q):
    """Order-``q`` Renyi entropy of the joint box partition per prefix.

    ``idx`` has shape ``(n_points, m)`` of integer box indices.
    ``h[wd]`` is computed from the counts of the unique joint boxes
    spanned by columns ``0..wd``, normalized by the total point number
    (every point falls into exactly one box at every depth).
    """
    n_points = idx.shape[0]
    h = np.empty(idx.shape[1], dtype=np.float64)
    for wd in range(idx.shape[1]):
        _, counts = np.unique(idx[:, : wd + 1], axis=0, return_counts=True)
        p = counts / n_points
        if q == 1.0:
            h[wd] = -np.sum(p * np.log(p))
        else:
            h[wd] = np.log(np.sum(p ** q)) / (1.0 - q)
    return h


def _quantize(E, epsi):
    """Box indices of the rescaled embedding at grid count ``epsi``.

    ``int(x * epsi)`` truncates; values rescaled
    exactly to ``1.0`` are clipped into the last box.
    """
    return np.minimum((E * epsi).astype(np.int64), epsi - 1)


def renyi_entropy(
    series,
    embed=10,
    delay=1,
    q=2.0,
    eps_min=None,
    eps_max=None,
    eps_count=20,
):
    """Order-``q`` Renyi entropies from box counting.

    Every component is rescaled to ``[0, 1]`` and covered by a uniform
    grid with ``epsi`` boxes per axis.  For each prefix of the interleaved 
    multivariate embedding (all components at delay ``0`` first, then delay ``1``,
    ...) the order-``q`` Renyi entropy of the joint box occupation is
    computed; ``q == 1`` gives the Shannon block entropy.

    Parameters
    ----------
    series : array_like
        Input data.  A 1-D array is a single component; a 2-D array
        must have shape ``(n_times, n_vars)`` with one column per
        component.
    embed : int
        Maximal embedding dimension per component (default 10); the
        full phase-space dimension is
        ``n_vars * embed``.
    delay : int
        Time delay.
    q : float
        Order of the Renyi entropy.  ``q == 1`` gives the Shannon
        entropy.
    eps_min, eps_max : float or None
        Ladder bounds in data units (divided by the largest component
        interval).  ``None`` uses interval/1000 and the full interval.
    eps_count : int
        Number of ladder steps.  The ladder descends
        geometrically; grid counts ``int(1/eps)`` that repeat are
        skipped.

    Returns
    -------
    dict
        ``"eps"`` : box sizes in data units, shape ``(eps_count,)``.
        ``"hq"`` : order-``q`` entropies, shape
        ``(n_vars * embed, eps_count)``; row ``wd = e * n_vars + c``
        belongs to component ``c + 1`` at embedding level ``e + 1``.
        ``"dhq"`` : increments ``hq[wd] - hq[wd - 1]`` (first row
        copies ``hq[0]``); for a
        single component these are the block-entropy increments.
        ``"components"`` : 1-based component label of each row.
        ``"embeddings"`` : 1-based embedding level of each row.

    Raises
    ------
    ValueError
        For invalid parameters, a series too short for the embedding,
        or degenerate epsilon bounds.
    RuntimeError
        For a constant component (zero range).
    """
    if not np.isfinite(q):
        raise ValueError(f"q must be a finite number, got {q}")
    if embed < 1:
        raise ValueError(f"embed must be >= 1, got {embed}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    scaled, maxinterval = _prepare_series(series)
    n_times, n_vars = scaled.shape

    n_points = n_times - (embed - 1) * delay
    if n_points < 2:
        raise ValueError(
            f"series of length {n_times} is too short for embed={embed}, "
            f"delay={delay}: only {n_points} delay vectors"
        )

    if eps_min is None:
        eps_min_rel = 1e-3
    else:
        eps_min_rel = float(eps_min) / maxinterval
    if eps_max is None:
        eps_max_rel = 1.0
    else:
        eps_max_rel = float(eps_max) / maxinterval
        if eps_max_rel > 1.0:
            raise ValueError(
                f"eps_max={eps_max} exceeds the largest component interval "
                f"{maxinterval}"
            )

    deps_rel, epsis = _epsilon_ladder(eps_min_rel, eps_max_rel, eps_count)

    E = _forward_embed(scaled, embed, delay)
    m_full = n_vars * embed
    hq = np.empty((m_full, eps_count), dtype=np.float64)
    for k in range(eps_count):
        idx = _quantize(E, int(epsis[k]))
        hq[:, k] = _partition_entropies(idx, q)

    dhq = np.empty_like(hq)
    dhq[0] = hq[0]
    dhq[1:] = hq[1:] - hq[:-1]

    return {
        "eps": deps_rel * maxinterval,
        "hq": hq,
        "dhq": dhq,
        "components": np.tile(np.arange(n_vars), embed) + 1,
        "embeddings": np.repeat(np.arange(embed), n_vars) + 1,
    }