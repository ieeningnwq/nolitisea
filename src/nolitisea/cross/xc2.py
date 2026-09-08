"""Cross-correlation integral of two data sets."""

import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import lag_block_delay_embed

__all__ = ["cross_correlation_integral"]

# Reference rows processed per KD-tree query chunk.
_CHUNK = 512


def _as_2d(series, name):
    """Return the series as a ``(n_times, n_vars)`` float array."""
    data = np.asarray(series, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D of shape (n_times, n_vars)")
    return data


def _counts_at_eps(tree, Ea, Eb, eps, m0, m_full, n_center, n_pairs):
    """Per-order cross pair counts at one radius.

    Reference rows of ``Eb`` are scanned in time order; the scan stops
    early once at least ``n_center`` rows were scanned and the full
    order count reached ``n_pairs`` (Fortran ``ncmin``/``ipmin``).

    Returns ``(counts, scanned)`` where ``counts[k - 1]`` is the number
    of scanned pairs within ``eps`` in the first ``k`` coordinates.
    """
    counts = np.zeros(m_full, dtype=np.int64)
    scanned = 0
    n_ref = Eb.shape[0]
    for lo in range(0, n_ref, _CHUNK):
        queries = Eb[lo : lo + _CHUNK, :m0]
        for offset, cand in enumerate(tree.query_ball_point(queries, eps, p=np.inf)):
            if cand:
                idx = np.asarray(cand, dtype=np.intp)
                diffs = np.abs(Ea[idx] - Eb[lo + offset])
                within = np.maximum.accumulate(diffs, axis=1) < eps
                counts += within.sum(axis=0)
            scanned += 1
            if scanned >= n_center and counts[m_full - 1] >= n_pairs:
                return counts, scanned
    return counts, scanned


def cross_correlation_integral(
    a,
    b,
    embed=2,
    delay=1,
    eps=None,
    *,
    resolution=2.0,
    eps_min=None,
    eps_max=None,
    n_center=1000,
    n_pairs=1000,
):
    """Cross-correlation integral of two (possibly multivariate) series.

    Both series are embedded with the same embedding space (``embed`` delay blocks per component, total
    order ``n_vars * embed``).  For every radius the pair counts are
    normalized by ``scanned * n_base``, the number of scanned reference
    vectors times the number of base delay vectors.

    Parameters
    ----------
    a, b : array_like
        Base and reference series.  A 1-D array is a single component;
        a 2-D array must have shape ``(n_times, n_vars)`` with one
        column per component.  Both series must have the same number of
        components and are used in their original scalings.
    embed : int
        Embedding dimension per component.
    delay : int
        Time delay.
    eps : array_like or None
        Explicit radii.  When given, ``resolution``, ``eps_min`` and
        ``eps_max`` are ignored and every radius is evaluated, even if
        some counts vanish.
    resolution : float
        Radii per octave of the automatic ladder;
        the ladder descends from ``min(eps_max, 1.001 * max range)``
        by the factor ``2 ** (1 / resolution)``.
    eps_min, eps_max : float or None
        Bounds of the automatic ladder in data units. ``eps_min`` defaults to ``1e-30``; the ladder
        additionally stops as soon as the smallest reported order has no
        pair left (Fortran behaviour).
    n_center : int
        Minimal number of reference points scanned before the scan may
        stop early.
    n_pairs : int
        Minimal full-order pair count that stops the scan early. Early stopping truncates the counts and is
        what makes large radii cheap in the original program.

    Returns
    -------
    dict
        ``"eps"`` : evaluated radii, descending, shape ``(n_eps,)``.
        ``"orders"`` : embedding orders of the rows,
        ``max(2, n_vars) .. n_vars * embed``.
        ``"c"`` : cross-correlation values, shape
        ``(len(orders), n_eps)``; zeros where no pair was found.
        ``"scanned"`` : number of reference vectors actually scanned
        per radius (smaller than the total when the scan stopped early).

    Raises
    ------
    ValueError
        For invalid parameters, mismatched components, a series too
        short for the embedding, degenerate radii or a constant series
        combined with the automatic ladder.
    """
    da = _as_2d(a, "a")
    db = _as_2d(b, "b")
    if da.shape[1] != db.shape[1]:
        raise ValueError(f"a has {da.shape[1]} component(s) but b has {db.shape[1]}")
    n_vars = da.shape[1]

    if embed < 1:
        raise ValueError(f"embed must be >= 1, got {embed}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if n_vars * embed < 2:
        raise ValueError("Increase embedding dimension")
    if n_center < 0:
        raise ValueError(f"n_center must be >= 0, got {n_center}")
    if n_pairs < 0:
        raise ValueError(f"n_pairs must be >= 0, got {n_pairs}")

    Ea = lag_block_delay_embed(da, embed, delay)
    Eb = lag_block_delay_embed(db, embed, delay)
    if Ea.shape[0] < 1 or Eb.shape[0] < 1:
        raise ValueError(f"series too short for embed={embed}, delay={delay}")

    m_full = n_vars * embed
    m0 = max(2, n_vars)
    orders = np.arange(m0, m_full + 1)
    tree = cKDTree(Ea[:, :m0])

    explicit = eps is not None
    if explicit:
        eps_arr = np.sort(np.asarray(eps, dtype=np.float64).ravel())[::-1]
        if eps_arr.size < 1:
            raise ValueError("eps must contain at least one radius")
        if (eps_arr <= 0.0).any():
            raise ValueError("all radii must be > 0")
        first = iter(eps_arr)
    else:
        if resolution <= 0.0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
        max_range = 1.001 * max(
            float(np.ptp(da, axis=0).max()), float(np.ptp(db, axis=0).max())
        )
        cap = np.inf if eps_max is None else abs(float(eps_max))
        top = min(cap, max_range)
        if not np.isfinite(top) or top <= 0.0:
            raise ValueError(
                "the automatic ladder needs a non-constant series; "
                "pass explicit radii instead"
            )
        floor = 1e-30 if eps_min is None else abs(float(eps_min))
        if floor <= 0.0:
            raise ValueError(f"eps_min must be > 0, got {floor}")
        factor = 2.0 ** (1.0 / resolution)

    eps_out = []
    count_rows = []
    scanned_out = []
    current = None
    while True:
        if explicit:
            try:
                current = next(first)
            except StopIteration:
                break
        elif current is None:
            current = top
        else:
            current = current / factor
            if current < floor:
                break
        counts, scanned = _counts_at_eps(
            tree, Ea, Eb, float(current), m0, m_full, n_center, n_pairs
        )
        if not explicit and counts[m0 - 1] == 0:
            # Fortran: stop entirely once even the smallest order is empty.
            break
        eps_out.append(float(current))
        count_rows.append(counts.copy())
        scanned_out.append(scanned)
        if explicit and len(eps_out) == eps_arr.size:
            break

    if eps_out:
        counts_mat = np.asarray(count_rows, dtype=np.float64)
        scanned_arr = np.asarray(scanned_out, dtype=np.float64)
        denom = scanned_arr * Ea.shape[0]
        # keep the reported orders only, rows = orders, columns = radii
        c = (counts_mat[:, m0 - 1 :] / denom[:, None]).T
    else:
        c = np.empty((orders.size, 0))

    return {
        "eps": np.asarray(eps_out, dtype=np.float64),
        "orders": orders,
        "c": c,
        "scanned": np.asarray(scanned_out, dtype=np.int64),
    }
