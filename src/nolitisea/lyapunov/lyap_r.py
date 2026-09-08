"""Maximal Lyapunov exponent via Rosenstein's algorithm."""

from functools import partial

import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.utils.parallel import parallel_map
from nolitisea.utils.rescale import rescale_data

__all__ = ["lyap_r"]

# Radius growth factor of the ladder (main loop: ``eps *= 1.1``).
_EPS_FAC = 1.1
# References processed per fallback block; bounds the memory of the
# (rows, n_elem) query result.
_FALLBACK_CHUNK = 256
# Pairs processed per evolution block; bounds the memory of the
# (pairs, steps, dim) difference tensor.
_PAIR_CHUNK = 65536


def _lyap_r_fallback_worker(block, E, tree, n_elem, theiler, eps_done):
    """Re-query the full neighbour list for one fallback block.

    Returns ``(hit_idx, hit_vals)``: the reference indices that gained
    a kept neighbour and their indices.  Each block writes to disjoint
    rows of ``first``, so the caller can assign in any order;
    ``parallel_map`` preserves block order regardless.
    """
    d, i = tree.query(E[block], k=n_elem)
    v = (np.abs(i - block[:, None]) > theiler) & (d > 0.0) \
        & (d <= eps_done[block, None])
    hit = v.any(axis=1)
    if hit.any():
        dm = np.where(v, d, np.inf).min(axis=1)
        cnd = np.where(v & (d == dm[:, None]), i, n_elem)
        return block[hit], cnd.min(axis=1)[hit]
    return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)


def _lyap_r_evolve_worker(pn, E, steps, first):
    """Evolve one pair chunk and return its raw contributions.

    Returns ``(chunk_count, chunk_logsum)`` of shape ``(max_steps + 1,)``
    each; the caller reduces them in pair order so the float
    accumulation grouping matches the plain serial loop.
    """
    pe = first[pn]
    diff = (E[pn[:, None] + steps[None, :], :]
            - E[pe[:, None] + steps[None, :], :])
    dx = (diff * diff).sum(axis=2)
    pos = dx > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        logdx = np.log(dx)
    logdx[~pos] = 0.0
    return pos.sum(axis=0), logdx.sum(axis=0)


def lyap_r(series, dim, delay, max_steps, theiler=0, eps=None,
           n_jobs=None, backend="thread"):
    """Estimate the maximal Lyapunov exponent using Rosenstein's method.

    Parameters
    ----------
    series : array_like
        Input scalar series (1-D).
    dim : int
        Embedding dimension; must be >= 2.
    delay : int
        Time delay.
    max_steps : int
        Maximum evolution time of every neighbour pair.
    theiler : int, default 0
        Minimum temporal separation between reference point and
        neighbour: pairs with index distance
        ``<= theiler`` are never used.
    eps : float or None
        Initial search radius in the units of the input data.  ``None`` (default) uses: the data
        interval divided by 1000.  The radius is grown by a
        factor of 1.1 until every reference point has an eligible
        neighbour; with exact duplicate values in the series the value
        of ``eps`` influences which neighbour is kept.
    n_jobs : int, optional (default = None)
        Workers for the (fallback block, pair chunk) task pool.
        ``None``/``1`` runs sequentially; ``-1`` uses all CPUs (see
        ``nolitisea.utils.parallel.parallel_map``).  Results are
        bitwise identical for every ``n_jobs``/``backend``.
    backend : {"thread", "process"}, optional (default = "thread")
        Executor backend.  ``"thread"`` suits the GIL-releasing
        NumPy/SciPy kernels (cKDTree queries, the difference tensor);
        ``"process"`` pays pickling costs.

    Returns
    -------
    dict
        ``"times"`` : evolution times ``0 .. max_steps``.
        ``"divergence"`` : Rosenstein statistic ``< log ||v|| >``,
        shape ``(max_steps + 1,)``; ``nan`` where no pair contributed.

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the requested
        ``dim``/``delay``/``max_steps``/``theiler``.
    RuntimeError
        If the series is constant (zero value range).

    Notes
    -----
    The fallback re-query and the pair-evolution loop are distributed
    over independent chunks: each task returns its raw per-chunk
    contributions, and the caller assigns (fallback) or accumulates
    (evolution) them strictly in pair order.  The float accumulation
    grouping therefore matches the plain serial loop, so results are
    bitwise identical for every ``n_jobs`` and ``backend``.

    References
    ----------
    .. [1] Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993).
           A practical method for calculating largest Lyapunov
           exponents from small data sets.  *Physica D*, 65(1-2),
           117-134.
    .. [2] Hegger, R., Kantz, H., & Schreiber, T. (1999). Practical
           implementation of nonlinear time series methods: The TISEAN
           package.  *Chaos*, 9(2), 413-435.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("series must be a 1-D array")
    length = x.size

    if dim < 2:
        raise ValueError(f"dim must be >= 2, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if theiler < 0:
        raise ValueError(f"theiler must be >= 0, got {theiler}")

    del_off = (dim - 1) * delay
    n_ref = length - del_off - max_steps - theiler
    if n_ref < 1:
        raise ValueError(
            f"series of length {length} is too short for dim={dim}, "
            f"delay={delay}, max_steps={max_steps}, theiler={theiler}"
        )

    x, _, interval = rescale_data(x)
    eps0 = 1.0e-3 if eps is None else abs(float(eps)) / interval

    E = lag_block_delay_embed(x, dim, delay)
    n_elem = E.shape[0] - max_steps  # neighbours need room to evolve
    tree = cKDTree(E[:n_elem])

    base = np.arange(n_ref)
    # The excluded window holds at most 2 * theiler + 1 points, so among
    # the 2 * theiler + 2 nearest every eligible neighbour is guaranteed
    # to appear unless exact duplicates crowd the list; the clamped and
    # crowded cases get a second, wider pass below.
    k = min(2 * theiler + 2, n_elem)
    dists, idxs = tree.query(E[:n_ref], k=k)
    dists = np.atleast_2d(dists)
    idxs = np.atleast_2d(idxs)
    elig = np.abs(idxs - base[:, None]) > theiler

    # --- processing radius of every reference point ---------
    # A point is processed at the first ladder level eps0 * 1.1**lev
    # that contains any eligible neighbour, identical values included.
    d_any = np.full(n_ref, np.inf)
    rows = np.flatnonzero(elig.any(axis=1))
    d_any[rows] = dists[rows, elig[rows].argmax(axis=1)]
    lev = np.zeros(n_ref)
    fin = np.isfinite(d_any) & (d_any > 0.0)
    lev[fin] = np.ceil(np.log(d_any[fin] / eps0) / np.log(_EPS_FAC))
    eps_done = np.where(np.isfinite(d_any),
                        eps0 * _EPS_FAC ** np.maximum(lev, 0.0), 0.0)

    # --- kept neighbour: nearest non-identical within the radius -------
    # Among candidates sharing the minimal distance the one with the
    # smallest index wins (deterministic, independent of the search
    # internals).
    valid = elig & (dists > 0.0) & (dists <= eps_done[:, None])
    ok = valid.any(axis=1)
    dmin = np.where(valid, dists, np.inf).min(axis=1)
    cand = np.where(valid & (dists == dmin[:, None]), idxs, n_elem)
    first = np.where(ok, cand.min(axis=1), -1).astype(np.intp)
    # The truncated k-nearest list may hide candidates: rows without a
    # kept neighbour cannot prove absence unless every point within the
    # radius is known to be listed (largest listed distance below the
    # radius), and rows whose minimal-distance tie group reaches the
    # list end may miss equidistant candidates beyond it.  Both retry
    # with the complete neighbour list.
    absent = ~ok & (eps_done > 0.0) & (dists[:, -1] <= eps_done)
    trunc = ok & (dists[:, -1] == dmin)
    retry = np.flatnonzero((absent | trunc) & (k < n_elem))
    blocks = [retry[lo:lo + _FALLBACK_CHUNK]
              for lo in range(0, retry.size, _FALLBACK_CHUNK)]
    fb_worker = partial(
        _lyap_r_fallback_worker,
        E=E,
        tree=tree,
        n_elem=n_elem,
        theiler=theiler,
        eps_done=eps_done,
    )
    for hit_idx, hit_vals in parallel_map(
            fb_worker, blocks, n_jobs=n_jobs, backend=backend):
        first[hit_idx] = hit_vals

    # --- evolve every kept pair and average the log separations --------
    steps = np.arange(max_steps + 1)
    count = np.zeros(max_steps + 1, dtype=np.int64)
    logsum = np.zeros(max_steps + 1)
    pairs = np.flatnonzero(first >= 0)
    pair_chunks = [pairs[lo:lo + _PAIR_CHUNK]
                   for lo in range(0, pairs.size, _PAIR_CHUNK)]
    ev_worker = partial(
        _lyap_r_evolve_worker,
        E=E,
        steps=steps,
        first=first,
    )
    for chunk_count, chunk_logsum in parallel_map(
            ev_worker, pair_chunks, n_jobs=n_jobs, backend=backend):
        count += chunk_count
        logsum += chunk_logsum

    divergence = np.full(max_steps + 1, np.nan)
    hit = count > 0
    divergence[hit] = logsum[hit] / count[hit] / 2.0

    return {"times": steps, "divergence": divergence}
