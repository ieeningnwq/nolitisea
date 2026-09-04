"""Finite Size Lyapunov Exponent (FSLE) via Vulpiani's method."""

from functools import partial

import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.utils.parallel import parallel_map
from nolitisea.utils.rescale import rescale_data

__all__ = ["fsle"]

# Radius growth factor of the ladder.
_EPS_FAC = np.sqrt(2.0)
# Pairs processed per evolution block; bounds the memory of the
# (pairs, howmany) contribution tensors returned by each worker.
_PAIR_CHUNK = 4096


def _fsle_evolve_worker(pn, x, first, mindx_arr, eps0, eps_levels,
                        advance, length):
    """Evolve one pair chunk and return per-level contributions.

    Returns ``(chunk_time, chunk_factor, chunk_count)`` of shape
    ``(n_pairs, howmany)`` each; the caller accumulates them strictly
    in pair order so the float accumulation grouping matches the plain
    serial loop.
    """
    pe = first[pn]
    mx0 = mindx_arr[pn]
    n_pairs = pn.size
    howmany = eps_levels.size
    log_fac = np.log(_EPS_FAC)

    c_time = np.zeros((n_pairs, howmany))
    c_factor = np.zeros((n_pairs, howmany))
    c_count = np.zeros((n_pairs, howmany), dtype=np.int64)

    for j in range(n_pairs):
        act = int(pn[j]) + advance
        nbr = int(pe[j]) + advance
        mindx = mx0[j]

        # Skip pairs that cannot evolve (out of bounds or zero distance).
        if act >= length or nbr >= length or mindx <= 0.0:
            continue

        # Level index of the initial distance (C: ``which``).
        which = int(np.log(mindx / eps0) / log_fac)

        # If the initial distance is below the first level, advance
        # until the 1D separation reaches ``eps_levels[0]`` (C: which < 0
        # branch).
        done = False
        if which < 0:
            while True:
                dx = abs(x[act] - x[nbr])
                if dx >= eps_levels[0]:
                    break
                act += 1
                nbr += 1
                if act >= length or nbr >= length:
                    done = True
                    break
            if done:
                continue
            mindx = dx
            if mindx <= 0.0:
                continue
            which = int(np.log(mindx / eps0) / log_fac)

        # Evolve through successive radius levels (C: for loop).
        for i in range(max(which, 0), howmany - 1):
            stime = 0
            while True:
                dx = abs(x[act] - x[nbr])
                if dx >= eps_levels[i + 1]:
                    break
                act += 1
                nbr += 1
                if act >= length or nbr >= length:
                    done = True
                    break
                stime += 1
            if done:
                break
            if stime > 0:
                c_time[j, i] += stime
                c_factor[j, i] += np.log(dx / mindx)
                c_count[j, i] += 1
            mindx = dx

    return c_time, c_factor, c_count


def fsle(series, dim=2, delay=1, mindist=0, eps0=None,
         n_jobs=None, backend="thread"):
    """Estimate the Finite Size Lyapunov Exponent (FSLE).

    Parameters
    ----------
    series : array_like
        Input scalar series (1-D).
    dim : int, default 2
        Embedding dimension; must be >= 2.
    delay : int, default 1
        Time delay between embedding coordinates.
    mindist : int, default 0
        Theiler window: neighbours within ``mindist`` samples of the
        reference point are excluded.
    eps0 : float or None
        Initial scale in the units of the input data.  ``None``
        (default) uses ``0.001 * variance`` of the rescaled series.
    n_jobs : int, optional (default = None)
        Workers for the pair-evolution task pool.
        ``None``/``1`` runs sequentially; ``-1`` uses all CPUs (see
        ``nolitisea.utils.parallel.parallel_map``).  Results are
        bitwise identical for every ``n_jobs``/``backend``.
    backend : {"thread", "process"}, optional (default = "thread")
        Executor backend.

    Returns
    -------
    dict
        ``"eps"`` : radii in original data units, ascending,
        shape ``(n_levels,)``.
        ``"fsle"`` : FSLE values, shape ``(n_levels,)``; ``nan``
        where no pair contributed.
        ``"count"`` : number of contributing pairs per level.
        ``"time"`` : total growth time per level.

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the
        requested ``dim``/``delay``/``mindist``.
    RuntimeError
        If the series is constant (zero value range).

    Notes
    -----
    The neighbour search uses the Chebyshev metric on the full
    delay-coordinate embedding. On continuous data this produces
    the same neighbour; on quantised data ties are broken by
    smallest index (deterministic).  The FSLE is a statistical
    average, so the selection metric has negligible impact on the
    estimate.

    The radius ladder upper bound is the variance of the rescaled
    series in both the default and user-``eps0`` cases, consistent
    in [0, 1] units.

    References
    ----------
    .. [1] Vulpiani, A. et al. (1998).  Finite size Lyapunov
           exponent.  *Physica D*, 119(1-2), 204-212.
    .. [2] Hegger, R., Kantz, H., & Schreiber, T. (1999).  Practical
           implementation of nonlinear time series methods: The
           TISEAN package.  *Chaos*, 9(2), 413-435.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("series must be a 1-D array")
    length = x.size

    if dim < 2:
        raise ValueError(f"dim must be >= 2, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if mindist < 0:
        raise ValueError(f"mindist must be >= 0, got {mindist}")

    del_off = delay * (dim - 1)
    maxlength = length - del_off - 1 - mindist
    if maxlength < 1:
        raise ValueError(
            f"series of length {length} is too short for dim={dim}, "
            f"delay={delay}, mindist={mindist}"
        )

    # Rescale to [0, 1]; radii are handled in rescaled units internally
    # and reported in the original data units.
    x, _, interval = rescale_data(x)
    se_var = np.var(x)

    if eps0 is not None:
        eps0_r = abs(float(eps0)) / interval
    else:
        eps0_r = 1e-3 * se_var
    epsmax = se_var

    if eps0_r >= epsmax:
        raise ValueError(
            f"eps0 ({eps0_r * interval:.4e}) is too large; "
            f"must be below the variance ({epsmax * interval:.4e})"
        )

    # Radius ladder: data[i].eps = eps0 * epsfactor^i.
    howmany = int(np.log(epsmax / eps0_r) / np.log(_EPS_FAC)) + 1
    eps_levels = eps0_r * _EPS_FAC ** np.arange(howmany)

    # --- neighbour search via cKDTree (Chebyshev metric) ---------------
    E = lag_block_delay_embed(x, dim, delay)
    n_ref = maxlength + 1
    tree = cKDTree(E[:n_ref])

    # Query enough neighbours to skip the Theiler window.
    k = min(2 * mindist + 2, n_ref)
    dists, idxs = tree.query(E[:n_ref], k=k, p=np.inf)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    base = np.arange(n_ref)
    elig = np.abs(idxs - base[:, None]) > mindist

    # Nearest eligible neighbour with dist > 0 (Chebyshev).  Among
    # min-distance ties the smallest index wins (deterministic).
    elig_pos = elig & (dists > 0.0)
    d_elig = np.where(elig_pos, dists, np.inf)
    dmin = d_elig.min(axis=1)
    has_pair = (dmin < np.inf) & (dmin < epsmax)
    cand = np.where(elig_pos & (dists == dmin[:, None]), idxs, n_ref)
    first = np.where(has_pair, cand.min(axis=1), -1).astype(np.intp)
    mindx_arr = np.where(has_pair, dmin, 0.0)

    # The truncated k-nearest list may hide candidates: rows whose
    # minimal-distance tie group reaches the list end may miss
    # equidistant candidates beyond it.  Retry those rows with the
    # complete neighbour list from query_ball_point.
    trunc = has_pair & (dists[:, -1] == dmin)
    for n in np.flatnonzero(trunc & (k < n_ref)):
        nb = tree.query_ball_point(E[n], dmin[n], p=np.inf)
        nb = np.asarray(nb, dtype=np.intp)
        nb = nb[(np.abs(nb - n) > mindist) & (nb < n)]
        if nb.size:
            first[n] = int(nb.min())
            # mindx unchanged: all candidates in the ball share dmin[n].

    # --- evolve every pair through the radius levels -------------------
    pairs = np.flatnonzero(first >= 0)
    advance = (dim - 1) * delay + 1
    total_time = np.zeros(howmany)
    total_factor = np.zeros(howmany)
    total_count = np.zeros(howmany, dtype=np.int64)

    pair_chunks = [pairs[lo:lo + _PAIR_CHUNK]
                   for lo in range(0, pairs.size, _PAIR_CHUNK)]
    ev_worker = partial(
        _fsle_evolve_worker,
        x=x,
        first=first,
        mindx_arr=mindx_arr,
        eps0=eps0_r,
        eps_levels=eps_levels,
        advance=advance,
        length=length,
    )
    for c_t, c_f, c_c in parallel_map(
            ev_worker, pair_chunks, n_jobs=n_jobs, backend=backend):
        total_time += c_t.sum(axis=0)
        total_factor += c_f.sum(axis=0)
        total_count += c_c.sum(axis=0)

    fsle_vals = np.full(howmany, np.nan)
    hit = total_count > 0
    fsle_vals[hit] = total_factor[hit] / total_time[hit]

    return {
        "eps": eps_levels * interval,
        "fsle": fsle_vals,
        "count": total_count,
        "time": total_time,
    }
