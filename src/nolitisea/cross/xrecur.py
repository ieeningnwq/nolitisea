"""Cross-recurrence plot of two data sets."""

import random

import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import lag_block_delay_embed

__all__ = ["cross_recurrence"]

# Radius growth factor and pass limit of the fixed-``kmin`` path
# (Fortran ``epsfac`` and the ``do 10 io=1,100`` loop).
_EPS_FAC = 1.1
_MAX_PASSES = 100
# Fixed seed for the ``percentage`` random thinning, so results are
# reproducible across runs.
_RNG_SEED = 13413241


def _as_2d(series, name):
    """Return the series as a ``(n_times, n_vars)`` float array."""
    data = np.asarray(series, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D of shape (n_times, n_vars)")
    return data


def _rescale_components(data):
    """Rescale every component to ``[0, 0.9999]``.

    Transformation ``scal = .9999/(xmax-xmin)``,
    ``x = (x - xmin) * scal`` exactly.
    """
    out = np.array(data, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        xmin = float(out[:, j].min())
        xmax = float(out[:, j].max())
        if xmax - xmin <= 0.0:
            raise ValueError(
                f"component {j} is constant and cannot be normalized; "
                "pass normalize=False to use the raw scalings"
            )
        scal = 0.9999 / (xmax - xmin)
        out[:, j] = (out[:, j] - xmin) * scal
    return out


def _full_distances(Eb, idx, point, p):
    """Distances between selected base rows and one reference point."""
    if p == np.inf:
        return np.abs(Eb[idx] - point).max(axis=1)
    return np.sqrt(((Eb[idx] - point) ** 2).sum(axis=1))


def cross_recurrence(a, b, dim=2, delay=1, eps=1e-3, metric="chebyshev", *,
                     normalize=True, kmin=0, percentage=100.0,
                     step_a=1, step_b=1):
    """Cross-recurrence pairs of two (possibly multivariate) series.

    Delay vectors of ``a`` (rows) are matched against delay vectors of ``b`` (columns).

    Parameters
    ----------
    a, b : array_like
        The two series.  A 1-D array is a single component; a 2-D array
        must have shape ``(n_times, n_vars)`` with one column per
        component.  Both series must have the same number of components.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    eps : float
        Neighbourhood diameter.  In the ``kmin`` path
        this is the starting diameter, grown by a factor of 1.1 per
        pass (at most 100 passes).
    metric : {"chebyshev", "euclidean"}, default "chebyshev"
        Distance metric; default maximum norm.
    normalize : bool, default True
        Rescale every component of both series to ``[0, 0.9999]``
        before the computation (C switches this off with ``-n``).
    kmin : int, default 0
        When positive, find for every reference point the first radius
        level at which at least ``kmin`` eligible neighbours exist and
        keep all of them (C option ``-k``); ``eps`` only sets the
        starting radius.
    percentage : float
        Percentage of eligible pairs to keep (C option ``-%``); only
        used when ``kmin == 0``.  100 keeps everything.
    step_a, step_b : int
        Use only every ``step_a``-th delay vector of ``a`` and every
        ``step_b``-th of ``b`` .

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n_pairs, 2)``; column 0 holds row
        (``a``) indices, column 1 column (``b``) indices, sorted
        lexicographically.  Empty with shape ``(0, 2)`` if nothing
        recurs.

    Raises
    ------
    ValueError
        For invalid parameters, mismatched components, a series too
        short for the embedding, or a constant component combined with
        ``normalize=True``.
    """
    da = _as_2d(a, "a")
    db = _as_2d(b, "b")
    if da.shape[1] != db.shape[1]:
        raise ValueError(
            f"a has {da.shape[1]} component(s) but b has {db.shape[1]}"
        )
    n_vars = da.shape[1]

    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if step_a < 1 or step_b < 1:
        raise ValueError("step_a and step_b must be >= 1")
    if kmin < 0:
        raise ValueError(f"kmin must be >= 0, got {kmin}")
    if not 0.0 < float(percentage) <= 100.0:
        raise ValueError(f"percentage must be in (0, 100], got {percentage}")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be a positive finite number, got {eps}")
    if metric == "chebyshev":
        p = np.inf
    elif metric == "euclidean":
        p = 2
    else:
        raise ValueError('metric must be "chebyshev" or "euclidean"')

    if normalize:
        da = _rescale_components(da)
        db = _rescale_components(db)

    Ea = lag_block_delay_embed(da, dim, delay)
    Eb = lag_block_delay_embed(db, dim, delay)
    if Ea.shape[0] < 1 or Eb.shape[0] < 1:
        raise ValueError(f"series too short for dim={dim}, delay={delay}")

    # Box search prefix: two delay coordinates for scalar input, the
    # spatial components for multivariate input (Fortran ``mb``/``mbase``).
    m0 = min(dim, 2) if n_vars == 1 else n_vars
    tree = cKDTree(Eb[:, :m0])
    col_mask = np.zeros(Eb.shape[0], dtype=bool)
    col_mask[::step_b] = True

    pairs = []
    if kmin == 0:
        rng = random.Random(_RNG_SEED) if percentage < 100.0 else None
        for i in range(0, Ea.shape[0], step_a):
            cand = tree.query_ball_point(Ea[i, :m0], eps, p=p)
            if not cand:
                continue
            idx = np.sort(np.asarray(cand, dtype=np.intp))
            idx = idx[col_mask[idx]]
            if idx.size == 0:
                continue
            dist = _full_distances(Eb, idx, Ea[i], p)
            idx = idx[dist <= eps]
            if idx.size == 0:
                continue
            if rng is not None:
                keep = np.fromiter(
                    (rng.random() <= percentage / 100.0 for _ in idx),
                    dtype=bool, count=idx.size,
                )
                idx = idx[keep]
            if idx.size:
                rows = np.full((idx.size, 2), i, dtype=np.int64)
                rows[:, 1] = idx
                pairs.append(rows)
    else:
        done = np.zeros(Ea.shape[0], dtype=bool)
        eps_cur = float(eps) / _EPS_FAC
        for _ in range(_MAX_PASSES):
            eps_cur *= _EPS_FAC
            ntodo = 0
            for i in range(0, Ea.shape[0], step_a):
                if done[i]:
                    continue
                ntodo += 1
                cand = tree.query_ball_point(Ea[i, :m0], eps_cur, p=p)
                if not cand:
                    continue
                idx = np.sort(np.asarray(cand, dtype=np.intp))
                idx = idx[col_mask[idx]]
                if idx.size == 0:
                    continue
                dist = _full_distances(Eb, idx, Ea[i], p)
                idx = idx[dist <= eps_cur]
                if idx.size < kmin:
                    continue
                done[i] = True
                ntodo -= 1
                rows = np.full((idx.size, 2), i, dtype=np.int64)
                rows[:, 1] = idx
                pairs.append(rows)
            if ntodo == 0:
                break

    if pairs:
        out = np.vstack(pairs)
        # The kmin path completes references pass by pass; restore the
        # documented lexicographic order.
        return out[np.lexsort((out[:, 1], out[:, 0]))]
    return np.empty((0, 2), dtype=np.int64)