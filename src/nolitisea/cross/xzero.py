"""Locally zeroth-order cross-prediction (TISEAN ``xzero``).

Python rewrite of ``xzero.c`` (Hegger) using a KD-tree neighbour
search.  The program estimates how well one scalar series can be
forecast from another: delay vectors of the second series ``b`` are
located among the delay vectors of the first series ``a`` and the
neighbours' successors in ``a`` are averaged as the zeroth-order
forecast of ``b``.  The relative rms cross forecast error is reported
for forecast horizons ``1 .. n_steps``.

Both series are rescaled to ``[0, 1]`` first (TISEAN ``rescale_data``)
and the errors are normalized by the standard deviation of the
rescaled second series (TISEAN ``variance``), so the reported error is
``1`` when the forecast is no better than a random guess and ``0`` for
a perfect forecast.

Conventions of this rewrite
---------------------------
* Delay vectors are formed backwards: the vector ending at time ``i``
  is ``(x[i-(dim-1)*delay], ..., x[i-delay], x[i])``, matching the
  C ``find_neighbors`` which reads the query components
  ``coord[-k*delay]`` and the data components ``s[element-k*delay]``.
* A neighbour is kept when its maximum-norm distance is smaller than
  or equal to ``eps`` (the C check ``dx > eps -> skip``); the KD-tree
  ball query returns exactly this set.
* A reference point is processed once at least ``n_neighbors``
  neighbours are found; points with too few neighbours are retried in
  the next pass with ``eps`` grown by ``eps_factor``.  The C program
  loops forever when even the full data range cannot supply enough
  neighbours; this rewrite raises ``RuntimeError`` once ``eps``
  reaches ``1`` (the largest possible distance of data in ``[0, 1]``).
* With ``-r`` the C program interprets the given radius in the units
  of the original data and divides it by the average of the two data
  ranges; the default radius ``1e-3`` is already in rescaled units
  (``(data interval) / 1000`` in original units).
"""
import numpy as np
from scipy.spatial import cKDTree

from nolitisea.utils.rescale import rescale_data

__all__ = ["cross_zeroth"]


def cross_zeroth(a, b, dim=3, delay=1, eps=None, *, n_neighbors=30,
                 eps_factor=1.2, n_steps=1, n_refs=None):
    """Average cross forecast error of a zeroth-order local model.

    Delay vectors of ``b`` (the predicted series) are searched among 
    delay vectors of ``a`` (the predictor series); the mean of the 
    neighbours' successors in ``a`` forecasts the future of ``b``.

    Parameters
    ----------
    a, b : array_like
        The two scalar series; they must have the same length.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    eps : float or None
        Neighbourhood diameter to start with in the
        units of the original data.  ``None`` uses ``1e-3`` in rescaled
        units, i.e. ``(data interval) / 1000`` in original units.
    n_neighbors : int
        Minimal number of neighbours required before a reference point
        is processed.
    eps_factor : float
        Factor by which ``eps`` is grown between passes; must be
        greater than 1.
    n_steps : int
        Largest forecast horizon; the error is reported for every
        horizon ``1 .. n_steps``.
    n_refs : int or None
        Number of reference points to use (C option ``-n``).  ``None``
        uses the whole series.  The reference times are
        ``(dim-1)*delay .. n_refs - n_steps - 1``.

    Returns
    -------
    dict
        ``"steps"`` : horizons ``1 .. n_steps``.
        ``"error"`` : relative rms cross forecast error of each
        horizon, normalized by the standard deviation of the rescaled
        second series.

    Raises
    ------
    ValueError
        For invalid parameters, a length mismatch, a series too short
        for the requested embedding, or a constant series.
    RuntimeError
        When even the full data range cannot provide ``n_neighbors``
        neighbours (the C program would loop forever in this case).
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = x.size
    if n == 0 or y.size == 0:
        raise ValueError("input series must not be empty")
    if n != y.size:
        raise ValueError(
            f"length mismatch: a has {n} points, b has {y.size}"
        )
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
    if eps_factor <= 1.0:
        raise ValueError(f"eps_factor must be > 1, got {eps_factor}")
    if eps is not None and (not np.isfinite(eps) or eps <= 0.0):
        raise ValueError(f"eps must be a positive finite number, got {eps}")
    if n_refs is not None and n_refs <= n_steps:
        raise ValueError(
            f"n_refs must be > n_steps, got n_refs={n_refs}, n_steps={n_steps}"
        )

    x, _, range_a = rescale_data(x)
    y, _, range_b = rescale_data(y)
    interval = (range_a + range_b) / 2.0

    rms2 = float(np.std(y))
    if rms2 == 0.0:
        raise ValueError("standard deviation of the data is zero")

    eps0 = 1e-3 if eps is None else float(eps) / interval

    clength = (n_refs if n_refs is not None and n_refs <= n else n) - n_steps
    emb_off = (dim - 1) * delay
    n_refs_used = clength - emb_off
    if n_refs_used < 1:
        raise ValueError(
            f"series too short for dim={dim}, delay={delay}, "
            f"n_steps={n_steps}: only {n_refs_used} reference point(s)"
        )
    n_cand = n - n_steps - emb_off
    if n_cand < 1:
        raise ValueError(
            f"series too short for dim={dim}, delay={delay}, "
            f"n_steps={n_steps}: no neighbour candidate left"
        )

    # Delay vectors written in forward component order; row k of ``Ea``
    # ends at time ``emb_off + k`` (C ``make_box`` over
    # ``[emb_off, LENGTH - STEP)``) and row ``t`` of ``Q`` ends at the
    # reference time ``emb_off + t``.  The maximum norm is invariant
    # under the component order, so forward and backward layouts agree.
    offs = np.arange(dim) * delay
    Ea = x[np.arange(n_cand)[:, None] + offs[None, :]]
    Q = y[np.arange(n_refs_used)[:, None] + offs[None, :]]
    end = np.arange(n_cand) + emb_off

    tree = cKDTree(Ea)
    done = np.zeros(n_refs_used, dtype=bool)
    error = np.zeros(n_steps, dtype=np.float64)
    epsilon = eps0 / eps_factor
    while not done.all():
        epsilon *= eps_factor
        for t in np.flatnonzero(~done):
            found = tree.query_ball_point(Q[t], epsilon, p=np.inf)
            if len(found) < n_neighbors:
                continue
            idx = np.sort(np.asarray(found, dtype=np.intp))
            # rows: neighbours, columns: forecast horizons 1..n_steps
            successors = x[end[idx, None] + np.arange(1, n_steps + 1)]
            forecasts = successors.mean(axis=0)
            error += (forecasts - y[t + emb_off + np.arange(1, n_steps + 1)]) ** 2
            done[t] = True
        if not done.all() and epsilon >= 1.0:
            raise RuntimeError(
                f"could not find {n_neighbors} neighbours within the "
                f"data range for {int((~done).sum())} reference point(s)"
            )

    error = np.sqrt(error / n_refs_used) / rms2
    return {"steps": np.arange(1, n_steps + 1), "error": error}
