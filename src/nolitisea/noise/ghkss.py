"""Multivariate noise reduction using the GHKSS algorithm ."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.utils.rescale import rescale_data

__all__ = ["ghkss"]

# Batch size for KD-tree ball queries; bounds the memory used by neighbour lists.
_QUERY_CHUNK = 4096
# Off-manifold coordinate weight of the partial metric.
_METRIC_HEAVY = 1.0e3
# Safety bound for the adaptive-epsilon loop: after rescaling, every component
# lies in [0, 1], so the Chebyshev diameter of the embedding space is <= 1 and
# eps > 2 must contain every point.
_EPS_MAX = 2.0


def _local_correction(E, row, nb, metric, qdim):
    """Local projection correction for one point.

    Parameters
    ----------
    E : ndarray
        Embedded phase-space matrix, one row per phase-space point.
    row : int
        Phase-space index of the point being corrected.
    nb : array
        Phase-space indices of its neighbours at the current epsilon
        (the point itself included).
    metric : ndarray
        Coordinate weights (1.0 or :data:`_METRIC_HEAVY`).
    qdim : int
        Number of dominant directions kept by the local projection.

    Returns
    -------
    ndarray
        Correction vector of length ``dim``.
    """
    X = E[nb]
    nf = len(nb)
    av = X.mean(axis=0)
    # Weighted covariance: cov[i, j] = metric[i] * metric[j] * Cov(x_i, x_j).
    Xw = (X - av) * metric
    cov = (Xw.T @ Xw) / nf
    w, V = np.linalg.eigh(cov)             # ascending eigenvalues
    order = np.argsort(-w, kind="stable")  # descending, stable
    Vs = V[:, order[qdim:]]                # dim - qdim least-variance directions
    y = metric * (E[row] - av)
    return (Vs @ (Vs.T @ y)) / metric


def _trend_update(delta, corr, row, nb, metric, trace, comp, delay, embed):
    """Accumulate the trend correction of one point."""
    av = corr[nb].mean(axis=0)
    d = (corr[row] - av) / (trace * metric)
    n_time = row + (embed - 1) * delay
    for k in range(embed):
        delta[n_time - k * delay, :] += d[k * comp:(k + 1) * comp]


def ghkss(
    series: np.ndarray,
    embed: int = 5,
    delay: int = 1,
    qdim: int = 2,
    minn: int = 50,
    mineps: float | None = None,
    epsfac: float = np.sqrt(2.0),
    iterations: int = 1,
    euclidean: bool = False,
) -> dict:
    """Multivariate noise reduction with the GHKSS local projection algorithm.

    Each phase-space point is locally projected onto the ``qdim`` dominant
    eigendirections of the metric-weighted covariance of its neighbourhood;
    the component in the orthogonal complement is interpreted as noise.  The
    correction of every point is smoothed against its neighbours (trend
    subtraction) and applied to the original scalar components.

    References
    ----------
    .. [1] Grassberger, P., Hegger, R., Kantz, H., Schaffrath, C., & Schreiber,
        T. (1993). On noise reduction methods for chaotic data.  *Chaos*,
        3(2), 127-141.
    .. [2] Hegger, R., Kantz, H., & Schreiber, T. (1999). Practical
        implementation of nonlinear time series methods: The TISEAN package.
        *Chaos*, 9(2), 413-435.

    Parameters
    ----------
    series : np.ndarray
        Input data.  A 1-D array is treated as a single component; a 2-D
        array must have shape ``(n_times, comp)`` with one column per
        component.
    embed : int
        Embedding dimension per component; the full phase-space dimension is
        ``dim = comp * embed``.
    delay : int
        Time delay (in samples) between consecutive embedding blocks.
    qdim : int
        Dimension of the local projection space; must satisfy
        ``1 <= qdim < comp * embed``.
    minn : int
        Minimum number of neighbours (the point itself included) required
        before a point is corrected.
    mineps : float, optional
        Minimal neighbourhood size.  ``None`` (default) uses the
        default ``interval / 1000`` in rescaled units.  A given value is
        interpreted as an absolute radius in the units of the input data
        and divided by the largest component interval.
    epsfac : float
        Growth factor applied to the neighbourhood size until every point
        has at least ``minn`` neighbours (Default ``sqrt(2)``).
    iterations : int
        Number of correction iterations; the series is re-embedded after
        each iteration.
    euclidean : bool
        If ``True`` weight all phase-space coordinates equally; otherwise
        use the partial metric, which down-weights the oldest and
        newest ``comp`` coordinates by ``1e3``.

    Returns
    -------
    dict
        ``"corrected"`` : ndarray with the same shape as the input series,
        in the original data units.
        ``"stats"`` : list with one entry per iteration, each containing the
        per-component ``"average_shift"`` and ``"rms_correction"`` of the
        applied correction in original data units.

    Raises
    ------
    ValueError
        For invalid parameter combinations or a series too short for the
        requested embedding/neighbour count.
    RuntimeError
        If a component has zero range (constant series).
    """
    original_ndim = np.asarray(series).ndim
    data = np.asarray(series, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(
            "series must be a 1-D array or a 2-D array of shape "
            "(n_times, comp)"
        )
    length, comp = data.shape
    dim = comp * embed
    emb_offset = (embed - 1) * delay
    n_points = length - emb_offset

    if delay < 1:
        raise ValueError("delay must be >= 1")
    if embed < 1:
        raise ValueError("embed must be >= 1")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if minn < 1:
        raise ValueError("minn must be >= 1")
    if epsfac <= 1.0:
        raise ValueError("epsfac must be > 1")
    if n_points <= 0:
        raise ValueError(
            f"series of length {length} is too short for embed={embed}, "
            f"delay={delay}"
        )
    if n_points < minn:
        raise ValueError(
            f"only {n_points} phase-space points available, cannot find "
            f"minn={minn} neighbours"
        )
    if not (1 <= qdim < dim):
        raise ValueError(f"qdim must satisfy 1 <= qdim < dim={dim}")

    # Rescale every component to [0, 1].
    s = np.empty((length, comp), dtype=np.float64)
    mins = np.empty(comp, dtype=np.float64)
    intervals = np.empty(comp, dtype=np.float64)
    for c in range(comp):
        s[:, c], mins[c], intervals[c] = rescale_data(data[:, c])

    if mineps is None:
        eps_min = 1.0 / 1000.0
    else:
        eps_min = float(mineps) / intervals.max()

    metric = np.ones(dim, dtype=np.float64)
    if not euclidean:
        # Partial metric: the first and last ``comp`` coordinates
        # (newest and oldest delay block) are down-weighted.
        metric[:comp] = _METRIC_HEAVY
        metric[dim - comp:] = _METRIC_HEAVY
    trace = float((1.0 / metric).sum())

    stats = []
    for _ in range(iterations):
        E = lag_block_delay_embed(s, embed, delay)
        tree = cKDTree(E)
        corr = np.zeros((n_points, dim), dtype=np.float64)
        ok = np.zeros(n_points, dtype=np.int64)
        delta = np.zeros((length, comp), dtype=np.float64)

        # --- correction pass with adaptive epsilon ---
        eps = eps_min
        level = 1
        resized = False
        while True:
            remaining = np.flatnonzero(ok == 0)
            if remaining.size == 0:
                break
            if eps > _EPS_MAX:
                raise RuntimeError(
                    "ghkss: adaptive epsilon exceeded the rescaled data "
                    "range before every point found minn neighbours"
                )
            pts = E if remaining.size == n_points else E[remaining]
            counts = tree.query_ball_point(
                pts, eps, p=np.inf, return_length=True
            )
            good = remaining[counts >= minn]
            if good.size:
                if level == 1:
                    resized = True
                for lo in range(0, good.size, _QUERY_CHUNK):
                    rows = good[lo:lo + _QUERY_CHUNK]
                    nb_lists = tree.query_ball_point(
                        E[rows], eps, p=np.inf, return_sorted=True
                    )
                    for row, nb in zip(rows, nb_lists):
                        corr[row] = _local_correction(
                            E, row, np.asarray(nb, dtype=np.intp),
                            metric, qdim,
                        )
                        ok[row] = level
            eps *= epsfac
            level += 1

        # --- trend pass, re-finding neighbours per level ---
        eps = eps_min
        for lev in range(1, level):
            idx = np.flatnonzero(ok == lev)
            for lo in range(0, idx.size, _QUERY_CHUNK):
                rows = idx[lo:lo + _QUERY_CHUNK]
                nb_lists = tree.query_ball_point(
                    E[rows], eps, p=np.inf, return_sorted=True
                )
                for row, nb in zip(rows, nb_lists):
                    _trend_update(
                        delta, corr, row, np.asarray(nb, dtype=np.intp),
                        metric, trace, comp, delay, embed,
                    )
            eps *= epsfac

        # --- apply the correction ---
        hav = delta.mean(axis=0)
        hsigma = np.sqrt(np.abs((delta * delta).mean(axis=0) - hav * hav))
        stats.append({
            "average_shift": (hav * intervals).tolist(),
            "rms_correction": (hsigma * intervals).tolist(),
        })
        s = s - delta
        if resized:
            eps_min /= epsfac

    corrected = s * intervals[None, :] + mins[None, :]
    if original_ndim == 1:
        corrected = corrected[:, 0]
    return {"corrected": corrected, "stats": stats}
