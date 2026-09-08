"""Maximal Lyapunov exponent via the Kantz algorithm."""

from functools import partial

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from nolitisea.utils.parallel import parallel_map
from nolitisea.utils.rescale import rescale_data

__all__ = ["lyap_k"]


# Reference points per parallel task.  The value is fixed (it does not
# depend on ``n_jobs``) so that serial and parallel runs issue exactly
# the same neighbour queries; combined with the order-preserving
# reduction in ``lyap_k`` this keeps results bitwise identical for
# every ``n_jobs``/``backend``.
_CHUNK = 256


def _lyap_k_chunk_worker(task, x, B, tree, steps, delays, theiler,
                         min_dim, n_dims, max_steps):
    """Raw Kantz contributions for one (radius, reference-point chunk).

    Returns per-point contribution arrays of shape
    ``(n_points, n_dims, max_steps + 1)``; the caller adds them in
    reference-point order.
    """
    eps_now, act_start, act_stop = task
    eps2 = eps_now * eps_now
    n_loc = act_stop - act_start
    contrib_lyap = np.zeros((n_loc, n_dims, max_steps + 1))
    contrib_count = np.zeros((n_loc, n_dims, max_steps + 1), dtype=np.int64)

    nb_lists = tree.query_ball_point(B[act_start:act_stop], eps_now)
    for i, nb in enumerate(nb_lists):
        if not nb:
            continue
        act = act_start + i
        el = np.asarray(nb, dtype=np.intp)
        # Theiler window: skip everything inside [act - theiler,
        # act + theiler] (the reference point itself included).
        el = el[(el < act - theiler) | (el > act + theiler)]
        if el.size == 0:
            continue
        # Squared separations of every neighbour from the reference
        # for all evolution steps and embedding coordinates,
        # cumulated along the coordinates: cum[e, i, k] is the
        # partial distance through coordinates 0 .. k.
        base = x[act + steps[:, None] + delays[None, :]]
        nbr = x[el[:, None, None] + steps[None, :, None]
                + delays[None, None, :]]
        cum = np.cumsum((nbr - base) ** 2, axis=2)
        for j in range(n_dims):
            # Bucket of embedding dimension dims[j] holds the
            # neighbours whose partial distance through coordinates
            # 0 .. dims[j] - 1 (evaluated at evolution time 0) is
            # within eps.
            dcol = cum[:, :, min_dim - 1 + j]
            member = dcol[:, 0] <= eps2
            if not member.any():
                continue
            dx = dcol[member]
            pos = dx > 0.0
            c = pos.sum(axis=0)
            f = np.where(pos, dx, 0.0).sum(axis=0)
            ok = c > 0
            # (S_{\text{code}}(t)=0.5\log\left(\frac{1}{N}\sum_{k} D_k^2(t)\right))
            # \(S_{\text{paper}}(t)=\frac{1}{N}\sum_{k}\log(D_k(t))\)
            # S_{\text{paper}}(t) is defined in paper.
            contrib_count[i, j, ok] += 1
            contrib_lyap[i, j, ok] += 0.5 * np.log(f[ok] / c[ok])
    return contrib_lyap, contrib_count


def lyap_k(
    series,
    dim,
    delay,
    max_steps,
    eps_list=None,
    n_ref=None,
    min_dim=None,
    eps_min=None,
    eps_max=None,
    eps_count=5,
    theiler=0,
    n_jobs=None,
    backend="thread",
):
    """Estimate the maximal Lyapunov exponent using Kantz's method.

    Parameters
    ----------
    series : array_like
        Input scalar series (1-D).
    dim : int
        Maximal embedding dimension; must be >= 2.
    delay : int
        Time delay.
    max_steps : int
        Maximum evolution time of every neighbour pair.
    eps_list : array_like or None
        Explicit neighbourhood radii in the units of the input data.
        When given, ``eps_min``, ``eps_max`` and ``eps_count`` are
        ignored; duplicates are removed and the radii are used in
        ascending order.
    n_ref : int or None
        Number of reference points; the *first*
        ``n_ref`` points of the series are used.  ``None`` (default)
        uses every usable point.
    min_dim : int or None
        Smallest embedding dimension reported.
        ``None`` (default) uses 2.  Results are returned for every
        dimension ``min_dim .. dim``.
    eps_min, eps_max : float or None
        Ladder bounds in the units of the input data.  ``None`` defaults: the data interval
        divided by 1000 and by 100.  ``eps_min >= eps_max``
        collapses the ladder to the single radius ``eps_min``.
    eps_count : int
        Number of ladder radii when ``eps_list`` is not given.  The ladder ascends geometrically from ``eps_min`` to
        ``eps_max``.
    theiler : int
        Theiler window: series indices within
        ``theiler`` of a reference point are never used as its
        neighbours.
    n_jobs : int, optional (default = None)
        Workers for the (radius, reference-point chunk) task pool.
        ``None``/``1`` runs sequentially; ``-1`` uses all CPUs (see
        ``nolitisea.utils.parallel.parallel_map``).  Results are
        bitwise identical for every ``n_jobs``/``backend``.
    backend : {"thread", "process"}, optional (default = "thread")
        Executor backend.  ``"thread"`` suits the GIL-releasing
        NumPy/SciPy kernels; ``"process"`` pays pickling costs.

    Returns
    -------
    dict
        ``"eps"`` : radii in the original data units, ascending,
        shape ``(n_eps,)``.
        ``"dim"`` : embedding dimensions, shape ``(n_dims,)``.
        ``"times"`` : evolution times ``0 .. max_steps``.
        ``"s2"`` : Kantz statistic ``< log ||v|| >``, shape
        ``(n_eps, n_dims, max_steps + 1)``; ``nan`` where no reference
        point contributed.
        ``"count"`` : number of contributing reference points, same
        shape as ``s2``.

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the requested
        ``dim``/``delay``/``max_steps``.
    RuntimeError
        If the series is constant (zero value range).

    Notes
    -----
    The work is distributed over (radius, reference-point chunk) task
    pairs: each task queries the neighbours of its own points and
    returns the raw per-point contributions, and the caller adds them
    strictly in reference-point order.  The float accumulation order
    therefore matches the plain serial loop, so results are bitwise
    identical for every ``n_jobs`` and ``backend``.

    References
    ----------
    .. [1] Kantz, H. (1994). A robust method to estimate the maximal
           Lyapunov exponent of a time series.  *Physics Letters A*,
           185(1), 77-87.
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
    if min_dim is None:
        min_dim = 2
    if min_dim < 2:
        raise ValueError(f"min_dim must be >= 2, got {min_dim}")
    if min_dim > dim:
        raise ValueError(f"min_dim={min_dim} must not exceed dim={dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if theiler < 0:
        raise ValueError(f"theiler must be >= 0, got {theiler}")

    blength = length - (dim - 1) * delay - max_steps
    if blength < 1:
        raise ValueError(
            f"series of length {length} is too short for dim={dim}, "
            f"delay={delay}, max_steps={max_steps}"
        )

    if n_ref is None:
        n_ref = blength
    else:
        n_ref = int(n_ref)
        if n_ref < 1:
            raise ValueError(f"n_ref must be >= 1, got {n_ref}")
        n_ref = min(n_ref, blength)

    # Rescale to [0, 1]; radii are handled in rescaled units internally
    # and reported in the original data units.
    x, _, interval = rescale_data(x)

    if eps_list is not None:
        eps_arr = np.asarray(eps_list, dtype=np.float64).ravel()
        if eps_arr.size == 0:
            raise ValueError("eps_list must not be empty")
        if (eps_arr <= 0.0).any():
            raise ValueError("all radii must be > 0")
        eps_arr = np.unique(eps_arr) / interval
    else:
        if eps_count < 1:
            raise ValueError(f"eps_count must be >= 1, got {eps_count}")
        emin = interval / 1000.0 if eps_min is None else abs(float(eps_min))
        emax = interval / 100.0 if eps_max is None else abs(float(eps_max))
        emin /= interval
        emax /= interval
        if emin >= emax:
            emax = emin
            eps_count = 1
        if eps_count == 1:
            eps_arr = np.array([emin])
        else:
            eps_arr = emin * (emax / emin) ** (
                np.arange(eps_count) / (eps_count - 1)
            )

    # Neighbour search space: the first two embedding coordinates.
    # The tree is built once; only the query radius depends on eps.
    steps = np.arange(max_steps + 1)
    delays = np.arange(dim) * delay
    B = np.column_stack([x[:blength], x[delay:delay + blength]])
    tree = cKDTree(B)

    dims = np.arange(min_dim, dim + 1)
    n_dims = dims.size
    times = np.arange(max_steps + 1)

    s2 = np.full((eps_arr.size, n_dims, max_steps + 1), np.nan)
    counts = np.zeros((eps_arr.size, n_dims, max_steps + 1), dtype=np.int64)

    # Task pool over (radius, reference-point chunk) pairs: the radius
    # ladder and the reference points both contribute parallelism.  Each
    # task queries its own neighbours and returns raw per-point
    # contributions; the reduction below adds them strictly in
    # reference-point order, so the float accumulation grouping matches
    # the plain serial loop and results stay bitwise identical.
    bounds = [
        (start, min(start + _CHUNK, n_ref))
        for start in range(0, n_ref, _CHUNK)
    ]
    tasks = [
        (eps_now, start, stop)
        for eps_now in eps_arr
        for start, stop in bounds
    ]
    worker = partial(
        _lyap_k_chunk_worker,
        x=x,
        B=B,
        tree=tree,
        steps=steps,
        delays=delays,
        theiler=theiler,
        min_dim=min_dim,
        n_dims=n_dims,
        max_steps=max_steps,
    )
    results = parallel_map(worker, tasks, n_jobs=n_jobs, backend=backend)

    n_chunks = len(bounds)
    for li in range(eps_arr.size):
        ref_lyap = np.zeros((n_dims, max_steps + 1))
        ref_count = np.zeros((n_dims, max_steps + 1), dtype=np.int64)
        for ci in range(n_chunks):
            chunk_lyap, chunk_count = results[li * n_chunks + ci]
            for i in range(chunk_lyap.shape[0]):
                ref_lyap += chunk_lyap[i]
                ref_count += chunk_count[i]
        have = ref_count > 0
        s2[li, have] = ref_lyap[have] / ref_count[have]
        counts[li] = ref_count

    return {
        "eps": eps_arr * interval,
        "dim": dims,
        "times": times,
        "s2": s2,
        "count": counts,
    }
