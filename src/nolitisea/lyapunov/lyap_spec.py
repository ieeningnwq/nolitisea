"""Full Lyapunov spectrum via the Sano and Sawada method."""

from functools import partial

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.utils.parallel import parallel_map

__all__ = ["lyap_spec"]


# Reference points whose local linear model is fitted per parallel
# task; bounds the memory of the per-step ``(n_vars, alldim)``
# dynamics tensor and the number of dispatched workers.
_FIT_CHUNK = 256


def _lyap_spec_fit_worker(bound, E, arr, dists_all, idxs_all, n_neighbors,
                          alldim, n_vars, embed):
    """Fit local linear models for one chunk of reference points.

    Covers reference points ``bound[0]:bound[1]``.  Returns
    ``(valid, dynamics, nb, eps)`` for that slice, where ``valid``
    flags the steps that produced a model; the caller skips the
    others.  Every step is data-only (neighbour query result +
    ``lstsq`` fit), so the chunk is independent of the tangent-space
    evolution and the result is identical for every execution order.

    The delay between consecutive embedding coordinates is fixed at 1;
    the tangent-space shift in :func:`lyap_spec` is only valid for that
    value.
    """
    start, stop = bound
    n = stop - start
    valid = np.zeros(n, dtype=bool)
    dynamics = np.zeros((n, n_vars, alldim))
    nb = np.zeros(n, dtype=np.int64)
    eps = np.zeros(n)
    chunk_d = dists_all[start:stop]
    chunk_i = idxs_all[start:stop]
    for j in range(n):
        step = start + j
        d = chunk_d[j]
        i = chunk_i[j]
        # Exclude self-reference.
        mask = i != step
        if not mask.any():
            continue
        d = d[mask]
        i = i[mask]
        if i.size < alldim + 1:
            continue  # too few for a well-determined model
        if i.size > n_neighbors:
            d = d[:n_neighbors]
            i = i[:n_neighbors]
        # Fit the local linear model with an explicit intercept.
        # Only the gradient columns (``beta[1:]``) affect the
        # tangent-space evolution.
        target_indices = i + embed  # delay = 1
        X_aug = np.column_stack([np.ones(i.size), E[i]])
        for dd in range(n_vars):
            y_t = arr[target_indices, dd]
            beta, *_ = np.linalg.lstsq(X_aug, y_t, rcond=None)
            dynamics[j, dd] = beta[1:]
        valid[j] = True
        nb[j] = i.size
        eps[j] = d[-1]
    return valid, dynamics, nb, eps


def lyap_spec(series, embed, n_iter=None, dt=1.0,
              n_neighbors=30, seed=0, n_jobs=None, backend="thread"):
    """Estimate the full Lyapunov spectrum via Sano and Sawada.

    Parameters
    ----------
    series : array_like
        Input series.  A 1-D array is treated as a single component;
        a 2-D array must have shape ``(n_times, n_vars)``.
    embed : int
        Embedding dimension *per component*.
        The total number of Lyapunov exponents returned is
        ``alldim = n_vars * embed``.

        The delay between consecutive embedding coordinates is fixed
        at 1: the tangent-space shift that propagates the perturbation
        matrix is only valid when each step advances the reference
        point by one sample, so the old delay coordinates age into
        the next slot.  With ``delay > 1`` that shift is invalid.
    n_iter : int or None
        Number of reference-point iterations.  ``None``
        (default) uses every usable point.
    dt : float, default 1.0
        Sampling interval.  The exponents are divided by ``dt`` so that
        a continuous-time system yields exponents per unit time.  For
        discrete maps leave ``dt = 1``.
    n_neighbors : int, default 30
        Number of nearest neighbours for each local linear model.
        Must be at least ``alldim + 1``.
    seed : int, default 0
        Seed for the initial random perturbation matrix.
    n_jobs : int, optional (default = None)
        Workers for the local-linear-model fit task pool.
        ``None``/``1`` runs sequentially; ``-1`` uses all CPUs (see
        ``nolitisea.utils.parallel.parallel_map``).  Results are
        bitwise identical for every ``n_jobs``/``backend``.
    backend : {"thread", "process"}, optional (default = "thread")
        Executor backend.  ``"thread"`` suits the GIL-releasing
        NumPy/SciPy kernels (the batched ``cKDTree`` query and the
        ``lstsq`` fits); ``"process"`` pays pickling costs.

    Returns
    -------
    dict
        ``"exponents"`` : array of Lyapunov exponents in descending
        order, shape ``(alldim,)``.
        ``"times"`` : cumulative number of reference points processed,
        shape ``(count,)``.
        ``"running"`` : running average of the exponents, shape
        ``(count, alldim)``.
        ``"ky_dim"`` : estimated Kaplan--Yorke dimension.
        ``"ave_neighbors"`` : average number of neighbours found per
        reference point.
        ``"ave_eps"`` : average Chebyshev radius of the neighbour
        cloud (on the rescaled [0, 1] data).
        ``"forecast_err"`` : placeholder (not computed in this rewrite).

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the requested
        ``embed``/``n_iter``.
    RuntimeError
        If any component is constant (zero value range), or if not
        enough neighbours can be found for the local linear model.

    Notes
    -----
    The computation is split into two phases.  Phase 1 fits, for every
    reference point, the local linear model: a single batched
    ``cKDTree`` query retrieves all neighbour lists, then
    ``parallel_map`` runs the ``lstsq`` fits over reference-point
    chunks.  Phase 2 is a strict serial fold of the perturbation matrix
    through the precomputed models (``dnew -> QR -> log |R_{jj}|``), so
    the ``delta`` forward dependency is preserved exactly.  Because the
    per-step fits are data-only and independent, and the fold runs in
    reference-point order, results are bitwise identical for every
    ``n_jobs`` and ``backend``.

    References
    ----------
    .. [1] Sano, M., & Sawada, Y. (1985). Measurement of the Lyapunov
           spectrum from a chaotic time series.  *Physical Review
           Letters*, 55(10), 1082-1085.
    .. [2] Hegger, R., Kantz, H., & Schreiber, T. (1999). Practical
           implementation of nonlinear time series methods: The TISEAN
           package.  *Chaos*, 9(2), 413-435.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(
            "series must be a 1-D array or a 2-D array of shape "
            "(n_times, n_vars)"
        )
    n_times, n_vars = arr.shape

    if embed < 1:
        raise ValueError(f"embed must be >= 1, got {embed}")
    alldim = n_vars * embed
    if n_neighbors < alldim + 1:
        raise ValueError(
            f"n_neighbors={n_neighbors} is too small for alldim={alldim}; "
            f"need at least {alldim + 1} for a well-determined linear model"
        )

    # Rescale each component to [0, 1]; every component is normalised
    # independently so that components with vastly different scales
    # contribute equally to the Chebyshev distance.
    for c in range(n_vars):
        comp = arr[:, c]
        rng = np.ptp(comp)
        if rng < 1e-30:
            raise RuntimeError(f"component {c} is constant (zero range)")
        arr[:, c] = (comp - comp.min()) / rng

    # Delay-coordinate embedding.  Coordinates are interleaved:
    # [comp0@delay0, comp1@delay0, ..., comp0@delay1, comp1@delay1, ...].
    # The delay is fixed at 1.
    E = lag_block_delay_embed(arr, embed, 1)

    # A neighbor at E-row ``i_c`` has original base time
    # ``i_c + (embed - 1)`` and target index ``i_c + embed``; the
    # latter must satisfy ``i_c + embed < n_times``.
    n_ref_max = n_times - embed
    if n_ref_max < 1:
        raise ValueError(
            f"series of length {n_times} is too short for embed={embed}"
        )
    # The tree needs at least ``n_neighbors`` rows that are eligible
    # after excluding self-reference.
    if n_ref_max <= n_neighbors:
        raise ValueError(
            f"series of length {n_times} with embed={embed} "
            f"leaves only {n_ref_max} usable rows, but n_neighbors={n_neighbors} "
            f"(need at least {n_neighbors + 1})"
        )
    if n_iter is None:
        n_iter = n_ref_max
    n_iter = min(int(n_iter), n_ref_max)

    tree = cKDTree(E[:n_ref_max])

    # Initial perturbation matrix: random, then QR-orthogonalised.
    # Gram-Schmidt operates on ROWS of ``delta``, so we QR the
    # TRANSPOSE: ``delta.T = Q @ R`` gives the stretch factors on the
    # diagonal of R, and ``Q.T`` has orthonormal rows.
    rng = np.random.default_rng(seed)
    delta = rng.standard_normal((alldim, alldim))
    Q, _ = np.linalg.qr(delta.T)
    delta = Q.T  # orthonormal rows

    log_sum = np.zeros(alldim)
    running = np.full((n_iter, alldim), np.nan)
    times_arr = np.arange(1, n_iter + 1)
    count_out = 0

    k_query = min(n_neighbors + 1, n_ref_max)

    # --- Phase 1: data-only local linear models (parallel) ----------
    # A single batched query retrieves every reference point's neighbour
    # list at once.  The per-step
    # ``lstsq`` fits depend only on the data, not on the perturbation
    # matrix, so they are distributed over reference-point chunks; each
    # worker returns its slice of results and the caller assembles them
    # in reference-point order.
    dists_all, idxs_all = tree.query(E[:n_iter], k=k_query, p=np.inf)
    if k_query == 1:
        dists_all = dists_all[:, None]
        idxs_all = idxs_all[:, None]

    valid_all = np.zeros(n_iter, dtype=bool)
    dynamics_all = np.zeros((n_iter, n_vars, alldim))
    nb_all = np.zeros(n_iter, dtype=np.int64)
    eps_all = np.zeros(n_iter)

    bounds = [(s, min(s + _FIT_CHUNK, n_iter))
              for s in range(0, n_iter, _FIT_CHUNK)]
    fit_worker = partial(
        _lyap_spec_fit_worker,
        E=E,
        arr=arr,
        dists_all=dists_all,
        idxs_all=idxs_all,
        n_neighbors=n_neighbors,
        alldim=alldim,
        n_vars=n_vars,
        embed=embed,
    )
    results = parallel_map(fit_worker, bounds, n_jobs=n_jobs, backend=backend)
    for (start, _), (valid, dynamics, nb, eps) in zip(bounds, results):
        valid_all[start:start + valid.size] = valid
        dynamics_all[start:start + dynamics.shape[0]] = dynamics
        nb_all[start:start + nb.size] = nb
        eps_all[start:start + eps.size] = eps

    # --- Phase 2: serial fold of the perturbation matrix ------------
    # ``delta`` is a forward state variable: each step's ``dnew`` uses
    # the previous ``delta`` and the precomputed ``dynamics`` of that
    # step, and QR produces the ``delta`` fed into the next step.  This
    # dependency is irreducible, so the fold runs strictly in
    # reference-point order; it matches the plain serial loop and keeps
    # results bitwise identical for every ``n_jobs``/``backend``.
    ave_neighbors = 0.0
    ave_eps = 0.0
    for step in range(n_iter):
        if not valid_all[step]:
            continue
        dynamics = dynamics_all[step]
        ave_neighbors += nb_all[step]
        ave_eps += eps_all[step]

        # Apply dynamics to perturbation vectors.  The tangent-space
        # evolution has two parts:
        #  * the first ``n_vars`` entries get the dynamics applied
        #  * the remaining entries (delay coordinates) simply shift
        #    down by one delay step
        #
        # In matrix form:
        #   dnew[:, :n_vars] = delta @ dynamics.T
        #   dnew[:, n_vars:] = delta[:, n_vars-1:alldim-1]
        dnew = np.empty_like(delta)
        dnew[:, :n_vars] = delta @ dynamics.T
        if alldim > n_vars:
            dnew[:, n_vars:] = delta[:, n_vars - 1:alldim - 1]

        # QR re-orthogonalise rows.  ``|R[j, j]|`` gives the stretch
        # factor for perturbation direction ``j`` before normalisation.
        Q, R = np.linalg.qr(dnew.T)
        stretch = np.abs(np.diag(R))
        stretch = np.where(stretch > 0.0, stretch, 1.0)

        log_sum += np.log(stretch)
        count_out += 1
        running[count_out - 1] = log_sum / count_out / dt
        delta = Q.T

    if count_out == 0:
        raise RuntimeError(
            "Could not fit any local linear models; series may be too short "
            "or n_neighbors too large."
        )

    exponents = log_sum / count_out / dt
    # The QR re-orthogonalisation places the largest stretch on the first
    # diagonal entry, so ``exponents`` is already in descending order.

    # Kaplan--Yorke dimension:
    # D_KY = k + (sum_{j=1}^k lambda_j) / |lambda_{k+1}|
    # where k is the largest index with non-negative cumulative sum.
    cum = np.cumsum(exponents)
    neg = cum < 0
    if neg.any():
        idx = np.argmax(neg)
        prev_cum = cum[idx - 1] if idx > 0 else 0.0
        ky = idx + prev_cum / abs(exponents[idx])
    else:
        ky = alldim

    ave_neighbors /= count_out
    ave_eps /= count_out

    return {
        "exponents": exponents,
        "times": times_arr[:count_out],
        "running": running[:count_out],
        "ky_dim": ky,
        "ave_neighbors": ave_neighbors,
        "ave_eps": ave_eps,
        "forecast_err": None,
    }
