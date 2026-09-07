"""Locally first-order (linear) prediction.

* :func:`lfo_ar` — epsilon-scan over neighbourhood sizes
  (``lfo-ar``) for parameter tuning.
* :func:`lfo_test` — average forecast error with *adaptive*
  neighbourhoods (``lfo`` / ``lfo-test``), expanding epsilon until
  every reference point has enough neighbours.
* :func:`lfo_run` — iterative one-step extrapolation of a trajectory
  (``lfo-run``), with optional zeroth-order fit and escape detection.

All three share the same core engine: rescale each component to
``[0, 1]``, build a delay embedding, use a Chebyshev-neighbour
``scipy.spatial.cKDTree``, fit a centred OLS local linear model, and
predict.  The differences are purely in workflow — when and how
epsilon is chosen, whether causal exclusion is applied, and whether
the query point is historical data or an iteratively updated rolling
state.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.utils.rescale import rescale_data

__all__ = [
    # helpers (re-exported by thin wrapper submodules)
    "_build_embedding",
    "_local_linear_predict",
    "_resolve_inputs",
    "_scan_epsilons",
    "lfo_ar",
    "lfo_run",
    "lfo_test",
]


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _resolve_inputs(series):
    """Return ``(n_times, n_vars, s2d)`` where ``s2d`` is ``(n_times, n_vars)``."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("series must be a 1-D or 2-D array")
    n_times, n_vars = arr.shape
    return n_times, n_vars, arr


def _build_embedding(s2d, embed, delay):
    """Return ``(E, valid_start)``.

    ``E`` is the delay embedding matrix ``(n_points, n_vars * embed)``.
    Row ``r`` of ``E`` corresponds to the original time index
    ``t = r + valid_start``, so the mapping is

        E[r]  <->  original time  t = r + valid_start
    """
    return lag_block_delay_embed(s2d, embed=embed, delay=delay), (embed - 1) * delay


def _scan_epsilons(eps0, eps1, eps_factor):
    """Yield geometrically spaced epsilon values up to ``eps1 * factor``.

    Matches the C loop: ``for epsilon = EPS0; epsilon < EPS1 * EPSF;
    epsilon *= EPSF``.
    """
    eps = float(eps0)
    while eps < eps1 * eps_factor:
        yield eps
        eps *= eps_factor


def _local_linear_predict(X, y, query):
    """Centred OLS fit (without intercept) + predict at ``query``.

    Matches TISEAN ``make_fit`` exactly: regressions are centred on
    the neighbours' embedding and target means, then the fitted
    coefficient is applied to the centred query point and finally
    the target mean is added back.  ``numpy.linalg.lstsq`` handles
    rank-deficient cases gracefully.
    """
    X_mean = X.mean(axis=0)
    Xc = X - X_mean
    y_mean = y.mean(axis=0)
    yc = y - y_mean
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    return y_mean + (query - X_mean) @ beta


# ---------------------------------------------------------------------------
# lfo-ar  —  epsilon-scan (fixed epsilon per pass)
# ---------------------------------------------------------------------------


def lfo_ar(
    series,
    embed=2,
    delay=1,
    step=1,
    causal=None,
    eps0=None,
    eps1=None,
    eps_factor=1.2,
    min_neighbors=None,
):
    """Scan neighbourhood sizes and record the average local forecast
    error for a local linear AR model.

    For each candidate ``epsilon`` (geometric scan starting at
    ``eps0``, multiplying by ``eps_factor`` up to ``eps1 *
    eps_factor``), the function:

    1. Normalises each component via :func:`rescale_data` and uses the
       largest component range as the ``epsilon`` scale (so
       ``epsilon`` values are relative to the data box size).
    2. Builds the lag-block delay embedding in column order.
    3. For each reference point ``i`` (original time), finds all
       ``epsilon``-neighbours in Chebyshev distance and excludes
       those inside the causality window.
    4. With the remaining neighbours, fits a local linear model that
       predicts each component's value at ``i + step`` from the
       embedding, using :func:`numpy.linalg.lstsq` (rank-deficient
       cases are handled gracefully).
    5. Records the root-mean-square prediction error normalised by
       the sample standard deviation of the target component over
       all valid reference points.

    Parameters
    ----------
    series : array_like
        1-D array (single component) or 2-D array of shape
        ``(n_times, n_vars)``.
    embed : int, default 2
        Embedding dimension per component.
    delay : int, default 1
        Delay between successive embedding coordinates.
    step : int, default 1
        Forecast horizon.
    causal : int or None, default None
        Causality window half-width.  ``None`` defaults to ``step``.  Neighbour times in
        ``[i - causal + 1,
         i + causal + (embed - 1) * delay - 1]`` are excluded.
    eps0 : float or None, default None
        Starting neighbourhood size, given in *relative* units
        (fraction of the largest component range).  ``None`` uses
        ``interval / 1000``, matching the C default.
    eps1 : float or None, default None
        Final neighbourhood size (relative).  ``None`` uses the full
        interval (``1.0`` after normalisation).
    eps_factor : float, default 1.2
        Geometric multiplier between successive epsilon values.
    min_neighbors : int or None, default None
        Minimum number of neighbours (after causal exclusion) required
        to fit.  ``None`` uses the default
        ``2 * (n_vars * embed + 1)``.  Reference points with fewer
        neighbours than this are skipped.

    Returns
    -------
    dict
        Keys:

        - ``"epsilon"`` : 1-D float array of relative neighbourhood
          sizes (length ``n_epsilons``).
        - ``"avg_error"`` : 1-D float array, average relative forecast
          error across components for each epsilon.
        - ``"comp_errors"`` : 2-D array of shape
          ``(n_epsilons, n_vars)``, per-component relative forecast
          errors.
        - ``"coverage"`` : 1-D float array, fraction of reference
          points that had enough neighbours to fit.
        - ``"avg_neighbors"`` : 1-D float array, mean number of
          neighbours actually used per fit.
        - ``"embedding_norm"`` : the per-component rescale info — a
          dict mapping component index to ``(minv, interval)`` so the
          caller can interpret ``epsilon`` values relative to the
          original data.
        - ``"n_ref_points"`` : int, number of valid reference points.

    Raises
    ------
    ValueError
        For invalid parameters, an input that cannot be rescaled, or a
        series too short for the requested embedding + forecast
        parameters.
    """
    n_times, n_vars, s2d = _resolve_inputs(series)

    # --- parameter validation ------------------------------------------
    if embed < 1:
        raise ValueError(f"embed must be >= 1, got {embed}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if causal is None:
        causal = step
    elif causal < 1:
        raise ValueError(f"causal must be >= 1, got {causal}")
    if eps_factor <= 1.0:
        raise ValueError(f"eps_factor must be > 1.0, got {eps_factor}")

    if n_times <= step + (embed - 1) * delay:
        raise ValueError(
            f"series of length {n_times} is too short for embed={embed}, "
            f"delay={delay}, step={step} (need > step + (embed-1)*delay "
            f"= {step + (embed - 1) * delay})"
        )

    if min_neighbors is None:
        min_neighbors = 2 * (n_vars * embed + 1)
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")

    # --- rescale each component ---------------------------------------
    rescale_info = {}
    rescaled = np.empty_like(s2d)
    max_interval = 0.0
    for c in range(n_vars):
        sr, minv, interval = rescale_data(s2d[:, c])
        rescale_info[c] = (minv, interval)
        rescaled[:, c] = sr
        max_interval = max(max_interval, interval)

    # --- default epsilon bounds (C convention, already relative here) -
    if eps0 is None:
        eps0 = 1.0 / 1000.0
    if eps1 is None:
        eps1 = 1.0

    # --- delay embedding -----------------------------------------------
    E, valid_start = _build_embedding(rescaled, embed, delay)
    n_ref_total = n_times - step - valid_start
    ref_orig_times = np.arange(valid_start, n_times - step)

    # --- build KD-tree for Chebyshev neighbours -----------------------
    tree = cKDTree(E)

    # --- results accumulators ----------------------------------------
    epsilons = []
    avg_errors = []
    comp_errors = []
    coverages = []
    avg_neighbors_list = []

    for eps_rel in _scan_epsilons(eps0, eps1, eps_factor):
        epsilon = float(eps_rel)
        sum_err2 = np.zeros(n_vars)
        sum_hrms2 = np.zeros(n_vars)
        sum_hav = np.zeros(n_vars)
        pfound = 0
        sum_nbrs = 0

        for ref_idx, i_orig in enumerate(ref_orig_times):
            nbr_embed_idx = np.asarray(
                tree.query_ball_point(E[ref_idx], epsilon, p=np.inf),
                dtype=np.intp,
            )
            nbr_times = nbr_embed_idx + valid_start
            keep = nbr_times <= n_times - step - 1
            nbr_embed_idx = nbr_embed_idx[keep]
            nbr_times = nbr_times[keep]
            lo = i_orig - causal + 1
            hi = i_orig + causal + (embed - 1) * delay - 1
            keep = (nbr_times < lo) | (nbr_times > hi)
            nbr_embed_idx = nbr_embed_idx[keep]
            nbr_times = nbr_times[keep]

            n_nbrs = len(nbr_times)
            if n_nbrs < min_neighbors:
                continue

            X = E[nbr_times - valid_start]
            y_all = rescaled[nbr_times + step, :]

            X_mean = X.mean(axis=0)
            Xc = X - X_mean
            y_mean = y_all.mean(axis=0)
            yc = y_all - y_mean

            beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)

            E_i_centred = E[ref_idx] - X_mean
            y_pred = y_mean + E_i_centred @ beta
            y_true = rescaled[i_orig + step, :]

            sum_err2 += (y_pred - y_true) ** 2
            pfound += 1
            y_ref = rescaled[i_orig + step, :]
            sum_hrms2 += y_ref**2
            sum_hav += y_ref
            sum_nbrs += max(0, n_nbrs - 1)

        if pfound <= 1:
            continue

        hav = sum_hav / pfound
        hrms = np.sqrt(np.maximum(0.0, (sum_hrms2 - pfound * hav**2) / (pfound - 1)))
        err_per_comp = np.sqrt(sum_err2 / pfound) / hrms
        err_per_comp = np.where(hrms > 0, err_per_comp, np.nan)
        avg_error = float(np.nanmean(err_per_comp))

        epsilons.append(epsilon)
        avg_errors.append(avg_error)
        comp_errors.append(err_per_comp)
        coverages.append(pfound / n_ref_total)
        avg_neighbors_list.append(sum_nbrs / pfound)

    if not epsilons:
        raise RuntimeError(
            "no epsilon values produced (did all scans fail the pfound > 1 "
            "gate?  Increase eps1 or lower min_neighbors)"
        )

    return {
        "epsilon": np.asarray(epsilons),
        "avg_error": np.asarray(avg_errors),
        "comp_errors": np.asarray(comp_errors),
        "coverage": np.asarray(coverages),
        "avg_neighbors": np.asarray(avg_neighbors_list),
        "embedding_norm": rescale_info,
        "n_ref_points": n_ref_total,
    }


# ---------------------------------------------------------------------------
# lfo-test  —  average forecast error with adaptive neighbourhoods
# ---------------------------------------------------------------------------


def lfo_test(
    series,
    embed=2,
    delay=1,
    step=1,
    causal=None,
    eps0=None,
    eps_factor=1.2,
    min_neighbors=30,
    max_epsilon=10.0,
):
    """Estimate the average local linear forecast error with an
    adaptively expanding neighbourhood.

    Unlike :func:`lfo_ar`, which scans a fixed epsilon grid and skips
    reference points that have too few neighbours, this function
    starts from a small ``epsilon`` and multiplies it by
    ``eps_factor`` on each round until **every** reference point has
    at least ``min_neighbors`` neighbours (after causal exclusion).
    The final result is a single per-component relative forecast
    error.

    Matches the TISEAN ``lfo`` / ``lfo-test`` program.

    Parameters
    ----------
    series : array_like
        1-D array (single component) or 2-D array of shape
        ``(n_times, n_vars)``.
    embed : int, default 2
        Embedding dimension per component.
    delay : int, default 1
        Delay between successive embedding coordinates.
    step : int, default 1
        Forecast horizon.
    causal : int or None, default None
        Causality window half-width.  ``None`` defaults to ``step``.
    eps0 : float or None, default None
        Starting neighbourhood size, given in *relative* units
        (fraction of the largest component range).  ``None`` uses
        ``1/1000``, matching the C default.
    eps_factor : float, default 1.2
        Geometric multiplier for the adaptive epsilon expansion loop.
    min_neighbors : int, default 30
        Minimum number of neighbours (after causal exclusion) required
        to accept a fit.  Reference points that still cannot reach
        this threshold once ``epsilon`` grows beyond ``max_epsilon``
        are reported in the result but do not contribute to the error.
    max_epsilon : float, default 10.0
        Safety cap on the epsilon expansion loop (in relative units).
        The data box is ``[0, 1]`` in rescaled space, so ``10.0`` is
        generous.  Prevents infinite loops on pathologically sparse
        data.

    Returns
    -------
    dict
        Keys:

        - ``"comp_errors"`` : 1-D float array of length ``n_vars``,
          per-component relative forecast error
          ``sqrt(MSE) / overall_std``.
        - ``"avg_error"`` : float, mean of ``comp_errors`` across
          components.
        - ``"final_epsilon"`` : float, the epsilon value at which the
          last reference point was accepted (or ``max_epsilon`` if
          some points remained unresolved).
        - ``"n_done"`` : int, number of reference points that had
          enough neighbours.
        - ``"n_total"`` : int, total number of reference points.
        - ``"n_unresolved"`` : int, number of reference points that
          still could not reach ``min_neighbors`` at ``max_epsilon``.
        - ``"embedding_norm"`` : per-component ``(minv, interval)``
          dict.
        - ``"rms"`` : 1-D float array of length ``n_vars``, the
          overall sample standard deviation of each component
          (used as the denominator in the relative error).

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the requested
        embedding + forecast parameters.
    """
    n_times, n_vars, s2d = _resolve_inputs(series)

    # --- parameter validation ------------------------------------------
    if embed < 1:
        raise ValueError(f"embed must be >= 1, got {embed}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if causal is None:
        causal = step
    elif causal < 1:
        raise ValueError(f"causal must be >= 1, got {causal}")
    if eps_factor <= 1.0:
        raise ValueError(f"eps_factor must be > 1.0, got {eps_factor}")
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")
    if max_epsilon <= 0:
        raise ValueError(f"max_epsilon must be > 0, got {max_epsilon}")

    hdim = (embed - 1) * delay
    if n_times <= step + hdim:
        raise ValueError(
            f"series of length {n_times} is too short for embed={embed}, "
            f"delay={delay}, step={step} (need > step + hdim = {step + hdim})"
        )

    # --- rescale each component ---------------------------------------
    rescale_info = {}
    rescaled = np.empty_like(s2d)
    max_interval = 0.0
    for c in range(n_vars):
        sr, minv, interval = rescale_data(s2d[:, c])
        rescale_info[c] = (minv, interval)
        rescaled[:, c] = sr
        max_interval = max(max_interval, interval)

    # --- overall rms per component (denominator for relative error) --
    overall_std = rescaled.std(axis=0, ddof=1)  # sample std

    # --- delay embedding -----------------------------------------------
    E, valid_start = _build_embedding(rescaled, embed, delay)
    n_ref_total = n_times - step - valid_start
    ref_orig_times = np.arange(valid_start, n_times - step)

    # --- default epsilon ----------------------------------------------
    if eps0 is None:
        eps0 = 1.0 / 1000.0
    eps_start = float(eps0)

    # --- build KD-tree ------------------------------------------------
    tree = cKDTree(E)

    # --- adaptive epsilon expansion -----------------------------------
    done = np.zeros(n_ref_total, dtype=bool)
    sum_err2 = np.zeros(n_vars)
    epsilon = eps_start / eps_factor
    final_epsilon = eps_start

    while not np.all(done):
        epsilon *= eps_factor
        final_epsilon = epsilon

        if epsilon > max_epsilon:
            break  # safety cap — remaining points stay unresolved

        for ref_idx, i_orig in enumerate(ref_orig_times):
            if done[ref_idx]:
                continue

            nbr_embed_idx = np.asarray(
                tree.query_ball_point(E[ref_idx], epsilon, p=np.inf),
                dtype=np.intp,
            )
            nbr_times = nbr_embed_idx + valid_start
            # C's LENGTH - STEP cutoff
            keep = nbr_times <= n_times - step - 1
            nbr_embed_idx = nbr_embed_idx[keep]
            nbr_times = nbr_times[keep]
            # Causality window
            lo = i_orig - causal + 1
            hi = i_orig + causal + (embed - 1) * delay - 1
            keep = (nbr_times < lo) | (nbr_times > hi)
            nbr_embed_idx = nbr_embed_idx[keep]
            nbr_times = nbr_times[keep]

            n_nbrs = len(nbr_times)
            if n_nbrs <= min_neighbors:
                continue  # need strictly more than min_neighbors (C: > MINN)

            X = E[nbr_times - valid_start]
            y_all = rescaled[nbr_times + step, :]
            y_true = rescaled[i_orig + step, :]

            y_pred = _local_linear_predict(X, y_all, E[ref_idx])

            sum_err2 += (y_pred - y_true) ** 2
            done[ref_idx] = True

    n_done = int(done.sum())
    n_unresolved = int((~done).sum())

    # --- relative error: sqrt(MSE / n_total) / overall_std ---------
    # C code uses norm = clength - hdim (total ref points, not just done).
    # When all points are done this is equivalent to sqrt(MSE_done) / std.
    # When some are unresolved, the C code would have kept expanding;
    # here we use n_total as the normalization denominator.
    norm = float(n_ref_total)
    comp_errors = np.zeros(n_vars)
    for c in range(n_vars):
        if overall_std[c] > 0 and norm > 0:
            comp_errors[c] = np.sqrt(sum_err2[c] / norm) / overall_std[c]
        else:
            comp_errors[c] = np.nan

    avg_error = float(np.nanmean(comp_errors))

    return {
        "comp_errors": comp_errors,
        "avg_error": avg_error,
        "final_epsilon": final_epsilon,
        "n_done": n_done,
        "n_total": n_ref_total,
        "n_unresolved": n_unresolved,
        "embedding_norm": rescale_info,
        "rms": overall_std,
    }


# ---------------------------------------------------------------------------
# lfo-run  —  iterative trajectory extrapolation
# ---------------------------------------------------------------------------


def lfo_run(
    series,
    embed=2,
    delay=1,
    n_steps=1000,
    method="linear",
    eps0=None,
    eps_factor=1.2,
    min_neighbors=30,
    escape_scale=1.0,
):
    """Iterate a local phase-space model forward.

    Builds a Chebyshev-neighbour cKDTree once from the delay-embedded
    rescaled data, then iterates from the last embedding state.  At
    each step the function looks up ``min_neighbors`` neighbours via
    an adaptively expanding epsilon, fits a local linear or zeroth-
    order model, extrapolates one step, and advances a rolling ring
    buffer.

    Matches the TISEAN ``lfo-run`` program (no causal exclusion — the
    query point is a synthetic rolling state, never part of the
    historical database).

    Parameters
    ----------
    series : array_like
        Input data of shape ``(n_times,)`` or ``(n_times, n_vars)``.
    embed : int, default 2
        Embedding dimension per component.
    delay : int, default 1
        Time delay between successive embedding coordinates.
    n_steps : int, default 1000
        Number of one-step extrapolations to perform.
    method : {"linear", "zeroth"}, default "linear"
        ``"linear"`` fits a local linear model by centred OLS;
        ``"zeroth"`` takes the mean of the neighbours' next values
        (the C ``-0`` option).
    eps0 : float or None, default None
        Starting neighbour radius in *relative* units (fraction of the
        data box).  ``None`` uses ``1/1000``.
    eps_factor : float, default 1.2
        Geometric multiplier for the adaptive epsilon expansion loop.
    min_neighbors : int, default 30
        Minimum number of neighbours required to fit.
    escape_scale : float, default 1.0
        Escape threshold.  A forecast whose rescaled value lies outside
        ``[-escape_scale, 1 + escape_scale]`` (the data box padded
        by ``escape_scale`` on both sides) terminates the iteration.

    Returns
    -------
    dict
        Keys:

        - ``"trajectory"`` : array of shape ``(n_steps_done, n_vars)``
          with the extrapolated values in **original data units**
          (rescaled back).
        - ``"n_steps_done"`` : int, number of steps actually completed
          before termination (equals ``n_steps`` if no escape).
        - ``"status"`` : str, one of ``"ok"``, ``"escaped"``, or
          ``"failed"`` (epsilon ran out without finding
          ``min_neighbors`` neighbours at any step).
        - ``"final_epsilon"`` : float, the epsilon used at the last
          successfully extrapolated step (relative).
        - ``"rescale_info"`` : per-component ``(minv, interval)``
          dict.

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the requested
        embedding.
    """
    if embed < 1:
        raise ValueError(f"embed must be >= 1, got {embed}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if method not in ("linear", "zeroth"):
        raise ValueError(f"method must be 'linear' or 'zeroth', got {method!r}")
    if eps_factor <= 1.0:
        raise ValueError(f"eps_factor must be > 1.0, got {eps_factor}")
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")
    if escape_scale < 0:
        raise ValueError(f"escape_scale must be >= 0, got {escape_scale}")

    n_times, n_vars, s2d = _resolve_inputs(series)
    valid_start = (embed - 1) * delay
    if n_times <= valid_start + 1:
        raise ValueError(
            f"series of length {n_times} is too short for embed={embed}, "
            f"delay={delay} (need at least {valid_start + 2} samples so "
            f"each embedding point has a one-step target)"
        )

    # --- rescale each component to [0, 1] ----------------------------
    rescale_info = {}
    rescaled = np.empty_like(s2d)
    max_interval = 0.0
    for c in range(n_vars):
        sr, minv, interval = rescale_data(s2d[:, c])
        rescale_info[c] = (minv, interval)
        rescaled[:, c] = sr
        max_interval = max(max_interval, interval)

    # --- build embedding database -----------------------------------
    E = lag_block_delay_embed(rescaled, embed=embed, delay=delay)
    n_embed_points = E.shape[0]

    # Each row E[r] corresponds to original time t = r + valid_start.
    # Keep rows with t <= n_times - 2 so t+1 is valid (C's LENGTH - 1 cutoff).
    last_valid_row = n_embed_points - 2
    E = E[: last_valid_row + 1]
    # E[r] corresponds to original time t = r + valid_start.
    # The one-step target for a point at time t is rescaled[t+1],
    # so targets[r] = rescaled[r + valid_start + 1].
    targets = rescaled[valid_start + 1 : valid_start + last_valid_row + 2, :]

    # --- KD-tree (Chebyshev distance, p=np.inf) ----------------------
    tree = cKDTree(E)

    # --- initial rolling state --------------------------------------
    initial_row = last_valid_row
    current = E[initial_row].copy()

    # --- epsilon default --------------------------------------------
    eps_start = float(eps0) if eps0 is not None else 1.0 / 1000.0

    # --- iterate -----------------------------------------------------
    out = np.empty((n_steps, n_vars), dtype=np.float64)
    status = "ok"
    final_eps = eps_start
    step_done = 0

    for step_done in range(n_steps):
        # Adaptive epsilon expansion (matches C: each iteration starts
        # at eps0 / eps_factor, then multiplies until min_neighbors).
        epsilon = eps_start / eps_factor
        nbr_indices = None
        while True:
            epsilon *= eps_factor
            nbr_indices = tree.query_ball_point(current, epsilon, p=np.inf)
            if len(nbr_indices) >= min_neighbors:
                break
            if epsilon > 1.0 + 2.0 * escape_scale:  # data in [0,1]
                status = "failed"
                break
        if status != "ok":
            break

        nbr_indices = np.asarray(nbr_indices, dtype=np.intp)
        X = E[nbr_indices]
        y = targets[nbr_indices]

        if method == "linear":
            newpoint = _local_linear_predict(X, y, current)
        else:  # zeroth
            newpoint = y.mean(axis=0)

        # Write output BEFORE escape check (matches C: output newcast,
        # then test for region escape and exit).
        out[step_done] = newpoint
        final_eps = epsilon

        # Escape check: C uses > 2 or < -1 after rescaling (data in [0,1]).
        if np.any(newpoint < -escape_scale) or np.any(newpoint > 1.0 + escape_scale):
            status = "escaped"
            break

        # Advance rolling state — shift oldest coordinate out,
        # append newpoint at the newest slot of each component's block.
        # Column layout per component block: [oldest ... newest],
        # so we shift left and write at the last column.
        current = current.copy()
        for c in range(n_vars):
            block_lo = c * embed
            current[block_lo : block_lo + embed - 1] = current[
                block_lo + 1 : block_lo + embed
            ]
            current[block_lo + embed - 1] = newpoint[c]

    # "ok"      → all n_steps written
    # "escaped" → step_done + 1 (the escaping point was written before break)
    # "failed"  → step_done     (no point computed for the failed step)
    if status == "ok":
        n_steps_done = n_steps
    elif status == "escaped":
        n_steps_done = step_done + 1
    else:  # "failed"
        n_steps_done = step_done

    out = out[:n_steps_done]

    # --- rescale back to original units -----------------------------
    for c in range(n_vars):
        minv, interval = rescale_info[c]
        out[:, c] = out[:, c] * interval + minv

    return {
        "trajectory": out,
        "n_steps_done": n_steps_done,
        "status": status,
        "final_epsilon": final_eps,
        "rescale_info": rescale_info,
    }
