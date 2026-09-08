"""Locally zeroth-order (constant) prediction."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.utils.rescale import rescale_data

__all__ = ["lzo_gm", "lzo_run", "lzo_test"]


# ---------------------------------------------------------------------------
# Shared helpers
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

    ``E`` is the delay embedding matrix ``(n_points, n_vars * embed)``
    with column order produced by
    :func:`nolitisea.core.embed.lag_block_delay_embed`.  Row ``r`` of
    ``E`` corresponds to the original time index
    ``t = r + valid_start``.
    """
    return lag_block_delay_embed(s2d, embed=embed, delay=delay), (embed - 1) * delay


def _scan_epsilons(eps0, eps1, eps_factor):
    """Yield geometrically spaced epsilon values up to ``eps1 * factor``.

    ``for epsilon = EPS0; epsilon < EPS1 * EPSF; epsilon *= EPSF``.
    """
    eps = float(eps0)
    while eps < eps1 * eps_factor:
        yield eps
        eps *= eps_factor


# ---------------------------------------------------------------------------
# lzo_gm — grid scan over fixed epsilon values (zeroth-order fit)
# ---------------------------------------------------------------------------


def lzo_gm(
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
    """Scan neighbourhood sizes and record the average local zeroth-order
    forecast error.

    For each candidate ``epsilon`` (geometric scan starting at
    ``eps0``, multiplying by ``eps_factor`` up to ``eps1 *
    eps_factor``), the function:

    1. Rescales each component to :math:`[0, 1]`.
    2. Builds the lag-block delay embedding.
    3. For each reference point, finds all ``epsilon``-neighbours in
       Chebyshev distance and excludes those inside the causality
       window.
    4. Predicts each component's value at ``i + step`` as the *mean*
       of the neighbours' values at that time (zeroth-order fit).
    5. Records the root-mean-square prediction error normalised by
       the sample standard deviation of the target component over
       all valid reference points.

    This is the zeroth-order analogue of :func:`lfo_ar`
    (local linear).

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
        Causality window half-width.  ``None`` defaults
        to ``step``.  Neighbour times in
        ``[i - causal + 1,
         i + causal + (embed - 1) * delay - 1]`` are excluded.
    eps0 : float or None, default None
        Starting neighbourhood size (relative units).  ``None`` uses
        ``1/1000``.
    eps1 : float or None, default None
        Final neighbourhood size (relative units).  ``None`` uses
        ``1.0`` (full data box).
    eps_factor : float, default 1.2
        Geometric multiplier between successive epsilon values.
    min_neighbors : int or None, default None
        Minimum number of neighbours (after causal exclusion) required
        to fit.  ``None`` uses ``2 * (n_vars * embed + 1)``.

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
        - ``"embedding_norm"`` : per-component ``(minv, interval)``
          rescale info.
        - ``"n_ref_points"`` : int, number of valid reference points.

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the requested
        embedding + forecast parameters.
    """
    n_times, n_vars, s2d = _resolve_inputs(series)

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

    # --- default epsilon bounds ----------------------------------------
    if eps0 is None:
        eps0 = 1.0 / 1000.0
    if eps1 is None:
        eps1 = 1.0

    # --- delay embedding -----------------------------------------------
    E, valid_start = _build_embedding(rescaled, embed, delay)
    ref_orig_times = np.arange(valid_start, n_times - step)
    n_ref_total = len(ref_orig_times)

    # --- KD-tree for Chebyshev neighbours -----------------------------
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
            # Drop points whose target overshoots the series
            keep = nbr_times <= n_times - step - 1
            nbr_embed_idx = nbr_embed_idx[keep]
            nbr_times = nbr_times[keep]
            # Exclude causal window
            lo = i_orig - causal + 1
            hi = i_orig + causal + (embed - 1) * delay - 1
            keep = (nbr_times < lo) | (nbr_times > hi)
            nbr_embed_idx = nbr_embed_idx[keep]
            nbr_times = nbr_times[keep]

            n_nbrs = len(nbr_times)
            if n_nbrs < min_neighbors:
                continue

            # --- zeroth-order fit: mean of neighbours' next values -----
            y_all = rescaled[nbr_times + step, :]
            y_pred = y_all.mean(axis=0)
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
        hrms = np.sqrt(
            np.maximum(0.0, (sum_hrms2 - pfound * hav**2) / (pfound - 1))
        )
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
# lzo_run — iterative zeroth-order trajectory
# ---------------------------------------------------------------------------


def lzo_run(
    series,
    embed=2,
    delay=1,
    n_steps=1000,
    eps0=None,
    eps_factor=1.2,
    min_neighbors=50,
    knn_mode=False,
    noise_pct=0.0,
    seed=None,
    escape_scale=2.0,
):
    """Iterate a local zeroth-order model forward.

    Builds a Chebyshev-neighbour cKDTree from the delay-embedded
    rescaled data, then iterates from the last embedding state.  At
    each step the function looks up ``min_neighbors`` neighbours via
    an adaptively expanding epsilon, fits a zeroth-order model (mean
    of neighbours' next values), extrapolates one step, and advances
    a rolling ring buffer.

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
    eps0 : float or None, default None
        Starting neighbour radius in *relative* units (fraction of the
        data box).  ``None`` uses ``1/1000``.
    eps_factor : float, default 1.2
        Geometric multiplier for the adaptive epsilon expansion loop.
    min_neighbors : int, default 50
        Minimum number of neighbours required to fit.
    knn_mode : bool, default False
        If ``True``, accumulate epsilon across iterations, sort
        neighbours by distance, and always use exactly
        ``min_neighbors`` nearest neighbours.  If ``False``, use all
        neighbours within the adaptive epsilon ball.
    noise_pct : float, default 0.0
        Standard deviation of Gaussian noise to add to each forecast,
        as a percentage of each component's variance.  Zero means no
        noise.
    seed : int or None, default None
        Random seed for reproducible noise injection.  ``None`` means
        unpredictable noise.
    escape_scale : float, default 2.0
        Escape threshold.  A forecast whose rescaled value lies outside
        ``[-escape_scale, 1 + escape_scale]`` terminates the iteration.

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
        - ``"epsilon_history"`` : array of length ``n_steps_done``
          with the epsilon used at each step.
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
    if eps_factor <= 1.0:
        raise ValueError(f"eps_factor must be > 1.0, got {eps_factor}")
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")
    if noise_pct < 0:
        raise ValueError(f"noise_pct must be >= 0, got {noise_pct}")
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

    # --- rescale each component ---------------------------------------
    rescale_info = {}
    rescaled = np.empty_like(s2d)
    variances = np.empty(n_vars)
    for c in range(n_vars):
        sr, minv, interval = rescale_data(s2d[:, c])
        rescale_info[c] = (minv, interval)
        rescaled[:, c] = sr
        # sample variance of rescaled data (≈ 1.0 since rescaled to [0,1])
        variances[c] = float(np.var(sr, ddof=1))

    # --- build embedding database -----------------------------------
    E = lag_block_delay_embed(rescaled, embed=embed, delay=delay)
    n_embed_points = E.shape[0]

    # Keep rows where the embedding has a valid one-step target
    last_valid_row = n_embed_points - 2
    E = E[: last_valid_row + 1]
    targets = rescaled[valid_start + 1 : valid_start + last_valid_row + 2, :]
    # E[r] corresponds to original time t = r + valid_start.
    # The one-step target for a point at time t is rescaled[t+1],
    # so targets[r] = rescaled[r + valid_start + 1].

    # --- KD-tree (Chebyshev distance, p=np.inf) ----------------------
    tree = cKDTree(E)

    # --- initial rolling state --------------------------------------
    initial_row = last_valid_row
    current = E[initial_row].copy()

    # --- epsilon setup --------------------------------------------
    eps_start = float(eps0) if eps0 is not None else 1.0 / 1000.0

    # knn_mode: accumulate epsilon across steps
    if knn_mode:
        epsilon_accum = eps_start / eps_factor
    else:
        epsilon_accum = eps_start

    # --- noise RNG ---------------------------------------------------
    rng = np.random.default_rng(seed)

    # --- iterate -----------------------------------------------------
    out = np.empty((n_steps, n_vars), dtype=np.float64)
    eps_history = np.empty(n_steps, dtype=np.float64)
    status = "ok"
    final_eps = eps_start
    step_done = 0

    for step_done in range(n_steps):
        # Adaptive epsilon expansion
        if knn_mode:
            # Start at the current accumulated value (pre-divided by
            # the factor), then expand until enough neighbours.
            epsilon = epsilon_accum
            nbr_indices = None
            while True:
                epsilon *= eps_factor
                nbr_indices = tree.query_ball_point(current, epsilon, p=np.inf)
                if len(nbr_indices) >= min_neighbors:
                    break
                if epsilon > 1.0 + 2.0 * escape_scale:
                    status = "failed"
                    break
            if status != "ok":
                break
            # Accumulate epsilon for the next step.
            epsilon_accum = epsilon
        else:
            epsilon = eps_start / eps_factor
            nbr_indices = None
            while True:
                epsilon *= eps_factor
                nbr_indices = tree.query_ball_point(current, epsilon, p=np.inf)
                if len(nbr_indices) >= min_neighbors:
                    break
                if epsilon > 1.0 + 2.0 * escape_scale:
                    status = "failed"
                    break
            if status != "ok":
                break

        nbr_indices = np.asarray(nbr_indices, dtype=np.intp)

        if knn_mode:
            # Sort neighbors by Chebyshev distance and take exactly
            # min_neighbors
            nbr_points = E[nbr_indices]
            dists = np.max(np.abs(nbr_points - current), axis=1)
            order = np.argsort(dists)
            nbr_indices = nbr_indices[order[:min_neighbors]]

        y = targets[nbr_indices]
        newpoint = y.mean(axis=0)

        # Add noise if requested (as % of component variance)
        if noise_pct > 0:
            noise_std = np.sqrt(variances) * (noise_pct / 100.0)
            newpoint += rng.normal(0.0, noise_std, size=n_vars)

        # Write output BEFORE the escape check: the new point is
        # recorded, then the region escape test may terminate.
        out[step_done] = newpoint
        final_eps = epsilon
        eps_history[step_done] = epsilon

        # Escape check on the rescaled values (data in [0, 1])
        if np.any(newpoint < -escape_scale) or np.any(newpoint > 1.0 + escape_scale):
            status = "escaped"
            break

        # Advance rolling state
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
    eps_history = eps_history[:n_steps_done]

    # --- rescale back to original units -----------------------------
    for c in range(n_vars):
        minv, interval = rescale_info[c]
        out[:, c] = out[:, c] * interval + minv

    return {
        "trajectory": out,
        "n_steps_done": n_steps_done,
        "status": status,
        "final_epsilon": final_eps,
        "epsilon_history": eps_history,
        "rescale_info": rescale_info,
    }


# ---------------------------------------------------------------------------
# lzo_test — adaptive zeroth-order forecast error test
# ---------------------------------------------------------------------------


def lzo_test(
    series,
    embed=2,
    delay=1,
    step=1,
    causal=None,
    eps0=None,
    eps_factor=1.2,
    min_neighbors=30,
    n_ref=None,
    refstep=1,
    verbose_single=False,
):
    """Estimate the average forecast error for a zeroth-order fit.

    Unlike :func:`lzo_gm`, which uses a fixed epsilon per scan, this
    routine adapts epsilon *per reference point*: for each point it
    starts at ``eps0`` and multiplies by ``eps_factor`` until at
    least ``min_neighbors`` (after causal exclusion) are found.

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
        Forecast horizon (multi-step forecasting supported).
    causal : int or None, default None
        Causality window half-width.  ``None`` defaults to ``step``.
    eps0 : float or None, default None
        Starting neighbourhood size (relative).  ``None`` uses
        ``1/1000``.
    eps_factor : float, default 1.2
        Geometric multiplier for adaptive epsilon expansion.
    min_neighbors : int, default 30
        Minimum number of neighbours (after causal exclusion) required.
    n_ref : int or None, default None
        Number of reference points to use.  ``None`` uses all valid
        reference points.
    refstep : int, default 1
        Temporal stride between reference points.
    verbose_single : bool, default False
        If ``True``, include per-reference-point forecast values in
        the output.

    Returns
    -------
    dict
        Keys:

        - ``"forecast_errors"`` : array of shape ``(step, n_vars)``
          with the relative forecast error (normalised by component
          std) at each forecast horizon.
        - ``"abs_errors"`` : array of shape ``(step, n_vars)``
          with the RMS absolute forecast error.
        - ``"n_ref_points"`` : int, number of reference points used.
        - ``"embedding_norm"`` : per-component ``(minv, interval)``.
        - ``"single_step_predictions"`` : optional array of shape
          ``(n_ref_points, n_vars)`` — only present when
          ``verbose_single=True``.

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the requested
        embedding + forecast parameters.
    """
    n_times, n_vars, s2d = _resolve_inputs(series)

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
    if refstep < 1:
        raise ValueError(f"refstep must be >= 1, got {refstep}")

    valid_start = (embed - 1) * delay
    if n_times <= 2 * step + causal + valid_start + min_neighbors:
        raise ValueError(
            f"series of length {n_times} is too short for embed={embed}, "
            f"delay={delay}, step={step}, causal={causal}"
        )

    # --- rescale each component ---------------------------------------
    rescale_info = {}
    rescaled = np.empty_like(s2d)
    variances = np.empty(n_vars)
    for c in range(n_vars):
        sr, minv, interval = rescale_data(s2d[:, c])
        rescale_info[c] = (minv, interval)
        rescaled[:, c] = sr
        variances[c] = float(np.var(sr, ddof=1))

    # --- default epsilon ---------------------------------------------
    if eps0 is None:
        eps0 = 1.0 / 1000.0

    # --- delay embedding -----------------------------------------------
    E, valid_start = _build_embedding(rescaled, embed, delay)

    # Reference points (with refstep subsampling).
    # clength already incorporates the - step adjustment, so the
    # arange stop must be clength (NOT clength - step).
    clength = (
        min(n_ref * refstep + step, n_times)
        if n_ref is not None
        else n_times
    )
    clength = min(clength, n_times - step)

    ref_orig_times = np.arange(valid_start, clength, refstep)
    n_ref_total = len(ref_orig_times)

    if n_ref_total <= 0:
        raise ValueError(
            f"no valid reference points for clength={clength}, "
            f"valid_start={valid_start}, refstep={refstep}"
        )

    # --- KD-tree ------------------------------------------------------
    tree = cKDTree(E)

    # --- adaptive epsilon per reference point -------------------------
    error_sum = np.zeros((step, n_vars))
    rms_sum = np.zeros(n_vars)  # E[y^2] for normalisation
    hav_sum = np.zeros(n_vars)  # E[y] for normalisation
    pfound = 0

    single_predictions = [] if verbose_single else None

    for hi_orig in ref_orig_times:
        # Map to embedding row index
        hi_row = hi_orig - valid_start

        # Adaptive epsilon expansion
        epsilon = eps0 / eps_factor
        found_ok = False
        nbr_times_final = None

        while True:
            epsilon *= eps_factor
            nbr_embed_idx = np.asarray(
                tree.query_ball_point(E[hi_row], epsilon, p=np.inf),
                dtype=np.intp,
            )
            nbr_times = nbr_embed_idx + valid_start
            # Drop points whose target at step overshoots
            keep = nbr_times <= n_times - step - 1
            nbr_embed_idx = nbr_embed_idx[keep]
            nbr_times = nbr_times[keep]
            # Exclude causal window
            lo = hi_orig - causal + 1
            hi = hi_orig + causal + (embed - 1) * delay - 1
            keep = (nbr_times < lo) | (nbr_times > hi)
            nbr_embed_idx = nbr_embed_idx[keep]
            nbr_times = nbr_times[keep]

            if len(nbr_times) >= min_neighbors:
                found_ok = True
                nbr_times_final = nbr_times
                break

            if epsilon > 2.0:  # data in [0,1], 2.0 is whole box
                break

        if not found_ok:
            continue

        pfound += 1

        # Compute multi-step zeroth-order forecasts
        for istep in range(1, step + 1):
            h = istep - 1
            y_all = rescaled[nbr_times_final + istep, :]  # pyright: ignore[reportOptionalOperand]
            y_pred = y_all.mean(axis=0)
            y_true = rescaled[hi_orig + istep, :]
            error_sum[h] += (y_pred - y_true) ** 2

        # Accumulate for normalisation (use the step=1 target).
        y_ref = rescaled[hi_orig + 1, :]
        rms_sum += y_ref**2
        hav_sum += y_ref

        if verbose_single:
            h1_all = rescaled[nbr_times_final + 1, :]  # pyright: ignore[reportOptionalOperand]
            h1_pred = h1_all.mean(axis=0)
            single_predictions.append(h1_pred)  # pyright: ignore[reportOptionalMemberAccess]

    if pfound == 0:
        raise RuntimeError(
            "no reference points could find enough neighbours; "
            "try increasing eps0 or reducing min_neighbors"
        )

    # Compute forecast errors relative to target component std
    hav = hav_sum / pfound
    # Sample standard deviation of the step-1 targets.
    hrms = np.sqrt(
        np.maximum(0.0, (rms_sum - pfound * hav**2) / (pfound - 1))
    )

    forecast_errors = np.zeros((step, n_vars))
    abs_errors = np.zeros((step, n_vars))
    for h in range(step):
        abs_err_h = np.sqrt(error_sum[h] / pfound)
        abs_errors[h] = abs_err_h
        forecast_errors[h] = abs_err_h / hrms

    result = {
        "forecast_errors": forecast_errors,
        "abs_errors": abs_errors,
        "n_ref_points": pfound,
        "embedding_norm": rescale_info,
    }
    if verbose_single:
        result["single_step_predictions"] = (
            np.asarray(single_predictions) if single_predictions else np.empty((0, n_vars))
        )

    return result
