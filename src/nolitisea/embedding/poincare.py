"""Poincare sections."""

from __future__ import annotations

from functools import partial

import numpy as np

from nolitisea.utils.parallel import parallel_map


def _raw_crossings_chunk(args, dim, delay, comp_to_cut, threshold,
                         direction, epsilon, series):
    """Find every raw crossing event within a t-index chunk (vectorized).

    This worker performs the purely-local phase of the algorithm: for every
    ``t`` in ``chunk`` it inspects the monitored component at ``t`` and
    ``t+1``, then -- if a crossing occurs -- records the interpolated
    crossing time together with the interpolated coordinates of every
    remaining component.  No inter-``t`` state is involved, so chunks can
    be processed independently and in any order (the caller concatenates
    them in ascending-t order).

    Implementation notes
    --------------------
    Hot per-t logic is expressed with NumPy vector operations so that the
    thread backend actually parallelises work (a pure-Python inner loop
    would keep the GIL and offer no speedup).  The component interpolation
    remains a small loop over components, which is typically ``dim`` < 10
    and negligible compared with scanning a large ``chunk`` length.
    """
    chunk_start, chunk_stop = args
    n = chunk_stop - chunk_start
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, dim - 1), dtype=np.float64)
    t_arr = np.arange(chunk_start, chunk_stop, dtype=np.int64)

    # Current/next values of the monitored component at every t in the chunk.
    idx_cur = t_arr + comp_to_cut * delay
    val_cur = series[idx_cur]
    val_nxt = series[idx_cur + 1]

    # Crossing mask (direction-sensitive).
    if direction == 0:
        mask = (val_cur < threshold) & (val_nxt >= threshold)
    else:
        mask = (val_cur > threshold) & (val_nxt <= threshold)
    if not np.any(mask):
        return np.empty(0, dtype=np.float64), np.empty((0, dim - 1), dtype=np.float64)

    t_hit = t_arr[mask]
    vc_hit = val_cur[mask]
    vn_hit = val_nxt[mask]

    # Interpolation weights ``delta``.
    denom = vc_hit - vn_hit
    safe_denom = np.where(np.abs(denom) < epsilon, 1.0, denom)
    delta = np.where(np.abs(denom) < epsilon, 0.0, (vc_hit - threshold) / safe_denom)
    crossing_times = t_hit.astype(np.float64) + delta

    # Interpolate each remaining component independently.
    other_j = [j for j in range(dim) if j != comp_to_cut]
    coords = np.empty((len(t_hit), len(other_j)), dtype=np.float64)
    for out_col, j in enumerate(other_j):
        jd = t_hit + j * delay
        diff = series[jd + 1] - series[jd]
        coords[:, out_col] = series[jd] + delta * diff
    return crossing_times, coords

  
def poincare_section(
    series,
    dim=2,
    delay=1,
    comp_to_cut=-1,
    threshold=None,
    direction=0,
    min_return_time=0.0,
    epsilon=1e-12,
    n_jobs=None,
    backend="thread",
):
    """
    Computes the Poincare section from a 1D time series using delay coordinate embedding.

    Parameters:
    -----------
    series : array_like
        The 1D input time series data.
    dim : int, optional
        The embedding dimension (default is 2).
    delay : int, optional
        The time delay for embedding (default is 1).
    comp_to_cut : int, optional
        The component index (0 to dim-1) to be used as the cutting plane.
        Default is -1 (the last component).
    threshold : float, optional
        The value at which the section cuts the component.
        If None, the mean of the series is used.
    direction : int, optional
        The crossing direction. 0 for bottom-up crossing, 1 for top-down crossing.
    min_return_time : float, optional
        [NOISE RESISTANCE] The minimum time steps required between two consecutive crossings.
        Filters out high-frequency noise/jitter around the threshold.
    epsilon : float, optional
        [NUMERICAL STABILITY] A small threshold to prevent division-by-zero during
        linear interpolation if consecutive points are nearly identical.
    n_jobs : int, optional (default = None)
        Workers for the raw-crossing pass.  ``None``/``1`` runs the original
        sequential loop (bit-identical to previous behavior); ``-1`` uses all
        CPUs (see ``nolitisea.utils.parallel.parallel_map``).
    backend : {"thread", "process"}, optional (default = "thread")
        Executor backend for the raw-crossing pass.  ``"thread"`` is preferred
        since the worker operates on NumPy views and releases the GIL.

    Returns:
    --------
    poincare_pts : ndarray
        A 2D array where each row represents a point on the Poincare section.
        The dimensionality of the points is (dim - 1).
    return_times : ndarray
        A 1D array of return times (time elapsed since the previous crossing).
    """
    series = np.asarray(series, dtype=float)
    N = len(series)

    if threshold is None:
        threshold = np.mean(series)

    if comp_to_cut == -1:
        comp_to_cut = dim - 1

    if not (0 <= comp_to_cut < dim):
        raise ValueError("comp_to_cut must be in the range [0, dim-1]")

    max_t = N - 1 - (dim - 1) * delay

    # ---- Sequential fast path: original loop, zero overhead, bit-identical ----
    if n_jobs is None or n_jobs == 1 or max_t <= 1:
        poincare_pts = []
        return_times = []
        last_time = -1.0

        for t in range(max_t):
            idx_current = t + comp_to_cut * delay
            val_current = series[idx_current]
            val_next = series[idx_current + 1]

            is_crossing = False
            if direction == 0:
                if val_current < threshold and val_next >= threshold:
                    is_crossing = True
            else:
                if val_current > threshold and val_next <= threshold:
                    is_crossing = True

            if not is_crossing:
                continue

            denom = val_current - val_next
            if abs(denom) < epsilon:
                delta = 0.0
            else:
                delta = (val_current - threshold) / denom

            crossing_time = float(t) + delta

            if last_time > 0.0:
                ret_time = crossing_time - last_time
                if ret_time < min_return_time:
                    last_time = crossing_time
                    continue

                pt = []
                for j in range(dim):
                    if j == comp_to_cut:
                        continue
                    jd = t + j * delay
                    xcut = series[jd] + delta * (series[jd + 1] - series[jd])
                    pt.append(xcut)

                poincare_pts.append(pt)
                return_times.append(ret_time)

            last_time = crossing_time

        return np.array(poincare_pts), np.array(return_times)

    # ---- Two-phase parallel path --------------------------------------------
    # Phase 1 (parallel): split the t-range into chunks and, for each chunk,
    #   find every raw crossing together with its interpolated coordinates.
    # Phase 2 (sequential): walk the (typically much smaller) list of raw
    #   crossings and apply the noise-resistance / return-time debouncing.
    # Chunking is done by splitting t_range evenly; cKDTree is not involved,
    # so thread backends still benefit from vectorised NumPy work that
    # releases the GIL inside its C loops.
    import os
    if n_jobs == -1:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = max(1, int(n_jobs))
    n_workers = min(n_workers, max(1, max_t))
    base = max_t // n_workers
    rem = max_t % n_workers
    chunks = []
    start = 0
    for w in range(n_workers):
        size = base + (1 if w < rem else 0)
        if size == 0:
            break
        chunks.append((start, start + size))
        start += size

    worker = partial(
        _raw_crossings_chunk,
        dim=dim,
        delay=delay,
        comp_to_cut=comp_to_cut,
        threshold=threshold,
        direction=direction,
        epsilon=epsilon,
        series=series,
    )
    chunk_results = parallel_map(worker, chunks, n_jobs=n_jobs, backend=backend)

    times_list, coords_list = zip(*chunk_results) if chunk_results else ([], [])
    all_times = np.concatenate(times_list) if times_list else np.empty(0)
    if coords_list:
        all_coords = np.vstack(coords_list)
    else:
        all_coords = np.empty((0, dim - 1))

    # Phase 2: debouncing + return-time assignment.
    # Matches the sequential algorithm exactly:
    #   * ``last_time`` is updated for EVERY raw crossing (including rejected
    #     ones), i.e. ``continue``-after-reject in the original code still
    #     ran ``last_time = crossing_time`` at the bottom of the ``if`` block
    #     once the outer ``if is_crossing`` body had executed.
    #   * The return_time of the first raw crossing is never emitted
    #     (``last_time <= 0`` branch in the sequential version).
    out_pts = []
    out_rets = []
    last_time = -1.0
    for i in range(len(all_times)):
        crossing_time = float(all_times[i])
        if last_time > 0.0:
            ret_time = crossing_time - last_time
            if ret_time < min_return_time:
                last_time = crossing_time
                continue
            out_pts.append(all_coords[i])
            out_rets.append(ret_time)
        last_time = crossing_time

    return np.array(out_pts), np.array(out_rets)
