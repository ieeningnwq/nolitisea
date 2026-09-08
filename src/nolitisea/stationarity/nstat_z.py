"""Nonstationarity test via cross-prediction."""

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.utils.rescale import rescale_data

__all__ = ["nstat_z"]

# Safety cap on the epsilon ladder: the series is rescaled to [0, 1],
# so the maximum Chebyshev distance is 1.  Once eps exceeds 2.0 every
# point is a neighbour of every other point; remaining un-done
# references simply lack enough non-excluded neighbours.
_EPS_CAP = 2.0


def nstat_z(
    series,
    dim=3,
    delay=1,
    n_pieces=10,
    step=1,
    min_neighbors=30,
    eps0=None,
    eps_factor=1.2,
    n_refs=None,
    causal=None,
):
    """Cross-prediction nonstationarity test.

    Parameters
    ----------
    series : array_like
        Input scalar series (1-D).
    dim : int, default 3
        Embedding dimension; must be >= 1.
    delay : int, default 1
        Time delay; must be >= 1.
    n_pieces : int, default 10
        Number of contiguous segments; must be >= 1.
    step : int, default 1
        Forecast horizon in samples; must be >= 1.
    min_neighbors : int, default 30
        Minimum number of neighbours required for a reference point
        to be considered done; must be >= 1.
    eps0 : float or None
        Initial neighbourhood radius in the units of the input data.
        ``None`` (default) uses the data interval divided by 1000.
    eps_factor : float, default 1.2
        Multiplicative growth factor of the epsilon ladder; must be > 1.0.
    n_refs : int or None
        Number of reference points per segment.  ``None`` (default)
        uses ``clength - step``.  When set, the value is capped at
        ``clength - step - pstart``.
    causal : int or None
        Half-width of the temporal exclusion (Theiler) window.
        ``None`` (default) uses ``step``.

    Returns
    -------
    dict
        ``"matrix"`` : ``(n_pieces, n_pieces)`` array of normalised
        RMSE.  ``matrix[first, second]`` is the zeroth-order forecast
        error when predicting segment ``second`` from segment
        ``first``, normalised by the standard deviation of segment
        ``second``.
        ``"pieces"`` : the number of segments.
        ``"clength"`` : the length of each segment.
        ``"rms"`` : per-segment standard deviation (Bessel's
        correction), shape ``(n_pieces,)``.

    Raises
    ------
    ValueError
        For invalid parameters or a series too short for the
        requested ``dim``/``delay``/``n_pieces``/``step``.
    RuntimeError
        If the series is constant (zero value range).

    Notes
    -----
    Uses a single :class:`cKDTree` built once per source
    segment.  The epsilon ladder grows from ``eps0 / eps_factor``
    (so the first iteration uses ``eps0``).  The
    forecast for reference point ``i`` is the mean of
    ``series1[nb + step]`` over all non-excluded neighbours found
    within the current epsilon; the error contribution is
    ``(forecast - series2[i + step]) ** 2``.  The output is
    ``sqrt(sum_error / center) / std[second]``.

    On continuous data the cKDTree closed-ball (``<= eps``) coincides
    with a strict comparison (``< eps``); on quantised data the
    boundary pairs may differ, consistent with the convention used
    throughout this package.

    References
    ----------
    .. [1] Hegger, R., Kantz, H., & Schreiber, T. (1999). Practical
           implementation of nonlinear time series methods: The TISEAN
           package.  *Chaos*, 9(2), 413-435.
    .. [2] Kantz, H., & Schreiber, T. (1997, 2004). *Nonlinear Time
           Series Analysis*.  Cambridge University Press.
    """
    y = np.asarray(series, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("series must be a 1-D array")
    n = y.size

    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if n_pieces < 1:
        raise ValueError(f"n_pieces must be >= 1, got {n_pieces}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")
    if eps_factor <= 1.0:
        raise ValueError(f"eps_factor must be > 1.0, got {eps_factor}")

    pstart = (dim - 1) * delay

    # Global affine rescale to [0, 1].
    scaled, _minv, interval = rescale_data(y)

    # Epsilon in rescaled units.
    if eps0 is None:
        eps0_rescaled = 1.0e-3
    else:
        eps0_rescaled = abs(float(eps0)) / interval

    # Segment length (integer division).
    if n - pstart < n_pieces:
        raise ValueError(
            f"series too short ({n} points) for {n_pieces} pieces with "
            f"embedding offset {pstart}"
        )
    clength = (n - pstart) // n_pieces
    if clength - pstart - step < min_neighbors:
        raise ValueError(
            f"too many pieces ({n_pieces}) for series length {n}: "
            f"segment length {clength} leaves only "
            f"{clength - pstart - step} source embedding points, "
            f"need at least {min_neighbors}"
        )

    # Per-segment standard deviation (Bessel's correction) for
    # normalisation.  Computed on the rescaled data.
    rms = np.empty(n_pieces)
    for i in range(n_pieces):
        seg = scaled[i * clength : (i + 1) * clength]
        rms[i] = np.std(seg, ddof=1)

    # Number of reference points per segment.
    if n_refs is None:
        center = clength - step
    else:
        center = min(int(n_refs), clength - step - pstart)
    if center < 1:
        raise ValueError(f"not enough reference points: center = {center}, need >= 1")

    # Causal (exclusion) window half-width.
    if causal is None:
        causal = step
    else:
        causal = int(causal)

    error_matrix = np.full((n_pieces, n_pieces), np.nan)

    for first in range(n_pieces):
        src_start = first * clength
        # Source embedding: covers segment-base indices
        # [pstart, clength - step - 1].
        s1_source = scaled[src_start : src_start + clength - step]
        E1 = lag_block_delay_embed(s1_source, dim, delay)
        tree = cKDTree(E1)

        for second in range(n_pieces):
            tgt_start = second * clength
            # Extended target: the default center = clength - step
            # accesses up to segment index pstart + clength - step - 1
            # for the embedding and pstart + clength - 1 for the
            # forecast target, i.e. pstart points beyond the segment
            # boundary (into the next segment).  The extended slice
            # covers this.
            s2_ext = scaled[tgt_start : tgt_start + clength + pstart]
            E2 = lag_block_delay_embed(s2_ext, dim, delay)
            query_pts = E2[:center]

            done = np.zeros(center, dtype=bool)
            errors = np.zeros(center)
            eps = eps0_rescaled / eps_factor

            while not done.all():
                eps *= eps_factor
                not_done = np.where(~done)[0]
                if not_done.size == 0:
                    break
                nb_lists = tree.query_ball_point(query_pts[not_done], eps, p=np.inf)

                for idx, r in enumerate(not_done):
                    if done[r]:
                        continue
                    i = r + pstart  # segment-base index in target
                    nb = nb_lists[idx]
                    if len(nb) == 0:
                        continue
                    nb = np.asarray(nb, dtype=np.intp)
                    # Tree rows -> source segment-base indices.
                    nb_seg_base = nb + pstart
                    # Exclusion: remove [i - causal + 1,
                    # i + causal + pstart - 1].  When the lower
                    # bound is negative, no exclusion
                    # is applied.
                    ex_lo = i - causal + 1
                    ex_hi = i + causal + pstart - 1
                    if ex_lo < 0:
                        nb_kept = nb_seg_base
                    else:
                        mask = (nb_seg_base < ex_lo) | (nb_seg_base > ex_hi)
                        nb_kept = nb_seg_base[mask]

                    if nb_kept.size >= min_neighbors:
                        # Zeroth-order forecast: mean of
                        # s1[nb + step] over all kept neighbours.
                        forecast = scaled[src_start + nb_kept + step].mean()
                        actual = s2_ext[i + step]
                        errors[r] = (forecast - actual) ** 2
                        done[r] = True

                if eps > _EPS_CAP:
                    break

            # Normalised RMSE.
            n_done = int(done.sum())
            if n_done > 0:
                total_error = 0.0
                for r in range(center):
                    if done[r]:
                        total_error += errors[r]
                rms_val = rms[second]
                if rms_val > 0.0:
                    error_matrix[first, second] = (
                        np.sqrt(total_error / center) / rms_val
                    )
                else:
                    error_matrix[first, second] = np.sqrt(total_error / center)

    return {
        "matrix": error_matrix,
        "pieces": n_pieces,
        "clength": clength,
        "rms": rms,
    }
