"""Poincare sections (TISEAN ``poincare``)."""

import numpy as np


def poincare_section(
    series,
    dim=2,
    delay=1,
    comp_to_cut=-1,
    threshold=None,
    direction=0,
    min_return_time=0.0,
    epsilon=1e-12,
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

    # Handle default parameters
    if threshold is None:
        threshold = np.mean(series)

    if comp_to_cut == -1:
        comp_to_cut = dim - 1

    if not (0 <= comp_to_cut < dim):
        raise ValueError("comp_to_cut must be in the range [0, dim-1]")

    poincare_pts = []
    return_times = []
    last_time = -1.0

    # max_t ensures we have enough data points ahead to form a complete embedded vector
    # for both the current step (t) and the next step (t+1)
    max_t = N - 1 - (dim - 1) * delay

    for t in range(max_t):
        # Index of the component we are monitoring for the cut
        idx_current = t + comp_to_cut * delay
        val_current = series[idx_current]
        val_next = series[idx_current + 1]

        # 1. Check if the trajectory crosses the Poincare section
        is_crossing = False
        if direction == 0:
            if val_current < threshold and val_next >= threshold:
                is_crossing = True
        else:
            if val_current > threshold and val_next <= threshold:
                is_crossing = True

        if is_crossing:
            denom = val_current - val_next

            # 2. IMPROVEMENT: Numerical Stability (Prevent division by zero)
            if abs(denom) < epsilon:
                # If the denominator is extremely small, the current point is practically
                # exactly on the threshold. Delta becomes 0.
                delta = 0.0
            else:
                # Linear interpolation weight
                delta = (val_current - threshold) / denom

            crossing_time = float(t) + delta

            if last_time > 0.0:
                ret_time = crossing_time - last_time

                # 3. IMPROVEMENT: Noise Resistance (Debouncing)
                # Ignore this crossing if it happened too soon after the previous one
                if ret_time < min_return_time:
                    continue

                # 4. Calculate the exact coordinates of the remaining dimensions via interpolation
                pt = []
                for j in range(dim):
                    if j == comp_to_cut:
                        continue  # Skip the cut plane component

                    jd = t + j * delay
                    xcut = series[jd] + delta * (series[jd + 1] - series[jd])
                    pt.append(xcut)

                poincare_pts.append(pt)
                return_times.append(ret_time)

            # Update last_time for the next valid crossing
            last_time = crossing_time

    return np.array(poincare_pts), np.array(return_times)
