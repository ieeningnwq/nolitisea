"""Space-time separation plot (TISEAN ``stp``)."""

import numpy as np


def space_time_separation(series, dim, delay, max_time, n_eps=100):
    """Compute the cumulative distance distribution for time-shifted pairs.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.
    max_time : int
        Maximum time shift.
    n_eps : int, default 100
        Number of distance thresholds.

    Returns
    -------
    tuple
        ``(times, eps_grid, fractions)``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: space-time separation plot.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
