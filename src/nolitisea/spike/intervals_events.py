"""Event/interval conversion (TISEAN ``intervals`` / ``events``)."""

import numpy as np


def events_to_intervals(event_times):
    """Convert event times to inter-event intervals.

    Parameters
    ----------
    event_times : array_like
        Sorted event times.

    Returns
    -------
    numpy.ndarray
        Differences between consecutive events.
    """
    raise NotImplementedError


def intervals_to_events(intervals, t0=0.0):
    """Convert inter-event intervals to event times.

    Parameters
    ----------
    intervals : array_like
        Inter-event intervals.
    t0 : float, default 0.0
        Time of the first event.

    Returns
    -------
    numpy.ndarray
        Cumulative event times.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: event/interval conversion.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
