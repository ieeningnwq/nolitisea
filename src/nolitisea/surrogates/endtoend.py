"""End-to-end mismatch measure (TISEAN ``endtoend``)."""

import numpy as np


def endtoend(series, dim, delay):
    """Quantify the mismatch at the wrap-around of the series.

    Used to choose a stationary sub-sequence.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Time delay.

    Returns
    -------
    float
        End-to-end mismatch value.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: end-to-end mismatch.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
