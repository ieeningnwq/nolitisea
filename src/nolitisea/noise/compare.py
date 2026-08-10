"""Compare two data sets (TISEAN ``compare``)."""

import numpy as np


def compare(a, b):
    """Return summary error statistics between two series.

    Parameters
    ----------
    a : array_like
        Reference series.
    b : array_like
        Test series.

    Returns
    -------
    dict
        Dictionary with ``mean_error``, ``rms_error`` and ``max_error``.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: compare two data sets.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
