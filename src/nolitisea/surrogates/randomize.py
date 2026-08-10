"""General constrained randomization by annealing (TISEAN ``randomize``)."""

import numpy as np


def randomize(series, cost_func, n_iter=1000, seed=None):
    """Generate surrogates by combinatorial minimisation of a cost.

    Parameters
    ----------
    series : array_like
        Input scalar series.
    cost_func : callable
        Function ``(series) -> float`` measuring the constraint violation.
    n_iter : int, default 1000
        Number of annealing iterations.
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    numpy.ndarray
        A surrogate series.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: constrained randomization.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
