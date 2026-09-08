"""Generate a logistic-map time series."""

import numpy as np


def logistic(r=3.7, x0=0.3, discard=1000, n=10000):
    """Generate a logistic-map time series after discarding transients.

    The logistic map is defined as:

        x_{n+1} = r * x_n * (1 - x_n)

    Parameters
    ----------
    r : float, default 3.7
        Growth-rate parameter.  Chaos sets in around r ~ 3.5699;
        r = 3.7 is a commonly used chaotic value.
    x0 : float, default 0.3
        Initial condition.  Should lie in (0, 1) for the map to
        remain bounded.
    discard : int, default 1000
        Number of transient iterations to discard so that the
        trajectory settles onto the attractor.
    n : int, default 10000
        Number of data points to keep.

    Returns
    -------
    X : ndarray, shape (n,)
        Array containing the trajectory on the attractor.
    """
    x = x0

    # Discard transient iterations
    for _ in range(discard):
        x = r * x * (1.0 - x)

    # Store the attractor trajectory
    X = np.empty(n)
    for i in range(n):
        x = r * x * (1.0 - x)
        X[i] = x

    return X
