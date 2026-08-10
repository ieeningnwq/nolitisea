"""Generate a Henon-map time series."""

import numpy as np


def henon(a=1.4, b=0.3, x0=0.0, y0=0.0, discard=1000, n=10000):
    """
    Generate Hénon map time series after discarding transients.
    The Hénon map is defined as:

        x_{n+1} = 1 - a * x_n^2 + b * y_n
        y_{n+1} = x_n

    Parameters
    ----------
    a, b : float
        Parameters of the Hénon map.
    x0, y0 : float
        Initial conditions.
    discard : int
        Number of transient iterations to discard.
    n : int
        Number of data points to keep.

    Returns
    -------
    X, Y : ndarray
        Arrays containing the trajectory on the attractor.
    """
    x, y = x0, y0

    # Discard discard iterations
    for _ in range(discard):
        x, y = a - x**2 + b * y, x

    # Store the attractor trajectory
    X = np.zeros(n)
    Y = np.zeros(n)

    for i in range(n):
        x, y = a - x**2 + b * y, x
        X[i], Y[i] = x, y

    return X, Y

