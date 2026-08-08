import numpy as np


def henon_map(a=1.4, b=0.3, x0=0.0, y0=0.0, transient=1000, n=10000):
    """
    Generate Hénon map time series after discarding transients.

    Parameters
    ----------
    a, b : float
        Parameters of the Hénon map.
    x0, y0 : float
        Initial conditions.
    transient : int
        Number of transient iterations to discard.
    n : int
        Number of data points to keep.

    Returns
    -------
    X, Y : ndarray
        Arrays containing the trajectory on the attractor.
    """
    x, y = x0, y0

    # Discard transient iterations
    for _ in range(transient):
        x, y = a - x**2 + b * y, x

    # Store the attractor trajectory
    X = np.zeros(n)
    Y = np.zeros(n)

    for i in range(n):
        x, y = a - x**2 + b * y, x
        X[i], Y[i] = x, y

    return X, Y
