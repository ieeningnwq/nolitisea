"""Generate time series using the Rössler oscillator."""

import numpy as np
from scipy.integrate import odeint


def roessler(length=10000, x0=None, a=0.2, b=0.2, c=5.7, step=0.001,
             sample=0.1, discard=1000):
    """Generate time series using the Rössler oscillator.

    Generates time series using the Rössler oscillator, a three
    coupled ordinary differential equations:

        dx/dt = -(y + z)
        dy/dt = x + a * y
        dz/dt = b + z * (x - c)

    where `x`, `y`, `z` are the state variables and `a`, `b`, `c`
    are the classical dimensionless parameters (default
    `a = b = 0.2`, `c = 5.7`).

    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the flow.
    a : float, optional (default = 0.2)
        Constant a in the Röessler oscillator.
    b : float, optional (default = 0.2)
        Constant b in the Röessler oscillator.
    c : float, optional (default = 5.7)
        Constant c in the Röessler oscillator.
    step : float, optional (default = 0.001)
        Approximate step size of integration.
    sample : int, optional (default = 0.1)
        Sampling step of the time series.
    discard : int, optional (default = 1000)
        Number of samples to discard in order to eliminate transients.

    Returns
    -------
    t : array
        The time values at which the points have been sampled.
    x : ndarray, shape (length, 3)
        Array containing points in phase space.
    """
    def _roessler(x, t):
        return [-(x[1] + x[2]), x[0] + a * x[1], b + x[2] * (x[0] - c)]

    sample = int(sample / step)
    t = np.linspace(0, (sample * (length + discard)) * step,
                    sample * (length + discard))

    if not x0:
        x0 = (-9.0, 0.0, 0.0) + 0.25 * (-1 + 2 * np.random.random(3))

    return (t[discard * sample::sample],
            odeint(_roessler, x0, t)[discard * sample::sample])
