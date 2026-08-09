import numpy as np
from scipy.integrate import solve_ivp


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


def lorenz_map(
    length=10000,
    x0=None,
    sigma=10.0,
    beta=8.0 / 3.0,
    rho=28.0,
    step=0.001,
    sample=0.03,
    discard=1000,
):
    """Generate time series using the Lorenz system.

    Generates time series using the Lorenz system.

    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the flow.
    sigma : float, optional (default = 10.0)
        Constant sigma of the Lorenz system.
    beta : float, optional (default = 8.0/3.0)
        Constant beta of the Lorenz system.
    rho : float, optional (default = 28.0)
        Constant rho of the Lorenz system.
    step : float, optional (default = 0.001)
        Approximate step size of integration.
    sample : int, optional (default = 0.03)
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

    def _lorenz(t, x):
        return [
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ]

    if x0 is None:
        x0 = (0.0, -0.01, 9.0) + 0.25 * (-1 + 2 * np.random.random(3))

    sample_int = int(sample / step)
    total_steps = sample_int * (length + discard)
    t_max = total_steps * step

    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, total_steps)

    sol = solve_ivp(fun=_lorenz, t_span=t_span, y0=x0, t_eval=t_eval, method="RK45")

    times = sol.t
    states = sol.y.T

    return (
        times[discard * sample_int :: sample_int],
        states[discard * sample_int :: sample_int],
    )
