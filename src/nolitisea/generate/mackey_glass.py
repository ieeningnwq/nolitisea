"""Generate time series using the Mackey-Glass equation."""
import numpy as np


def mackey_glass(length=10000, x0=None, a=0.2, b=0.1, c=10.0, tau=23.0,
                 n=1000, sample=0.46, discard=250):
    """Generate time series using the Mackey-Glass equation.

    Original Mackey-Glass delay differential equation:

    .. math::
        \\frac{dx(t)}{dt} = -b\\,x(t) + a \\frac{x(t-\\tau)}{1 + x(t-\\tau)^c}

    Discretization using the trapezoidal scheme (Grassberger & Procaccia, 1983).
    The delay interval :math:`\\tau` is divided into ``n`` sub-intervals,
    giving internal time step :math:`h = \\tau / n`.
    Recurrence relation:

    .. math::
        x_{i+1} = A \\, x_i
        + B \\left(
        \\frac{x_{i-n}}{1+x_{i-n}^c}
        + \\frac{x_{i-n+1}}{1+x_{i-n+1}^c}
        \\right)

    with coefficients

    .. math::
        A = \\frac{2n - b\\tau}{2n + b\\tau}, \\quad
        B = \\frac{a\\tau}{2n + b\\tau}.

    Parameters
    ----------
    length : int, optional (default = 10000)
        Number of output samples in final time series.
    x0 : array_like, optional (default = None)
        Initial condition array of length ``n`` for the discretized system.
    a : float, optional (default = 0.2)
        Positive constant a in Mackey-Glass equation.
    b : float, optional (default = 0.1)
        Decay constant b.
    c : float, optional (default = 10.0)
        Nonlinear exponent c.
    tau : float, optional (default = 23.0)
        Delay time constant.
    n : int, optional (default = 1000)
        Number of discrete sub-intervals over one delay interval :math:`\\tau`.
        Internal integration step :math:`h = \\tau / n`.
    sample : float, optional (default = 0.46)
        Physical time interval between output samples.
        Internally converted to integer step index for subsampling.
    discard : int, optional (default = 250)
        Number of full delay-blocks (n internal steps) to discard to wash out
        transients. Total discarded internal steps = ``n * discard``.

    Returns
    -------
    x : ndarray, shape (length,)
        One-dimensional Mackey-Glass scalar time series.
    """
    sample = int(n * sample / tau)
    grids = n * discard + sample * length
    x = np.empty(grids)

    if x0 is None:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                   x[i - n + 1] / (1 + x[i - n + 1] ** c))
    return x[n * discard::sample]
