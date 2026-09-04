"""Generate an Ikeda-map time series."""

import numpy as np


def ikeda(length=10000, x0=None, alpha=6.0, beta=0.4, gamma=1.0, mu=0.9,
          discard=500):
    """Generate time series from the Ikeda map.

    The Ikeda map is a discrete-time, 2-D complex map:
    .. math::
        z_{n+1} = \\gamma + \\mu \\, z_n \\exp\\left(i\\left(\\beta - \\frac{\\alpha}{1+|z_n|^2}\\right)\\right)

    Decompose complex variable :math:`z_n = x_n + i y_n` into real-valued update:

    .. math::
        \\phi_n = \\beta - \\frac{\\alpha}{1 + x_n^2 + y_n^2}

    .. math::
        \\begin{cases}
        x_{n+1} = \\gamma + \\mu\\left(x_n \\cos\\phi_n - y_n \\sin\\phi_n\\right)\\\\
        y_{n+1} = \\mu\\left(x_n \\sin\\phi_n + y_n \\cos\\phi_n\\right)
        \\end{cases}

    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the map, shape (2,).
    alpha : float, optional (default = 6.0)
        Constant alpha in the Ikeda map.
    beta : float, optional (default = 0.4)
        Constant beta in the Ikeda map.
    gamma : float, optional (default = 1.0)
        Constant gamma in the Ikeda map.
    mu : float, optional (default = 0.9)
        Constant mu in the Ikeda map.
    discard : int, optional (default = 500)
        Number of initial transient steps to discard to remove orbit transients.

    Returns
    -------
    x : ndarray, shape (length, 2)
        Array containing points in phase space, each row is ``(x_n, y_n)``.
    """
    x = np.empty((length + discard, 2))

    if x0 is None:
        x[0] = 0.1 * (-1 + 2 * np.random.random(2))
    else:
        x[0] = x0

    for i in range(1, length + discard):
        phi = beta - alpha / (1 + x[i - 1][0] ** 2 + x[i - 1][1] ** 2)
        x[i] = (gamma + mu * (x[i - 1][0] * np.cos(phi) - x[i - 1][1] *
                np.sin(phi)),
                mu * (x[i - 1][0] * np.sin(phi) + x[i - 1][1] * np.cos(phi)))

    return x[discard:]
