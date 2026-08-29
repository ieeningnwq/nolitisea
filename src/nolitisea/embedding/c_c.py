import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial import cKDTree


def delay_embedding(ts: np.ndarray, emb: int, delay: int) -> np.ndarray:
    """Create a delay-coordinate embedding matrix.

    Parameters
    ----------
    ts : np.ndarray
        1-D time series.
    emb : int
        Embedding dimension.
    delay : int
        Time delay (lag) used in the reconstruction.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (n_points, emb).
    """
    n_points = len(ts) - (emb - 1) * delay
    if n_points <= 0:
        return np.empty((0, emb))
    return np.column_stack(
        [ts[i * delay : n_points + i * delay] for i in range(emb)]
    )


def correlation_integral_1d(ts: np.ndarray, eps: float) -> float:
    """Calculate the 1-D correlation integral of a time series.

    C_1(eps) = (# unordered pairs with Chebyshev distance <= eps)
             / (total unordered pairs)

    Parameters
    ----------
    ts : np.ndarray
        1-D time series.
    eps : float
        Distance radius.

    Returns
    -------
    float
        The 1-D correlation integral value.
    """
    n_points = len(ts)
    if n_points <= 1:
        return 0.0

    ts_2d = ts.reshape(-1, 1)
    tree = cKDTree(ts_2d)
    raw_counts = tree.count_neighbors(tree, eps, p=np.inf)

    # count_neighbors counts ordered pairs including self-pairs (i, i).
    # Subtract self-pairs and divide by 2 to obtain unordered i < j pairs.
    counts = (raw_counts - n_points) // 2
    valid_total = (n_points * (n_points - 1)) // 2

    return counts / valid_total if valid_total > 0 else 0.0


def correlation_integral_md(ts: np.ndarray, m: int, t: int, eps: float) -> float:
    """Calculate the m-dimensional correlation integral for delay t.

    Parameters
    ----------
    ts : np.ndarray
        1-D time series.
    m : int
        Embedding dimension.
    t : int
        Delay used inside the reconstruction.
    eps : float
        Distance radius.

    Returns
    -------
    float
        The m-dimensional correlation integral value.
    """
    emb_m = delay_embedding(ts, m, t)
    n_points = len(emb_m)
    if n_points <= 1:
        return 0.0

    tree = cKDTree(emb_m)
    raw_counts = tree.count_neighbors(tree, eps, p=np.inf)

    counts = (raw_counts - n_points) // 2
    valid_total = (n_points * (n_points - 1)) // 2

    return counts / valid_total if valid_total > 0 else 0.0


def cc_method(ts: np.ndarray, max_t: int = 200) -> dict:
    """Estimate the time delay (tau) and embedding dimension (m) via C-C.

    References
    ----------
    .. [1] Kim, H. S., Eykholt, R., & Salas, J. D. (1999).
           Nonlinear dynamics, delay times, and embedding windows.
           Physica D: Nonlinear Phenomena, 127(1-2), 48-60.

    Parameters
    ----------
    ts : np.ndarray
        1-D scalar time series.
    max_t : int
        Maximum candidate time delay to evaluate.

    Returns
    -------
    dict
        - 'tau'     : optimal time delay (first local minimum of S_mean).
        - 't_w'     : embedding window (global minimum of S_cor).
        - 'm'       : embedding dimension = round(t_w / tau) + 1.
        - 'S_mean'  : S_mean(t) statistic array (length max_t).
        - 'delta_S' : delta_S(t) statistic array (length max_t).
        - 'S_cor'   : S_cor(t) statistic array (length max_t).
    """
    ts = np.ascontiguousarray(ts, dtype=np.float64)
    std_dev = np.std(ts)
    if std_dev == 0.0:
        raise ValueError("Time series has zero variance.")

    m_values = [2, 3, 4, 5]
    r_values = [0.5 * std_dev, 1.0 * std_dev, 1.5 * std_dev, 2.0 * std_dev]

    t_range = np.arange(1, max_t + 1)
    S_mean = np.zeros(max_t)
    delta_S = np.zeros(max_t)
    S_cor = np.zeros(max_t)

    for t_idx, t in enumerate(t_range):
        # S_mr[a, b] = S(m_a, r_b, t), averaged over the t sub-series.
        S_mr = np.zeros((len(m_values), len(r_values)))

        for s in range(t):
            # The s-th disjoint sub-series: x_s, x_{s+t}, x_{s+2t}, ...
            sub = ts[s::t]
            if len(sub) <= max(m_values):
                continue

            # Reconstruct *inside the sub-series* with delay = 1, because the
            # sub-series elements are already spaced by t in the original data.
            for a, m in enumerate(m_values):
                for b, r in enumerate(r_values):
                    C_m = correlation_integral_md(sub, m, 1, r)
                    C_1 = correlation_integral_1d(sub, r)
                    S_mr[a, b] += (C_m - C_1 ** m)

        S_mr /= max(t, 1)

        # S_mean(t): average of S(m, r, t) over all m and all r.
        S_mean[t_idx] = S_mr.mean()
        # delta_S(t): for each m take max - min over r, then average over m.
        delta_S[t_idx] = (S_mr.max(axis=1) - S_mr.min(axis=1)).mean()
        # S_cor(t) = delta_S(t) + |S_mean(t)|.
        S_cor[t_idx] = delta_S[t_idx] + np.abs(S_mean[t_idx])

    # --- Parameter extraction ---
    # Time delay tau: first local minimum of delta_S(t).  This follows the
    # widely-reproduced C-C implementation (e.g. the standard Matlab/Python
    # C-C codes), which select the delay from delta_S rather than from
    # S_mean.  (The original paper's wording uses S_mean, but the consensus
    # implementation uses delta_S.)
    local_minima = argrelextrema(delta_S, np.less)[0]
    if len(local_minima) > 0:
        tau = int(t_range[local_minima[0]])
    else:
        # Fallback: first zero crossing of S_mean(t).
        zero_crossings = np.where(np.diff(np.sign(S_mean)))[0]
        tau = int(t_range[zero_crossings[0]]) if len(zero_crossings) > 0 else 1

    # Embedding window t_w: global minimum of S_cor(t).
    t_w = int(t_range[np.argmin(S_cor)])

    # Embedding dimension m = t_w / tau + 1.
    m_opt = int(np.round(t_w / tau)) + 1
    m_opt = max(m_opt, 2)

    return {
        "tau": tau,
        "t_w": t_w,
        "m": m_opt,
        "S_mean": S_mean.tolist(),
        "delta_S": delta_S.tolist(),
        "S_cor": S_cor.tolist(),
    }


if __name__ == "__main__":
    # Quick self-test on the Lorenz x-component and on white noise.
    def _lorenz_x(n=4000, dt=0.01, s=10.0, r=28.0, b=8.0 / 3.0):
        state = np.array([1.0, 1.0, 1.0])
        xs = []
        for _ in range(n):
            x, y, z = state
            state += np.array(
                [s * (y - x), x * (r - z) - y, x * y - b * z]
            ) * dt
            xs.append(state[0])
        return np.array(xs)

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(4000)

    print("Lorenz x-component:")
    res_l = cc_method(_lorenz_x(), max_t=30)
    print(f"  tau = {res_l['tau']}, t_w = {res_l['t_w']}, m = {res_l['m']}")

    print("White noise:")
    res_n = cc_method(noise, max_t=30)
    print(f"  tau = {res_n['tau']}, t_w = {res_n['t_w']}, m = {res_n['m']}")
