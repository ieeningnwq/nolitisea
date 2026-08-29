from __future__ import annotations

from functools import partial

import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial import cKDTree

from nolitisea.core.embed import delay_embedding
from nolitisea.utils.parallel import parallel_map


def _integral_from_tree(tree, n_points: int, eps: float) -> float:
    """Compute the correlation integral from a prebuilt KDTree."""
    # count_neighbors counts ordered pairs including self-pairs (i, i).
    # Subtract self-pairs and divide by 2 to obtain unordered i < j pairs.
    raw_counts = tree.count_neighbors(tree, eps, p=np.inf)
    counts = (raw_counts - n_points) // 2
    valid_total = (n_points * (n_points - 1)) // 2

    return counts / valid_total if valid_total > 0 else 0.0


def _cc_single_t(ts, t, m_values, r_values):
    """Compute (S_mean, delta_S, S_cor) for a single candidate delay t.

    Each KDTree is built once per (sub-series, dimension) and reused across
    all radii, since the tree depends only on the embedding, not on r.
    """
    # S_mr[a, b] = S(m_a, r_b, t), averaged over the t sub-series.
    S_mr = np.zeros((len(m_values), len(r_values)))

    for s in range(t):
        # The s-th disjoint sub-series: x_s, x_{s+t}, x_{s+2t}, ...
        sub = np.ascontiguousarray(ts[s::t])
        if len(sub) <= max(m_values):
            continue

        tree_1d = cKDTree(sub.reshape(-1, 1))
        c1_vals = [_integral_from_tree(tree_1d, len(sub), r) for r in r_values]

        # Reconstruct *inside the sub-series* with delay = 1, because the
        # sub-series elements are already spaced by t in the original data.
        for a, m in enumerate(m_values):
            emb = delay_embedding(sub, m, 1)
            tree_md = cKDTree(emb)
            for b, r in enumerate(r_values):
                C_m = _integral_from_tree(tree_md, len(emb), r)
                S_mr[a, b] += C_m - c1_vals[b] ** m

    S_mr /= max(t, 1)

    # S_mean(t): average of S(m, r, t) over all m and all r.
    s_mean = S_mr.mean()
    # delta_S(t): for each m take max - min over r, then average over m.
    delta_s = (S_mr.max(axis=1) - S_mr.min(axis=1)).mean()
    # S_cor(t) = delta_S(t) + |S_mean(t)|.
    s_cor = delta_s + abs(s_mean)
    return s_mean, delta_s, s_cor


def cc_method(
    ts: np.ndarray,
    max_t: int = 200,
    n_jobs: int | None = None,
    backend: str = "thread",
) -> dict:
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
    n_jobs : int, optional (default = None)
        Workers for the per-delay loop.  ``None``/``1`` runs
        sequentially; ``-1`` uses all CPUs (see
        ``nolitisea.utils.parallel.parallel_map``).
    backend : {"thread", "process"}, optional (default = "thread")
        Executor backend for the per-delay loop.  ``"thread"`` is
        preferred: cKDTree kernels release the GIL and no data
        serialization is needed.

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

    worker = partial(_cc_single_t, ts, m_values=m_values, r_values=r_values)
    per_t = parallel_map(worker, t_range, n_jobs=n_jobs, backend=backend)
    S_mean, delta_S, S_cor = (np.asarray(col, dtype=np.float64) for col in zip(*per_t))

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
            state += np.array([s * (y - x), x * (r - z) - y, x * y - b * z]) * dt
            xs.append(state[0])
        return np.array(xs)

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(4000)

    print("Lorenz x-component:")
    res_l = cc_method(_lorenz_x(), max_t=30, n_jobs=-1)
    print(f"  tau = {res_l['tau']}, t_w = {res_l['t_w']}, m = {res_l['m']}")

    print("White noise:")
    res_n = cc_method(noise, max_t=30, n_jobs=-1)
    print(f"  tau = {res_n['tau']}, t_w = {res_n['t_w']}, m = {res_n['m']}")
