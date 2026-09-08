"""Fixed-mass information dimension D1."""

from functools import partial

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.utils.parallel import parallel_map

__all__ = ["c1"]

_TINY = 1e-20
# Bounds the candidate-neighbour work per chunk independently of
# ``n_jobs`` so serial and parallel reductions group identically.
_NEIGH_BUDGET = 1 << 20
_MAX_SWEEPS = 64

# Digamma table (values for k = 0..20).
_PSI_TABLE = (
    0.0,
    -0.57721566490, 0.42278433509, 0.92278433509, 1.25611766843,
    1.50611766843, 1.70611766843, 1.87278433509, 2.01564147795,
    2.14064147795, 2.25175258906, 2.35175258906, 2.44266167997,
    2.52599501330, 2.60291809023, 2.67434666166, 2.74101332832,
    2.80351332832, 2.86233685773, 2.91789241329, 2.97052399224,
)


def _psi(k):
    """Digamma approximation."""
    if k <= 20:
        return _PSI_TABLE[k]
    return float(np.log(k) - 1.0 / (2.0 * k))


def _embed_c1(data, m, delay):
    """Order-``m`` embedding in the oldest-first coordinate order.

    Column blocks run from the oldest delay block to the newest, each
    block holding all components in file order; the first ``m`` columns
    are exactly the coordinates entering the distance computation.

    Returns ``(E, mt)`` where ``E`` has shape ``(n_points, m)`` and
    ``mt = (m - 1) // n_vars + 1`` is the delay span in steps.
    """
    n_vars = data.shape[1]
    mt = (m - 1) // n_vars + 1
    E = lag_block_delay_embed(data, mt, delay)
    n_points = E.shape[0]
    # ``lag_block_delay_embed`` orders blocks newest-first; reverse the
    # block axis without permuting components inside a block.
    E = E.reshape(n_points, mt, n_vars)[:, ::-1, :].reshape(
        n_points, mt * n_vars
    )
    return np.ascontiguousarray(E[:, :m]), mt


def _c1_sweep_worker(centers, tree, eps, ncomp_eff, theiler, k):
    """One ``eps`` sweep over a chunk of centers.

    For every center index, return the log of the k-th smallest
    Chebyshev distance among the eligible neighbours within ``eps``
    (strictly), or ``None`` when fewer than ``k`` neighbours were
    found (the center is retried in the next sweep with ``eps``
    multiplied by ``sqrt(2)``).
    """
    data = tree.data
    candidate_lists = tree.query_ball_point(data[centers], r=eps, p=np.inf)
    out = []
    for n, neigh in zip(centers, candidate_lists):
        neigh = np.asarray(neigh, dtype=np.intp)
        if neigh.size:
            md = np.abs(neigh - n) % ncomp_eff
            neigh = neigh[(md > theiler) & (md < ncomp_eff - theiler)]
        if neigh.size:
            cheb = np.abs(data[neigh] - data[n]).max(axis=1)
            cheb = cheb[cheb <= eps]
        else:
            cheb = np.empty(0)
        if cheb.size < k:
            out.append(None)
        else:
            kth = float(np.partition(cheb, k - 1)[k - 1])
            out.append(float(np.log(max(kth, _TINY))))
    return out


def _d1_step(E, tree, sd, m, delay, mt, ncmin, theiler, kmax, pr, pl,
             rng, n_jobs, backend):
    """One fixed-mass ladder step.

    Returns ``(pln, eln)``; ``eln`` is ``None`` when the requested mass
    equals the previous one (``k == kpr``), in which case the search is
    skipped and the caller skips the output.
    """
    ncomp = E.shape[0]
    N0 = ncomp - 2 * theiler - 1

    kpr = int(np.exp(pr) * N0) + 1
    k = int(np.exp(pl) * N0) + 1
    if k > kmax:
        # ncomp = real(N0) * real(kmax) / k + 2 * nmin + 1
        # assigned to an INTEGER variable (truncation).
        N = int(N0 * kmax / k)
        if N < 1:
            raise ValueError(
                f"kmax={kmax} is too small for the requested mass: the "
                "effective point count would drop below one"
            )
        k = kmax
        ncomp_eff = N + 2 * theiler + 1
    else:
        N = N0
        ncomp_eff = ncomp
        # Guard the degenerate ladder end pl == 0 (k = N + 1 would leave
        # every center short of neighbours forever).
        k = min(k, N)
    pln = _psi(k) - np.log(N)
    if k == kpr:
        return pln, None

    if ncmin > ncomp:
        raise ValueError(
            f"n_centers={ncmin} exceeds the {ncomp} available center "
            f"points for embedding dimension {m}"
        )
    n_skip = (mt - 1) * delay
    if ncmin - n_skip < 1:
        raise ValueError(
            f"n_centers={ncmin} must exceed the embedding loss "
            f"(mt - 1) * delay = {n_skip} for embedding dimension {m}"
        )

    centers = rng.permutation(ncomp)[:ncmin]
    eps = float(np.exp(pln / m) * sd)

    contribs = []
    queue = centers
    for _sweep in range(_MAX_SWEEPS):
        chunk = max(1, _NEIGH_BUDGET // max(ncomp, 1))
        chunks = [
            queue[i:i + chunk] for i in range(0, len(queue), chunk)
        ]
        worker = partial(
            _c1_sweep_worker, tree=tree, eps=eps, ncomp_eff=ncomp_eff,
            theiler=theiler, k=k,
        )
        results = parallel_map(worker, chunks, n_jobs=n_jobs, backend=backend)
        next_queue = []
        for chunk_centers, chunk_results in zip(chunks, results):
            for center, result in zip(chunk_centers, chunk_results):
                if result is None:
                    next_queue.append(center)
                else:
                    contribs.append(result)
        if not next_queue:
            break
        queue = np.asarray(next_queue, dtype=np.intp)
        eps = eps * np.sqrt(2.0)
    else:
        raise RuntimeError(
            "fixed-mass neighbour search did not converge after "
            f"{_MAX_SWEEPS} eps sweeps (k={k}, ncomp={ncomp}); the "
            "requested mass may exceed the number of eligible neighbours"
        )

    eln = float(np.sum(contribs)) / (ncmin - n_skip)
    return pln, eln


def c1(
    series,
    embed_min=1,
    embed_max=10,
    delay=1,
    theiler=0,
    n_centers=None,
    resolution=2.0,
    kmax=100,
    seed=0,
    n_jobs=None,
    backend="thread",
):
    """Fixed-mass information dimension D1.

    Parameters
    ----------
    series : array_like
        Input data.  A 1-D array is a single component; a 2-D array
        must have shape ``(n_times, n_vars)`` with one column per
        component.  ``embed_min``..``embed_max`` are *total* embedding
        dimensions spanning all components.
    embed_min, embed_max : int
        Minimal / maximal total embedding dimensions.
    delay : int
        Time delay.
    theiler : int
        Minimal time separation of neighbour pairs.
        Pairs with modular separation ``<= theiler`` are excluded.
    n_centers : int or None
        Number of randomly drawn center points per ladder step.
        Uses the largest count compatible with ``embed_max``.
    resolution : float
        Ladder resolution, values per octave.
    kmax : int
        Maximal number of neighbours.
        Larger requested masses are clamped by reducing the effective
        point count.
    seed : int
        Seed for the random center permutation.
    n_jobs : int, optional (default = None)
        Workers for the per-sweep neighbour queries.  ``None``/``1``
        runs sequentially; ``-1`` uses all CPUs (see
        ``nolitisea.utils.parallel.parallel_map``).  Results are
        bitwise identical for every ``n_jobs``/``backend``.
    backend : {"thread", "process"}, optional (default = "thread")
        Executor backend.

    Returns
    -------
    dict
        ``"embed"`` : int array of total embedding dimensions.
        ``"eps"`` : list of float arrays, one per embedding dimension;
        the geometric mean k-th neighbour distances (radii).
        ``"mass"`` : list of float arrays, one per embedding dimension;
        the Grassberger-corrected masses.  D1 is the slope of
        ``log(mass)`` versus ``log(eps)`` in the scaling region.

    Raises
    ------
    ValueError
        For invalid parameters, a series too short for the embedding
        ladder, a constant series, or center counts incompatible with
        the embedding loss.
    RuntimeError
        If the neighbour search fails to converge within the sweep cap.
    """
    data = np.asarray(series, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim != 2:
        raise ValueError(
            "series must be a 1-D array or a 2-D array of shape "
            "(n_times, n_vars)"
        )
    n, n_vars = data.shape

    if embed_min < 1:
        raise ValueError(f"embed_min must be >= 1, got {embed_min}")
    if embed_max < embed_min:
        raise ValueError(
            f"embed_max ({embed_max}) must be >= embed_min ({embed_min})"
        )
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if theiler < 0:
        raise ValueError(f"theiler must be >= 0, got {theiler}")
    if kmax < 1:
        raise ValueError(f"kmax must be >= 1, got {kmax}")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(
            f"resolution must be finite and > 0, got {resolution}"
        )
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    col = data[:, 0]
    sd = float(np.sqrt(((col - col.mean()) ** 2).mean()))
    if sd <= 0.0:
        raise ValueError(
            "series is constant; the fixed-mass eps ladder requires "
            "a non-constant first column"
        )

    if n_centers is None:
        mt_max = (embed_max - 1) // n_vars + 1
        n_centers = n - (mt_max - 1) * delay
    if n_centers < 1:
        raise ValueError(
            f"n_centers must be >= 1, got {n_centers}"
        )

    rng = np.random.default_rng(seed)
    resl = np.log(2.0) / resolution

    embeds = []
    eps_out = []
    mass_out = []
    for m in range(embed_min, embed_max + 1):
        span = (m - 1) * delay
        if n - span < 1:
            raise ValueError(
                f"embedding dimension m={m} needs n - (m - 1) * delay "
                f">= 1, got n={n}, delay={delay}"
            )
        pl_start = np.log(1.0 / (n - span))
        E, mt = _embed_c1(data, m, delay)
        ncomp = E.shape[0]
        if ncomp - 2 * theiler - 1 < 1:
            raise ValueError(
                f"embedding dimension m={m} leaves only "
                f"{ncomp} - 2 * {theiler} - 1 eligible neighbour(s); "
                "reduce theiler or the embedding dimension"
            )
        tree = cKDTree(E)

        pr = 0.0
        eps_list = []
        mass_list = []
        i = 0
        while True:
            pl = pl_start + i * resl
            if pl > 0.0:
                break
            pln, eln = _d1_step(
                E, tree, sd, m, delay, mt, n_centers, theiler, kmax,
                pr, pl, rng, n_jobs, backend,
            )
            # The output is skipped whenever the corrected mass repeats
            # the previous one (this also covers the clamped tail of the
            # ladder, which recomputes the same mass every step).
            if eln is not None and pln != pr:
                eps_list.append(float(np.exp(eln)))
                mass_list.append(float(np.exp(pln)))
                pr = pln
            i += 1

        embeds.append(m)
        eps_out.append(np.asarray(eps_list, dtype=np.float64))
        mass_out.append(np.asarray(mass_list, dtype=np.float64))

    return {
        "embed": np.asarray(embeds, dtype=np.int64),
        "eps": eps_out,
        "mass": mass_out,
    }
