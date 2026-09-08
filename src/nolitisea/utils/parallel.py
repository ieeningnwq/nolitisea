"""Lightweight parallel helpers built on the standard library."""

from __future__ import annotations

import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)


def parallel_map(func, items, *, n_jobs=None, backend="thread"):
    """Apply ``func`` to every item in ``items`` in parallel, preserving order.

    Parameters
    ----------
    func : callable
        Function applied to each item.  Must be picklable when
        ``backend="process"`` (e.g., a top-level function or a
        ``functools.partial`` wrapping one).
    items : sequence
        Inputs to distribute across workers.
    n_jobs : int, optional (default = None)
        Number of workers.  ``None`` or ``1`` runs sequentially in the
        current process (no executor overhead); ``-1`` uses all CPUs;
        any other positive integer caps the pool size.
    backend : {"thread", "process"}, optional
        Executor backend.  ``"thread"`` suits workers that release the
        GIL (NumPy/SciPy kernels, numba ``nogil`` code); ``"process"``
        suits pure-Python work but pays pickling and spawn costs,
        notably on Windows.

    Returns
    -------
    list
        Results in the same order as ``items``.  Exceptions raised in
        workers propagate to the caller.
    """
    items = list(items)
    if not items:
        return []

    if backend not in ("thread", "process"):
        raise ValueError('backend must be "thread" or "process".')

    if n_jobs is None or n_jobs == 1 or len(items) == 1:
        return [func(item) for item in items]

    max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    max_workers = max(1, min(max_workers, len(items)))  # pyright: ignore[reportArgumentType]

    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=max_workers) as pool:
        return list(pool.map(func, items))
