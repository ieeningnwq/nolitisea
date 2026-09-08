"""Delay-coordinate embedding routines shared by every phase-space method."""

import numpy as np

__all__ = [
    "delay_embedding",
    "delay_vectors",
    "lag_block_delay_embed",
    "mixed_embedding",
]


def delay_embedding(series, dim, delay=1):
    """Build a delay-coordinate embedding matrix.

    Parameters
    ----------
    series : array_like
        One-dimensional scalar series.
    dim : int
        Embedding dimension ``m``.
    delay : int, default 1
        Time delay ``tau`` between coordinates.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, dim)`` where
        ``n_points = len(series) - (dim - 1) * delay``.

    Raises
    ------
    ValueError
        If the input series is not 1-dimensional or if the series
        length is insufficient for the requested embedding parameters.
    """
    arr = np.asarray(series)

    # Input validation
    if arr.ndim != 1:
        raise ValueError("Input series must be a 1-dimensional array.")

    n_points = len(arr) - (dim - 1) * delay

    if n_points <= 0:
        raise ValueError(
            f"Series length ({len(arr)}) is too short for the specified "
            f"dimension ({dim}) and delay ({delay})."
        )

    # Create the delay embedding by slicing the array with varying offsets
    # and stacking them as columns in a 2D matrix
    embedded = np.column_stack(
        [arr[i * delay : n_points + i * delay] for i in range(dim)]
    )

    return embedded


def mixed_embedding(series_list, dims, delays):
    """Build a multivariate mixed delay embedding.

    Each variable is embedded with its own dimension and delay; all rows
    share the same base index, i.e. row ``r`` holds
    ``[x_i(r), x_i(r + delay_i), ..., x_i(r + (dim_i - 1) * delay_i)]``
    for every variable ``i``.  The number of rows is the minimum value
    compatible with all variables,
    ``n_points = min_i(n_i - (dim_i - 1) * delay_i)``.

    Parameters
    ----------
    series_list : sequence of array_like
        One series per variable.
    dims : sequence of int
        Embedding dimension per variable.
    delays : sequence of int
        Delay per variable.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, sum(dims))`` where the delay
        coordinates of variable ``i`` occupy the column block
        ``sum(dims[:i]) : sum(dims[:i + 1])``.

    Raises
    ------
    ValueError
        If ``series_list``, ``dims`` and ``delays`` disagree in length,
        if any embedding dimension or delay is below 1, or if any series
        is not 1-dimensional or too short for its requested embedding.
    """
    series_list = list(series_list)
    dims = list(dims)
    delays = list(delays)
    if not (len(series_list) == len(dims) == len(delays)):
        raise ValueError(
            "series_list, dims and delays must have the same length, got "
            f"{len(series_list)}, {len(dims)} and {len(delays)}."
        )
    if not series_list:
        raise ValueError("At least one series is required.")

    blocks = []
    n_points = None
    for i, (series, dim, delay) in enumerate(zip(series_list, dims, delays)):
        if dim < 1:
            raise ValueError(f"variable {i}: embedding dimension must be >= 1.")
        if delay < 1:
            raise ValueError(f"variable {i}: delay must be >= 1.")
        try:
            block = delay_embedding(series, dim, delay)
        except ValueError as exc:
            raise ValueError(f"variable {i}: {exc}") from exc
        n_points = block.shape[0] if n_points is None else min(n_points, block.shape[0])
        blocks.append(block)

    return np.hstack([block[:n_points] for block in blocks])


def lag_block_delay_embed(data, embed, delay):
    """Build the interleaved delay-coordinate matrix (column order).

    Each row of the result is a phase-space point.  Row ``r``
    corresponds to base time ``r + (embed - 1) * delay`` and column
    ``k * n_vars + c`` holds the ``c``-th component at delay index
    ``k`` (time ``r + (embed - 1 - k) * delay``); for a flat column
    index ``i`` the component index is ``i % n_vars`` and the delay
    index is ``(i // n_vars) * delay``.

    The prefix ``E[:, :m]`` is the order-``m`` embedding whose Chebyshev
    diameter equals the running maximum over the first ``m``
    coordinates.

    Parameters
    ----------
    data : array_like
        Input data.  A 1-D array is treated as a single component;
        a 2-D array must have shape ``(n_times, n_vars)``.
    embed : int
        Embedding dimension per component.
    delay : int
        Time delay between consecutive embedding blocks.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, n_vars * embed)`` with
        ``n_points = n_times - (embed - 1) * delay``.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_times, n_vars = arr.shape
    emb_offset = (embed - 1) * delay
    n_points = n_times - emb_offset
    E = np.empty((n_points, n_vars * embed), dtype=np.float64)
    for k in range(embed):
        lo = emb_offset - k * delay
        E[:, k * n_vars : (k + 1) * n_vars] = arr[lo : lo + n_points, :]
    return E


def delay_vectors(series, embdim=None, delay=1, dims=None, increments=None):
    """Produce delay vectors.

    Build the multivariate delay-coordinate matrix
    convention: row ``r`` corresponds to base time ``n = r + max_offset``
    and holds, for every variable, its delay coordinates looking
    *backward* from ``n``::

        E[r, block_i + k] = x_i(n - offset_{i, k}),   offset_{i, 0} = 0,

    where the block of variable ``i`` occupies the columns
    ``sum(dims[:i]) : sum(dims[:i + 1])``.  Note that
    :func:`delay_embedding` and :func:`mixed_embedding` anchor the
    vectors at the *oldest* sample instead; both conventions describe
    the same set of delay vectors.

    Parameters
    ----------
    series : array_like
        Input data.  A 1-D array is a single variable; a 2-D array must
        have shape ``(n_vars, n_times)`` with one row per variable).
    embdim : int, optional
        Total embedding dimension.  Defaults to ``2``
        when ``dims`` is not given and to ``sum(dims)`` otherwise.
    delay : int, default 1
        Delay increment between successive coordinates of a variable.  Ignored when ``increments`` is given.
    dims : sequence of int, optional
        Embedding dimension per variable.  When given,
        ``embdim`` (if provided) must equal ``sum(dims)``.
    increments : sequence of int, optional
        Delay increment between successive coordinates, consumed across
        variables in order.  Must contain
        ``embdim - n_vars`` entries and overrides ``delay``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, embdim)`` with
        ``n_points = n_times - max(offsets)``.

    Raises
    ------
    ValueError
        For invalid parameter combinations or a series too short for
        the requested delay offsets.
    """
    arr = np.asarray(series)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(
            "series must be a 1-D array or a 2-D array of shape (n_vars, n_times)"
        )
    n_vars, length = arr.shape

    # Per-variable embedding dimensions.
    if dims is None:
        if embdim is None:
            embdim = 2  # default
        if embdim < 1:
            raise ValueError("embdim must be >= 1")
        if embdim % n_vars:
            raise ValueError(
                f"embdim={embdim} is not a multiple of the {n_vars} "
                "variable(s); supply per-variable dims instead"
            )
        dims = [embdim // n_vars] * n_vars
    else:
        if np.ndim(dims) == 0:
            raise ValueError("dims must be a sequence with one entry per variable")
        dims = [int(d) for d in dims]
        if len(dims) != n_vars:
            raise ValueError(
                f"dims has {len(dims)} entries but the series has {n_vars} variable(s)"
            )
        if any(d < 1 for d in dims):
            raise ValueError("every entry of dims must be >= 1")
        if embdim is None:
            embdim = sum(dims)
        elif embdim != sum(dims):
            raise ValueError(f"embdim={embdim} does not match sum(dims)={sum(dims)}")

    # Cumulative delay offset of every coordinate.
    offsets = []
    if increments is None:
        if delay < 1:
            raise ValueError("delay must be >= 1")
        for m in dims:
            offsets.append([k * delay for k in range(m)])
    else:
        if np.ndim(increments) == 0:
            raise ValueError(
                "increments must be a sequence with embdim - n_vars entries"
            )
        increments = [int(d) for d in increments]
        if len(increments) != embdim - n_vars:
            raise ValueError(
                f"increments must contain embdim - n_vars = "
                f"{embdim - n_vars} entries, got {len(increments)}"
            )
        if any(d < 1 for d in increments):
            raise ValueError("every entry of increments must be >= 1")
        pos = 0
        for m in dims:
            offs = [0]
            for _ in range(m - 1):
                offs.append(offs[-1] + increments[pos])
                pos += 1
            offsets.append(offs)

    max_offset = max(max(offs) for offs in offsets)
    n_points = length - max_offset
    if n_points <= 0:
        raise ValueError(
            f"series length ({length}) is too short for the requested "
            f"delay offsets (largest {max_offset})"
        )

    embedded = np.empty((n_points, embdim), dtype=arr.dtype)
    col = 0
    for component, offs in zip(arr, offsets):
        for off in offs:
            start = max_offset - off
            embedded[:, col] = component[start : start + n_points]
            col += 1
    return embedded
