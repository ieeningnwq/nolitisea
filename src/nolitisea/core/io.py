"""Data input/output routines.

Replaces ``get_series`` / ``get_multi_series`` / ``search_datafile`` /
``myfgets`` from the TISEAN routines.
"""

import numpy as np


def read_series(path, column=0, length=None, skip=0):
    """Read a single column of real values from a text file.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path. ``None`` or ``"-"`` means standard input.
    column : int, default 0
        Zero-based index of the column to read.
    length : int or None
        Maximum number of points to keep (``None`` = all).
    skip : int, default 0
        Number of leading lines to ignore.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of floats.
    """
    raise NotImplementedError


def read_multi_series(path, columns, length=None, skip=0):
    """Read several columns from a text file into a 2-D array.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.
    columns : sequence of int
        Zero-based column indices to read.
    length : int or None
        Maximum number of rows to keep.
    skip : int, default 0
        Number of leading lines to ignore.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, n_columns)``.
    """
    raise NotImplementedError


def write_series(path, data, header=None):
    """Write an array to a text file, one row per line.

    Parameters
    ----------
    path : str or pathlib.Path or None
        Output file path. ``None`` means standard output.
    data : numpy.ndarray
        1-D or 2-D array to write.
    header : str or None
        Optional comment line written before the data.
    """
    raise NotImplementedError


def search_datafile(argv):
    """Locate a data file name inside an argument vector.

    Returns ``None`` when no file is given, meaning standard input.

    Parameters
    ----------
    argv : list[str]
        Command-line tokens to scan.

    Returns
    -------
    str or None
        Path to the data file, or ``None`` for stdin.
    """
    raise NotImplementedError


def myfgets(stream, max_len=1024):
    """Read one logical line, skipping comment lines starting with ``#``.

    Parameters
    ----------
    stream : file-like
        Open text stream.
    max_len : int, default 1024
        Maximum line length.

    Returns
    -------
    str or None
        The next non-comment line, or ``None`` on EOF.
    """
    raise NotImplementedError
