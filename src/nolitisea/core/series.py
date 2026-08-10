"""Lightweight time series container used across the package."""

import numpy as np


class TimeSeries:
    """Container wrapping a 1-D or 2-D array with metadata.

    Attributes
    ----------
    data : numpy.ndarray
        Underlying values, shape ``(n,)`` or ``(n, n_cols)``.
    columns : list[str] or None
        Optional names for each column.
    dt : float
        Sampling interval.
    name : str or None
        Optional series label.
    """

    def __init__(self, data, columns=None, dt=1.0, name=None):
        """Initialise the container.

        Parameters
        ----------
        data : array_like
            1-D or 2-D real array.
        columns : sequence of str or None
            Column names.
        dt : float, default 1.0
            Sampling interval.
        name : str or None
            Optional label for the series.
        """
        raise NotImplementedError

    @property
    def n_cols(self):
        """int: number of variables (columns)."""
        raise NotImplementedError

    @property
    def length(self):
        """int: number of samples."""
        raise NotImplementedError

    def column(self, idx):
        """Return a 1-D view of the ``idx``-th column.

        Parameters
        ----------
        idx : int
            Zero-based column index.

        Returns
        -------
        numpy.ndarray
            One-dimensional view of the requested column.
        """
        raise NotImplementedError

    def to_array(self):
        """Return the underlying data as a 2-D numpy array.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(length, n_cols)``.
        """
        raise NotImplementedError
