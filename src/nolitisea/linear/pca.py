"""Principal component analysis (TISEAN ``pca``)."""

import numpy as np


def pca(data, n_components=None):
    """Perform PCA on multivariate data.

    Parameters
    ----------
    data : array_like
        Array of shape ``(n_samples, n_vars)``.
    n_components : int or None
        Number of components to keep (``None`` = all).

    Returns
    -------
    tuple
        ``(components, values, projection)`` where ``components`` has
        shape ``(n_components, n_vars)``, ``values`` are the variances,
        and ``projection`` is the data projected onto the components.
    """
    raise NotImplementedError


def svd_truncate(data, n_components):
    """Denoise ``data`` by truncated SVD reconstruction.

    Parameters
    ----------
    data : array_like
        Input 2-D array.
    n_components : int
        Number of singular values to keep.

    Returns
    -------
    numpy.ndarray
        Reconstructed (denoised) array.
    """
    raise NotImplementedError


def run(argv=None):
    """CLI entry point: principal component analysis.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector.
    """
    raise NotImplementedError
