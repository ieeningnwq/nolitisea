"""Core infrastructure: data I/O, embedding, neighbor search, matrix and
random utilities shared by every algorithm module.

Corresponds to the ``source_c/routines/`` directory of TISEAN.
"""

from . import (  # noqa: F401
    boxcount_kernel,
    eigen,
    embed,
    exclude,
    io,
    matrix,
    neighbors,
    rescale,
)
