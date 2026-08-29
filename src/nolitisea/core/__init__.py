"""Core infrastructure: data I/O, embedding, neighbor search, matrix and
random utilities shared by every algorithm module.

Corresponds to the ``source_c/routines/`` directory of TISEAN.
"""

from . import io, embed, neighbors, boxcount_kernel  # noqa: F401
from . import matrix, eigen, rescale, exclude  # noqa: F401
