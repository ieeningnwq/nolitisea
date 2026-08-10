"""Test suite for nolitisea."""

import pathlib
import sys

# Add src/ to sys.path so that `import nolitisea` works under unittest.
# (conftest.py only works for pytest; unittest loads __init__.py instead.)
_src = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
