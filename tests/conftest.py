"""Test fixtures.

`aicsimageio` is not in the shared test venv, so `reference.py` resolves it:
first as an installed package, then from a cached wheel extraction, then by
fetching the wheel from PyPI. The tests fall back to an independent NumPy
reference of the documented transform semantics when none of that works, and
report which reference they used.
"""

import os
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-aicsimageio.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-aicsimageio.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )
