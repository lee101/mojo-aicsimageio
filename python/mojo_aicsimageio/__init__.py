"""Mojo port of the compute surface of `aicsimageio`.

`aicsimageio` is a microscopy image IO library: readers for Bio-Formats, CZI,
LIF, ND2, DV, OME-TIFF and OME-Zarr, writers for OME-TIFF and OME-Zarr, and
OME metadata parsing. None of that has a numeric inner loop, so none of it is
ported here.

What is ported is `aicsimageio.transforms`, the dimension engine that turns a
reader stack into the `(B, S, T, C, Z, Y, X)` shape a user asked for. It is the
only place the package moves pixels at array scale, and it is what this package
compiles into a single Mojo gather.

>>> import numpy as np, mojo_aicsimageio
>>> stack = np.arange(3 * 4 * 8 * 8, dtype=np.uint16).reshape(3, 4, 8, 8)
>>> plane = mojo_aicsimageio.reshape_data(stack, "ZYXC", "YXC", Z=2)
>>> plane.shape
(8, 8, 4)
"""

from .exceptions import (
    ConflictingArgumentsError,
    InvalidDimensionOrderingError,
    UnexpectedShapeError,
    UnsupportedFileFormatError,
)
from .transforms import (
    reduce_to_slice,
    reshape_data,
    transpose_to_dims,
)

__all__ = [
    "ConflictingArgumentsError",
    "InvalidDimensionOrderingError",
    "UnexpectedShapeError",
    "UnsupportedFileFormatError",
    "reduce_to_slice",
    "reshape_data",
    "transpose_to_dims",
    "kernels",
]

__version__ = "0.1.0"


def kernels():
    """Return the exported kernel symbols, for introspection and tests."""
    from . import _lib

    return sorted(s for s in dir(_lib.lib) if not s.startswith("_"))
