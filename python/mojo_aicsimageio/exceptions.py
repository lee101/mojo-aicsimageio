"""Exception types mirrored from `aicsimageio.exceptions`."""


class ConflictingArgumentsError(Exception):
    """Two arguments to the same function are in conflict."""


class InvalidDimensionOrderingError(Exception):
    """A general exception for dimension ordering or validation failures."""


class UnexpectedShapeError(Exception):
    """A general exception for shape validation failures."""


class UnsupportedFileFormatError(Exception):
    """The file extension is not one of the supported types."""
