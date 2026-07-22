class RiceTillerError(Exception):
    """Base class for expected pipeline failures."""


class ConfigurationError(RiceTillerError):
    """Raised when configuration or model metadata is invalid."""


class PointCloudIOError(RiceTillerError):
    """Raised when a point-cloud file cannot be read or written."""


class QualityError(RiceTillerError):
    """Raised when input data cannot safely be analysed."""


class CheckpointError(RiceTillerError):
    """Raised when a model checkpoint does not match its metadata."""

