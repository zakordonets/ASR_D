class AppError(Exception):
    """Base application error."""


class InputResolutionError(AppError):
    """Raised when an input path cannot be resolved."""


class ProviderError(AppError):
    """Raised for provider construction or runtime errors."""


class ProviderDependencyError(ProviderError):
    """Raised when optional dependencies for a provider are missing."""


class PreprocessingError(AppError):
    """Raised when media preprocessing fails."""


class ExportError(AppError):
    """Raised when export fails."""


class CombineProcessingError(AppError):
    """Raised when combined mode fails as a whole."""

