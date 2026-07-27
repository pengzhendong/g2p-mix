class G2PError(Exception):
    """Base exception for the package."""


class ConfigurationError(G2PError, ValueError):
    """Raised when profiles or backends are combined incorrectly."""


class BackendError(G2PError, RuntimeError):
    """Raised when a pronunciation backend cannot produce a result."""


class AlignmentError(BackendError):
    """Raised when backend output cannot be aligned to input text."""


class RenderingError(G2PError, RuntimeError):
    """Raised when a pronunciation unit cannot be rendered."""


class UnsupportedFeatureError(G2PError, NotImplementedError):
    """Raised when a requested output is not implemented by a profile."""
