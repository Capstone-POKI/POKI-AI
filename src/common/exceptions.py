class POKIError(Exception):
    """Base exception for POKI AI service."""


class PipelineError(POKIError):
    """Raised when a domain pipeline fails."""


class ExternalServiceError(POKIError):
    """Raised when external infrastructure calls fail."""
